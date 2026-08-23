"""Hermetic PCAR-013 public-surface manifest tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.architecture_ir import (
    ArchitectureEdge,
    ArchitectureIR,
    ArchitectureNode,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import (
    Confidence,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    SourceSpan,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.graph_builder import (
    extract_architecture_graph,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.public_surface import (
    CLOSED_ACCIDENTAL_KINDS,
    CLOSED_CONSUMER_EVIDENCE,
    CLOSED_EXPORT_CLASSES,
    CLOSED_IMPORT_EFFECTS,
    CLOSED_IMPORT_LAZINESS,
    CLOSED_PROJECTION_MISMATCHES,
    CLOSED_PROJECTIONS,
    CLOSED_REMOVAL_BLOCKERS,
    CURRENT_SURFACE_BINDINGS,
    DEFAULT_FRESHNESS,
    EFFECT_CLASS,
    EXTRACTOR_IDENTITY,
    MANIFEST_CAN_AUTHORIZE_REMOVAL,
    MANIFEST_CAN_CHANGE_PUBLIC_API,
    MANIFEST_CAN_DEPRECATE,
    MANIFEST_CAN_PROMOTE_INTERNAL,
    PUBLIC_SURFACE_EVIDENCE,
    PUBLIC_SURFACE_SCHEMA,
    PUBLIC_SURFACE_VERSION,
    REQUIRED_EXPORT_CLASSES,
    REQUIRED_PROJECTIONS,
    TASK_ID,
    AccidentalExportKind,
    ConsumerEvidenceKind,
    ConsumerReference,
    DiscoveryOrigin,
    ExportClassification,
    ExportDeclaration,
    ImportEffectKind,
    ImportLaziness,
    ProjectionBinding,
    ProjectionKind,
    ProjectionMismatchKind,
    PublicSurfaceAuthorityError,
    PublicSurfaceError,
    PublicSurfaceManifest,
    RemovalBlockerKind,
    RemovalEvidence,
    StablePublicSymbolRecord,
    assess_imports_from_sources,
    build_public_surface_manifest,
    classify_export,
    classify_exports,
    detect_projection_mismatch,
    discover_exports_from_sources,
    refuse_deprecation,
    refuse_public_promotion,
    refuse_removal,
    unknown_consumers_block_removal,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
_FRESHNESS = "pcar-013-fixture"
_ROOT = Path(__file__).resolve().parents[3]

_BOOM = """raise RuntimeError("imported")

__all__ = ["surviving", "_leaked"]

def surviving() -> int:
    return 1
"""

_PKG_INIT = """from .ops import run_task, Handler, compat_run, sim_probe
from .internal import helper
from .ops import SCHEMA as OPS_SCHEMA

__all__ = ["run_task", "Handler", "compat_run", "sim_probe", "_leaked", "helper"]
"""

_PKG_OPS = '''"""Canonical operations."""
SCHEMA = {"type": "object"}

def run_task(name: str) -> dict:
    return {"name": name}

class Handler:
    def execute(self, name: str) -> str:
        return name

def compat_run(name: str) -> dict:
    return run_task(name)

def sim_probe() -> str:
    return "simulated"
'''

_PKG_INTERNAL = """def helper() -> int:
    return 1

def secret_helper() -> int:
    return 2
"""

_PKG_CLI = '''"""CLI projection."""
COMMANDS = ("run_task", "invented")

def run_task() -> None:
    from .ops import run_task as impl
    return impl("cli")
'''

_PKG_MCP = '''"""MCP projection."""
AGENT_SUPERVISOR_OPERATION_TOOLS = {"run_task": run_task}

def run_task() -> None:
    from .ops import run_task as impl
    return impl("mcp")
'''

_EAGER_SIDE_EFFECT = """import os
print("loading")
open("state.json", "w").write("x")

def use() -> None:
    return None
"""

_LAZY_CLEAN = """def use() -> None:
    from .ops import run_task
    return run_task("ok")
"""

_STAR = """from .ops import *

__all__ = ["*"]
"""

_PYPROJECT = """[project.scripts]
pkg-run = "pkg.ops:run_task"
"""


def _span(path: str, start: int, end: int | None = None) -> SourceSpan:
    return SourceSpan(path, start, start if end is None else end)


def _fact(
    path: str,
    start: int,
    *,
    confidence: Confidence = Confidence.EXACT,
    end: int | None = None,
) -> SourceFactIdentity:
    return SourceFactIdentity(
        extractor_identity="pcar-013-fixture",
        span=_span(path, start, end),
        confidence=confidence,
        freshness=_FRESHNESS,
        repository_tree=_TREE,
    )


def _decl(
    symbol: str,
    classification: ExportClassification,
    path: str,
    start: int,
    *,
    module: str = "pkg",
    owner: str = "",
    schema: str = "",
    version: str = "",
    effects: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
    authority: str = "",
    tests: tuple[str, ...] = (),
    proofs: tuple[str, ...] = (),
    consumers: tuple[str, ...] = (),
    consumer_evidence: ConsumerEvidenceKind = ConsumerEvidenceKind.UNKNOWN,
    projections: tuple[ProjectionKind, ...] = (),
) -> ExportDeclaration:
    return ExportDeclaration(
        symbol=symbol,
        classification=classification,
        provenance=_fact(path, start),
        module=module,
        owner=owner,
        schema=schema,
        version=version,
        effects=effects,
        errors=errors,
        authority=authority,
        tests=tests,
        proofs=proofs,
        consumers=consumers,
        consumer_evidence=consumer_evidence,
        projections=projections,
    )


def _stable_decl() -> ExportDeclaration:
    return _decl(
        "run_task",
        ExportClassification.STABLE,
        "pkg/ops.py",
        4,
        module="pkg.ops",
        owner="pkg.ops.Handler",
        schema="pkg/run-task@1",
        version="1",
        effects=("read_state",),
        errors=("InvalidName",),
        authority="pkg.ops.Handler",
        tests=("test/test_ops.py::test_run_task",),
        proofs=("proofs/run_task.proof.json",),
        consumers=("pkg.cli.run_task", "pkg.mcp.run_task"),
        consumer_evidence=ConsumerEvidenceKind.KNOWN,
        projections=(ProjectionKind.PYTHON, ProjectionKind.CLI, ProjectionKind.MCP),
    )


def _all_class_declarations() -> tuple[ExportDeclaration, ...]:
    return (
        _stable_decl(),
        _decl(
            "Handler",
            ExportClassification.PROVISIONAL,
            "pkg/ops.py",
            8,
            module="pkg.ops",
        ),
        _decl(
            "helper",
            ExportClassification.INTERNAL,
            "pkg/internal.py",
            1,
            module="pkg.internal",
        ),
        _decl(
            "secret_helper",
            ExportClassification.INTERNAL,
            "pkg/internal.py",
            4,
            module="pkg.internal",
        ),
        _decl(
            "compat_run",
            ExportClassification.COMPATIBILITY,
            "pkg/ops.py",
            12,
            module="pkg.ops",
        ),
        _decl(
            "legacy_run",
            ExportClassification.DEPRECATED,
            "pkg/ops.py",
            12,
            module="pkg.ops",
            consumers=("pkg.old_client",),
            consumer_evidence=ConsumerEvidenceKind.KNOWN,
        ),
        _decl(
            "sim_probe",
            ExportClassification.SIMULATION,
            "pkg/ops.py",
            16,
            module="pkg.ops",
        ),
        _decl(
            "test_run_task",
            ExportClassification.TEST_ONLY,
            "test/test_ops.py",
            4,
            module="test.test_ops",
        ),
        _decl(
            "_leaked",
            ExportClassification.ACCIDENTALLY_PUBLIC,
            "pkg/__init__.py",
            5,
            module="pkg",
        ),
    )


def _parity_bindings() -> tuple[ProjectionBinding, ...]:
    return (
        ProjectionBinding(
            operation="run_task",
            projection=ProjectionKind.PYTHON,
            schema="pkg/run-task@1",
            version="1",
            provenance=_fact("pkg/ops.py", 4),
            effects=("read_state",),
            errors=("InvalidName",),
        ),
        ProjectionBinding(
            operation="run_task",
            projection=ProjectionKind.CLI,
            schema="pkg/run-task@1",
            version="1",
            provenance=_fact("pkg/cli.py", 4),
            effects=("read_state",),
            errors=("InvalidName",),
        ),
        ProjectionBinding(
            operation="run_task",
            projection=ProjectionKind.MCP,
            schema="pkg/run-task@1",
            version="1",
            provenance=_fact("pkg/mcp.py", 4),
            effects=("read_state",),
            errors=("InvalidName",),
        ),
    )


def _sources() -> dict[str, str]:
    return {
        "pkg/__init__.py": _PKG_INIT,
        "pkg/ops.py": _PKG_OPS,
        "pkg/internal.py": _PKG_INTERNAL,
        "pkg/cli.py": _PKG_CLI,
        "pkg/mcp.py": _PKG_MCP,
        "pkg/eager.py": _EAGER_SIDE_EFFECT,
        "pkg/lazy.py": _LAZY_CLEAN,
        "pkg/star.py": _STAR,
        "pkg/boom.py": _BOOM,
        "pyproject.toml": _PYPROJECT,
        "test/test_ops.py": "from pkg.ops import run_task\n\ndef test_run_task() -> None:\n    assert run_task('x')['name'] == 'x'\n",
    }


def _manifest(**kwargs: object) -> PublicSurfaceManifest:
    defaults = {
        "declarations": _all_class_declarations(),
        "projections": _parity_bindings(),
        "consumers": (
            ConsumerReference(
                symbol="run_task",
                consumer="pkg.cli.run_task",
                provenance=_fact("pkg/cli.py", 6),
                kind="call",
            ),
            ConsumerReference(
                symbol="run_task",
                consumer="pkg.mcp.run_task",
                provenance=_fact("pkg/mcp.py", 6),
                kind="call",
            ),
        ),
        "sources": _sources(),
        "repository_tree": _TREE,
        "freshness": _FRESHNESS,
    }
    defaults.update(kwargs)
    return build_public_surface_manifest(**defaults)  # type: ignore[arg-type]


def test_closed_export_classes_and_evidence_pins() -> None:
    assert PUBLIC_SURFACE_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/public-surface-manifest@1"
    )
    assert PUBLIC_SURFACE_SCHEMA.endswith("public-surface-manifest@1")
    assert PUBLIC_SURFACE_VERSION == 1
    assert PUBLIC_SURFACE_EVIDENCE == "pcar/public-surface-manifest@1"
    assert EXTRACTOR_IDENTITY == "pcar-013-public-surface-manifest"
    assert TASK_ID == "PCAR-013"
    assert DEFAULT_FRESHNESS == "pcar-013-public-surface"
    assert EFFECT_CLASS == "read_only_analysis"
    assert MANIFEST_CAN_AUTHORIZE_REMOVAL is False
    assert MANIFEST_CAN_PROMOTE_INTERNAL is False
    assert MANIFEST_CAN_DEPRECATE is False
    assert MANIFEST_CAN_CHANGE_PUBLIC_API is False
    assert tuple(item.value for item in REQUIRED_EXPORT_CLASSES) == (
        "stable",
        "provisional",
        "internal",
        "compatibility",
        "deprecated",
        "simulation",
        "test_only",
        "accidentally_public",
    )
    assert CLOSED_EXPORT_CLASSES == {item.value for item in ExportClassification}
    assert CLOSED_PROJECTIONS == {"python", "cli", "mcp"}
    assert tuple(item.value for item in REQUIRED_PROJECTIONS) == ("python", "cli", "mcp")
    assert CLOSED_CONSUMER_EVIDENCE == {"known", "unknown"}
    assert CLOSED_IMPORT_LAZINESS == {"lazy", "eager", "unknown"}
    assert CLOSED_IMPORT_EFFECTS == {
        "none",
        "filesystem",
        "process",
        "network",
        "mutation",
        "exception",
        "unknown",
    }
    assert CLOSED_ACCIDENTAL_KINDS == {
        "undeclared_public",
        "internal_reexport",
        "private_name_in_all",
        "star_reexport",
        "wildcard_surface",
    }
    assert CLOSED_PROJECTION_MISMATCHES == {
        "missing_python",
        "missing_cli",
        "missing_mcp",
        "schema_mismatch",
        "version_mismatch",
        "effect_mismatch",
        "error_mismatch",
        "semantic_invention",
    }
    assert RemovalBlockerKind.UNKNOWN_CONSUMERS.value in CLOSED_REMOVAL_BLOCKERS
    with pytest.raises(ValueError):
        ExportClassification("public_enough")
    with pytest.raises(ValueError):
        ProjectionKind("http")
    with pytest.raises(ValueError):
        AccidentalExportKind("maybe")


def test_all_export_classes_are_classified_with_provenance() -> None:
    manifest = _manifest()
    covered = {item.classification for item in manifest.exports}
    assert covered >= set(REQUIRED_EXPORT_CLASSES)
    by_symbol = {item.symbol: item.classification for item in manifest.exports}
    assert by_symbol["run_task"] is ExportClassification.STABLE
    assert by_symbol["Handler"] is ExportClassification.PROVISIONAL
    assert by_symbol["secret_helper"] is ExportClassification.INTERNAL
    assert by_symbol["compat_run"] is ExportClassification.COMPATIBILITY
    assert by_symbol["legacy_run"] is ExportClassification.DEPRECATED
    assert by_symbol["sim_probe"] is ExportClassification.SIMULATION
    assert by_symbol["test_run_task"] is ExportClassification.TEST_ONLY
    assert by_symbol["_leaked"] is ExportClassification.ACCIDENTALLY_PUBLIC
    assert by_symbol["helper"] is ExportClassification.ACCIDENTALLY_PUBLIC
    for record in manifest.exports:
        assert record.classification in ExportClassification
        assert record.provenance.extractor_identity
        assert record.provenance.span.path
        assert record.provenance.repository_tree == _TREE
        assert record.provenance.freshness == _FRESHNESS
        assert len(record.origins) >= 1
    assert manifest.export_closure_complete is True
    classes = {record.qualified_name: record.classification for record in manifest.exports}
    assert len(classes) == len(manifest.exports)


def test_stable_metadata_is_complete_and_round_trips() -> None:
    manifest = _manifest()
    stable = manifest.stable_record("run_task")
    assert isinstance(stable, StablePublicSymbolRecord)
    assert stable.owner == "pkg.ops.Handler"
    assert stable.schema == "pkg/run-task@1"
    assert stable.version == "1"
    assert stable.effects == ("read_state",)
    assert stable.errors == ("InvalidName",)
    assert stable.authority == "pkg.ops.Handler"
    assert stable.tests == ("test/test_ops.py::test_run_task",)
    assert stable.proofs == ("proofs/run_task.proof.json",)
    assert stable.consumers == ("pkg.cli.run_task", "pkg.mcp.run_task")
    payload = stable.to_dict()
    restored = StablePublicSymbolRecord.from_mapping(payload)
    assert restored == stable
    claimed = payload.pop("content_identity")
    validate_cid(claimed, codecs=("dag-json",))
    assert claimed == cid_for_dag_json(payload)
    incomplete = _decl(
        "run_task",
        ExportClassification.STABLE,
        "pkg/ops.py",
        4,
        module="pkg.ops",
        owner="pkg.ops.Handler",
        schema="pkg/run-task@1",
        version="1",
    )
    with pytest.raises(PublicSurfaceError, match="incomplete"):
        classify_export(incomplete)
    with pytest.raises(PublicSurfaceError, match="incomplete"):
        StablePublicSymbolRecord(
            symbol="run_task",
            owner="pkg.ops.Handler",
            schema="pkg/run-task@1",
            version="1",
            effects=(),
            errors=(),
            authority="pkg.ops.Handler",
            tests=(),
            proofs=(),
            consumers=(),
        )


def test_unknown_consumers_block_removal() -> None:
    unknown = _decl(
        "compat_run",
        ExportClassification.DEPRECATED,
        "pkg/ops.py",
        12,
        module="pkg.ops",
        consumer_evidence=ConsumerEvidenceKind.UNKNOWN,
    )
    manifest = build_public_surface_manifest(
        declarations=(unknown,),
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        removal_evidence=(
            RemovalEvidence(
                symbol="compat_run",
                deprecated=True,
                replacement="run_task",
                consumers_migrated=True,
                compatibility_satisfied=True,
                negative_import_tests=("test/test_compat_removed.py",),
                release_notes="docs/notes.md",
                still_exported=False,
            ),
        ),
    )
    record = manifest.export_for("compat_run")
    assert record.consumer_evidence is ConsumerEvidenceKind.UNKNOWN
    assert unknown_consumers_block_removal(record) is True
    gate = manifest.removal_gate("compat_run")
    assert gate.gates_satisfied is False
    assert RemovalBlockerKind.UNKNOWN_CONSUMERS in gate.blockers
    assert RemovalBlockerKind.MANIFEST_CANNOT_AUTHORIZE in gate.blockers
    assert manifest.removal_blocked("compat_run") is True
    with pytest.raises(PublicSurfaceAuthorityError, match="cannot authorize removal"):
        manifest.authorize_removal("compat_run")
    known = _manifest()
    remaining = known.removal_gate("run_task")
    assert RemovalBlockerKind.CONSUMERS_REMAIN in remaining.blockers
    assert remaining.gates_satisfied is False


def test_consumer_evidence_is_bound_from_references() -> None:
    declaration = _decl(
        "Handler",
        ExportClassification.PROVISIONAL,
        "pkg/ops.py",
        8,
        module="pkg.ops",
        consumer_evidence=ConsumerEvidenceKind.UNKNOWN,
    )
    manifest = build_public_surface_manifest(
        declarations=(declaration,),
        consumers=(
            ConsumerReference(
                symbol="Handler",
                consumer="pkg.cli.Handler",
                provenance=_fact("pkg/cli.py", 1),
                kind="import",
            ),
        ),
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    record = manifest.export_for("Handler")
    assert record.consumer_evidence is ConsumerEvidenceKind.KNOWN
    assert record.consumers == ("pkg.cli.Handler",)
    assert unknown_consumers_block_removal(record) is False
    assert RemovalBlockerKind.CONSUMERS_REMAIN in manifest.removal_gate("Handler").blockers


def test_accidental_surface_from_undeclared_private_internal_and_star() -> None:
    manifest = _manifest()
    kinds = {item.kind for item in manifest.accidental_exports}
    assert AccidentalExportKind.PRIVATE_NAME_IN_ALL in kinds
    assert AccidentalExportKind.INTERNAL_REEXPORT in kinds
    assert AccidentalExportKind.STAR_REEXPORT in kinds or AccidentalExportKind.WILDCARD_SURFACE in kinds
    leaked = [
        item for item in manifest.accidental_exports if item.symbol == "_leaked"
    ]
    assert leaked
    assert leaked[0].kind is AccidentalExportKind.PRIVATE_NAME_IN_ALL
    helper = [
        item for item in manifest.accidental_exports if item.symbol == "helper"
    ]
    assert helper
    assert helper[0].kind is AccidentalExportKind.INTERNAL_REEXPORT
    helper_export = manifest.export_for("helper")
    assert helper_export.classification is ExportClassification.ACCIDENTALLY_PUBLIC
    undeclared = build_public_surface_manifest(
        declarations=(),
        sources={"pkg/boom.py": _BOOM},
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    names = {item.symbol for item in undeclared.exports}
    assert "surviving" in names
    assert "_leaked" in names
    assert all(
        item.classification is ExportClassification.ACCIDENTALLY_PUBLIC
        for item in undeclared.exports
    )
    assert any(
        item.kind is AccidentalExportKind.UNDECLARED_PUBLIC
        for item in undeclared.accidental_exports
    )


def test_side_effect_free_fixture_never_imports_inspected_modules() -> None:
    discovered = discover_exports_from_sources(
        {"pkg/boom.py": _BOOM},
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    assert {item.symbol for item in discovered} >= {"surviving", "_leaked"}
    traces = assess_imports_from_sources(
        {"pkg/boom.py": _BOOM, "pkg/eager.py": _EAGER_SIDE_EFFECT, "pkg/lazy.py": _LAZY_CLEAN},
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    boom = [item for item in traces if "boom" in item.module or item.imported_symbol == "pkg.boom"]
    assert any(ImportEffectKind.EXCEPTION in item.effects for item in traces)
    eager = [item for item in traces if item.module in {"os", "pkg.eager"} or "eager" in item.provenance.span.path]
    assert any(item.laziness is ImportLaziness.EAGER for item in eager)
    assert any(
        ImportEffectKind.FILESYSTEM in item.effects or ImportEffectKind.MUTATION in item.effects
        for item in traces
    )
    lazy = [item for item in traces if item.imported_symbol == "run_task"]
    assert lazy
    assert all(item.laziness is ImportLaziness.LAZY for item in lazy)
    assert all(item.side_effect_free is True for item in lazy)
    manifest = build_public_surface_manifest(
        declarations=(_stable_decl(),),
        sources={"pkg/boom.py": _BOOM},
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    assert manifest.export_for("run_task").classification is ExportClassification.STABLE
    assert any(
        item.classification is ExportClassification.ACCIDENTALLY_PUBLIC
        for item in manifest.exports
        if item.symbol == "surviving"
    )


def test_projection_mismatch_vectors_for_python_cli_mcp() -> None:
    aligned = detect_projection_mismatch(_parity_bindings())
    assert aligned == ()
    missing_mcp = detect_projection_mismatch(_parity_bindings()[:2])
    assert any(item.kind is ProjectionMismatchKind.MISSING_MCP for item in missing_mcp)
    invented = detect_projection_mismatch(
        (
            ProjectionBinding(
                operation="invented",
                projection=ProjectionKind.CLI,
                schema="pkg/invented@1",
                version="1",
                provenance=_fact("pkg/cli.py", 2),
            ),
        )
    )
    kinds = {item.kind for item in invented}
    assert ProjectionMismatchKind.SEMANTIC_INVENTION in kinds
    assert ProjectionMismatchKind.MISSING_PYTHON in kinds
    diverged = detect_projection_mismatch(
        (
            ProjectionBinding(
                operation="run_task",
                projection=ProjectionKind.PYTHON,
                schema="pkg/run-task@1",
                version="1",
                provenance=_fact("pkg/ops.py", 4),
                effects=("read_state",),
                errors=("InvalidName",),
            ),
            ProjectionBinding(
                operation="run_task",
                projection=ProjectionKind.CLI,
                schema="pkg/run-task@2",
                version="2",
                provenance=_fact("pkg/cli.py", 4),
                effects=("write_state",),
                errors=("OtherError",),
            ),
            ProjectionBinding(
                operation="run_task",
                projection=ProjectionKind.MCP,
                schema="pkg/run-task@1",
                version="1",
                provenance=_fact("pkg/mcp.py", 4),
                effects=("read_state",),
                errors=("InvalidName",),
            ),
        )
    )
    mismatch_kinds = {item.kind for item in diverged}
    assert ProjectionMismatchKind.SCHEMA_MISMATCH in mismatch_kinds
    assert ProjectionMismatchKind.VERSION_MISMATCH in mismatch_kinds
    assert ProjectionMismatchKind.EFFECT_MISMATCH in mismatch_kinds
    assert ProjectionMismatchKind.ERROR_MISMATCH in mismatch_kinds
    manifest = build_public_surface_manifest(
        declarations=(_stable_decl(),),
        projections=(
            *_parity_bindings()[:2],
            ProjectionBinding(
                operation="invented",
                projection=ProjectionKind.MCP,
                schema="pkg/invented@1",
                version="1",
                provenance=_fact("pkg/mcp.py", 8),
            ),
        ),
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    finding_kinds = {item.kind for item in manifest.projection_findings}
    assert ProjectionMismatchKind.MISSING_MCP in finding_kinds
    assert ProjectionMismatchKind.SEMANTIC_INVENTION in finding_kinds


def test_architecture_ir_reexports_are_selected_without_executing_modules() -> None:
    graph = extract_architecture_graph(
        {
            "pkg/__init__.py": _PKG_INIT,
            "pkg/ops.py": _PKG_OPS,
            "pkg/internal.py": _PKG_INTERNAL,
        },
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    manifest = build_public_surface_manifest(
        declarations=(_stable_decl(),),
        architecture=graph,
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    names = {item.symbol for item in manifest.exports}
    assert "run_task" in names
    assert manifest.export_for("run_task").classification is ExportClassification.STABLE
    assert any(
        item.classification is ExportClassification.ACCIDENTALLY_PUBLIC
        for item in manifest.exports
        if item.symbol != "run_task"
    )


def test_conflicting_classifications_and_unknown_fields_fail_closed() -> None:
    first = _stable_decl()
    second = _decl(
        "run_task",
        ExportClassification.PROVISIONAL,
        "pkg/ops.py",
        4,
        module="pkg.ops",
    )
    with pytest.raises(PublicSurfaceError, match="conflicting classifications"):
        build_public_surface_manifest(
            declarations=(first, second),
            repository_tree=_TREE,
            freshness=_FRESHNESS,
        )
    with pytest.raises(PublicSurfaceError, match="unknown public-surface field"):
        PublicSurfaceManifest.from_mapping(
            {
                **_manifest(sources=None, projections=(), consumers=()).to_dict(),
                "unexpected": True,
            }
        )
    stable_payload = classify_export(_stable_decl()).stable.to_dict()  # type: ignore[union-attr]
    with pytest.raises(PublicSurfaceError, match="unknown public-surface field"):
        StablePublicSymbolRecord.from_mapping({**stable_payload, "extra": 1})
    with pytest.raises(PublicSurfaceError):
        ExportDeclaration.from_mapping({**first.to_dict(), "hidden": True})


def test_manifest_round_trip_and_canonical_identity() -> None:
    manifest = _manifest(sources=None)
    payload = manifest.to_dict()
    restored = PublicSurfaceManifest.from_mapping(payload)
    assert restored == manifest
    assert restored.to_dict() == payload
    assert PublicSurfaceManifest.from_json(manifest.to_json()) == manifest
    claimed = payload.pop("content_identity")
    validate_cid(claimed, codecs=("dag-json",))
    assert claimed == cid_for_dag_json(payload)
    assert claimed == manifest.content_identity
    assert not claimed.startswith("sha256:")
    assert restored.schema == PUBLIC_SURFACE_SCHEMA
    assert restored.version == PUBLIC_SURFACE_VERSION
    assert restored.can_authorize_removal is False
    assert restored.can_promote_internal is False
    assert restored.can_deprecate is False
    reordered = PublicSurfaceManifest(
        repository_tree=manifest.repository_tree,
        freshness=manifest.freshness,
        exports=tuple(reversed(manifest.exports)),
        stable_symbols=tuple(reversed(manifest.stable_symbols)),
        accidental_exports=tuple(reversed(manifest.accidental_exports)),
        projection_findings=tuple(reversed(manifest.projection_findings)),
        import_traces=tuple(reversed(manifest.import_traces)),
        removal_gates=tuple(reversed(manifest.removal_gates)),
    )
    assert reordered.content_identity == manifest.content_identity
    assert classify_exports is build_public_surface_manifest


def test_manifest_does_not_promote_deprecate_or_remove() -> None:
    manifest = _manifest(sources=None, projections=(), consumers=())
    with pytest.raises(PublicSurfaceAuthorityError, match="internal symbols"):
        manifest.promote_internal("helper")
    with pytest.raises(PublicSurfaceAuthorityError, match="deprecate"):
        manifest.deprecate_symbol("run_task")
    with pytest.raises(PublicSurfaceAuthorityError, match="cannot authorize removal"):
        refuse_removal("run_task")
    with pytest.raises(PublicSurfaceAuthorityError, match="internal symbols"):
        refuse_public_promotion("helper")
    with pytest.raises(PublicSurfaceAuthorityError, match="deprecate"):
        refuse_deprecation("run_task")
    with pytest.raises(PublicSurfaceAuthorityError):
        PublicSurfaceManifest(
            repository_tree=_TREE,
            freshness=_FRESHNESS,
            exports=manifest.exports,
            stable_symbols=manifest.stable_symbols,
            accidental_exports=(),
            projection_findings=(),
            import_traces=(),
            removal_gates=manifest.removal_gates,
            can_authorize_removal=True,
        )


def test_pyproject_cli_and_mcp_discovery_is_static() -> None:
    discovered = discover_exports_from_sources(
        {
            "pyproject.toml": _PYPROJECT,
            "pkg/cli.py": _PKG_CLI,
            "pkg/mcp.py": _PKG_MCP,
        },
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    origins = {item.origin for item in discovered}
    assert DiscoveryOrigin.PYPROJECT_SCRIPT in origins
    assert DiscoveryOrigin.ENTRYPOINT in origins
    assert DiscoveryOrigin.CLI_REGISTRY in origins
    assert DiscoveryOrigin.MCP_REGISTRY in origins
    names = {item.symbol for item in discovered}
    assert "run_task" in names
    assert "pkg-run" in names
    assert "AGENT_SUPERVISOR_OPERATION_TOOLS" in names


def test_current_tree_surface_bindings_exist() -> None:
    origins = {item.origin for item in CURRENT_SURFACE_BINDINGS}
    assert DiscoveryOrigin.ALL_LIST in origins
    assert DiscoveryOrigin.PYPROJECT_SCRIPT in origins
    assert DiscoveryOrigin.CLI_REGISTRY in origins
    assert DiscoveryOrigin.MCP_REGISTRY in origins
    assert DiscoveryOrigin.ENTRYPOINT in origins
    for binding in CURRENT_SURFACE_BINDINGS:
        path = _ROOT / binding.path
        assert path.is_file(), binding.path
        text = path.read_text(encoding="utf-8")
        assert binding.nominated_symbol in text
        lines = text.splitlines()
        assert 1 <= binding.start_line <= binding.end_line <= len(lines)


def test_ir_constructed_entrypoint_is_selected() -> None:
    fact = _fact("pkg/cli.py", 4)
    graph = ArchitectureIR.from_parts(
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        nodes=(
            ArchitectureNode("n:entrypoint:pkg.cli.main", NodeKind.ENTRYPOINT, fact),
            ArchitectureNode("n:symbol:pkg.cli.main", NodeKind.SYMBOL, fact),
            ArchitectureNode("n:module:pkg.cli", NodeKind.MODULE, fact),
        ),
        edges=(
            ArchitectureEdge(
                "e:contains:pkg.cli:main",
                EdgeKind.CONTAINS,
                "n:module:pkg.cli",
                "n:entrypoint:pkg.cli.main",
                fact,
            ),
        ),
    )
    manifest = build_public_surface_manifest(
        declarations=(),
        architecture=graph,
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    record = manifest.export_for("main")
    assert record.classification is ExportClassification.ACCIDENTALLY_PUBLIC
    assert DiscoveryOrigin.ENTRYPOINT in record.origins
