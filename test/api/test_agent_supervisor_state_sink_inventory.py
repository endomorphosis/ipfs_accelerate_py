"""DQP-001: deterministic inventory of mutable supervisor state sinks."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNER = (
    REPO_ROOT / "scripts/ops/agent_supervisor/inventory_state_sinks.py"
)
INVENTORY_DOC = (
    REPO_ROOT / "docs/architecture/AGENT_SUPERVISOR_STATE_SINK_INVENTORY.md"
)


def _load_scanner():
    spec = importlib.util.spec_from_file_location(
        "agent_supervisor_inventory_state_sinks",
        SCANNER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inv():
    return _load_scanner()


@pytest.fixture(scope="module")
def report(inv):
    return inv.build_inventory(REPO_ROOT, fail_on_unclassified=True)


def test_scanner_module_and_inventory_doc_exist() -> None:
    assert SCANNER.is_file()
    assert INVENTORY_DOC.is_file()


def test_interface_contract(inv) -> None:
    assert inv.INTERFACE_ID == "SupervisorStateSinkInventory@1"
    assert inv.INVENTORY_SCHEMA.endswith("state-sink-inventory@1")
    assert inv.INVENTORY_IS_COMPLETION_EVIDENCE is False
    assert inv.INVENTORY_IS_PROOF_EVIDENCE is False
    assert inv.INVENTORY_AUTHORIZES_MUTATION is False


def test_catalog_is_sorted_unique_and_well_formed(inv) -> None:
    sinks = inv.known_sinks()
    ids = [sink.sink_id for sink in sinks]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
    assert len(sinks) >= 20
    classes = {sink.classification for sink in sinks}
    assert inv.SinkClassification.AUTHORITY in classes
    assert inv.SinkClassification.STATIC_INPUT in classes
    assert inv.SinkClassification.CACHE in classes
    assert inv.SinkClassification.OS_BOOTSTRAP in classes
    assert inv.SinkClassification.EMERGENCY_DIAGNOSTIC in classes
    # High-value OS bootstrap / cache locks must be first-class catalog rows.
    for required_id in (
        "run-registry-lock",
        "analysis-cache-lock",
        "repository-index-lock",
        "proof-certificate-locks",
        "integration-tool-install-locks",
    ):
        assert required_id in ids
    for sink in sinks:
        module_path = REPO_ROOT / sink.writer_module
        assert module_path.is_file(), f"missing writer module: {sink.writer_module}"
        assert sink.reuse_candidate.strip()
        assert sink.destination_domain
        assert sink.retirement_stage
        if sink.is_git_source_bytes:
            assert sink.classification is inv.SinkClassification.STATIC_INPUT
            assert sink.destination_domain is inv.DestinationDomain.NON_STATE


def test_inventory_is_deterministic(inv) -> None:
    first = inv.build_inventory(REPO_ROOT, fail_on_unclassified=True)
    second = inv.build_inventory(REPO_ROOT, fail_on_unclassified=True)
    assert first.to_dict() == second.to_dict()
    assert first.ok is True
    rendered = inv.render_markdown(first)
    assert rendered == inv.render_markdown(second)


def test_scanner_fails_closed_for_unclassified_mutable_sink(inv, report) -> None:
    assert report.unclassified == ()
    assert report.ok is True

    rogue = inv.DiscoveredMarker(
        module="ipfs_accelerate_py/agent_supervisor/todo_daemon/rogue_writer.py",
        basename="brand_new_secret_authority.duckdb",
        line=1,
        literal="brand_new_secret_authority.duckdb",
    )
    # Family coverage would classify a .duckdb under todo_daemon; put it outside
    # every covered package so classification must fail closed.
    outside = inv.DiscoveredMarker(
        module="ipfs_accelerate_py/agent_supervisor/unknown_plane/writer.py",
        basename="brand_new_secret_authority.duckdb",
        line=7,
        literal="brand_new_secret_authority.duckdb",
    )
    assert inv.classify_discovery(outside, report.sinks) is None
    with pytest.raises(inv.UnclassifiedMutableSinkError) as excinfo:
        inv.build_inventory(
            REPO_ROOT,
            discoveries=tuple(report.discoveries) + (outside,),
            fail_on_unclassified=True,
        )
    assert "brand_new_secret_authority.duckdb" in str(excinfo.value)

    # Also ensure a known marker remains classified.
    known = inv.DiscoveredMarker(
        module="ipfs_accelerate_py/agent_supervisor/merge/merge_queue.py",
        basename="merge_queue.duckdb",
        line=281,
        literal="merge_queue.duckdb",
    )
    classified = inv.classify_discovery(known, report.sinks)
    assert classified is not None
    assert classified.sink_id == "merge-queue-duckdb"
    assert rogue.basename.endswith(".duckdb")


def test_includes_direct_duckdb_writers(report, inv) -> None:
    writers = set(report.direct_duckdb_writers)
    required = {
        "duckdb-task-source",
        "duckdb-state-primitives",
        "lease-coordination-duckdb",
        "merge-queue-duckdb",
        "merge-resolver-duckdb",
        "artifact-store-json-duckdb",
        "proof-scheduler-duckdb",
        "prover-evidence-duckdb",
        "formal-verification-cache-duckdb",
    }
    assert required <= writers
    assert len(writers) >= len(required)
    for sink in report.sinks:
        if sink.sink_id in required:
            assert sink.is_direct_duckdb_writer is True
            assert sink.media_type in {
                inv.MediaType.DUCKDB,
                inv.MediaType.SQLITE,
                inv.MediaType.ARTIFACT,
            }


def test_records_cross_file_atomicity_gaps(report) -> None:
    gaps = report.cross_file_atomicity_gaps
    assert gaps
    joined = "\n".join(gaps)
    assert "objective-heap-markdown" in joined
    assert "taskboard-markdown" in joined
    assert "plan-revision-store" in joined
    assert "artifact-store-json-duckdb" in joined
    assert "merge-train-state-dir" in joined
    for sink in report.sinks:
        if sink.cross_file_atomicity_gap:
            assert sink.atomicity_model.value in {
                "cross_file_non_atomic",
                "best_effort_mirror",
                "append_only",
                "flock_plus_transaction",
            }


def test_records_reuse_candidates(report) -> None:
    assert report.reuse_candidates
    assert len(report.reuse_candidates) == len(report.sinks)
    joined = "\n".join(report.reuse_candidates)
    assert "duckdb_state" in joined or "exclusive_file_lock" in joined
    assert "LeaseCoordinator" in joined
    assert "domain_events" in joined


def test_distinguishes_git_source_bytes_from_supervisor_state(report, inv) -> None:
    git_sinks = [sink for sink in report.sinks if sink.is_git_source_bytes]
    assert len(git_sinks) == 1
    git = git_sinks[0]
    assert git.sink_id == "git-source-bytes"
    assert git.classification is inv.SinkClassification.STATIC_INPUT
    assert git.destination_domain is inv.DestinationDomain.NON_STATE
    assert git.retirement_stage is inv.RetirementStage.RETAIN_PERMANENT
    assert report.git_source_distinctions
    assert "git-source-bytes" in report.git_source_distinctions[0]

    orchestration = [
        sink
        for sink in report.sinks
        if not sink.is_git_source_bytes
        and sink.classification is inv.SinkClassification.AUTHORITY
    ]
    assert orchestration
    for sink in orchestration:
        assert sink.destination_domain is not inv.DestinationDomain.NON_STATE


def test_live_discovery_classifies_all_markers(report) -> None:
    assert report.discoveries, "expected live path markers under agent_supervisor"
    assert report.unclassified == ()
    # High-signal concrete sinks must appear in discovery.
    basenames = {item.basename for item in report.discoveries}
    for required in (
        "merge_queue.duckdb",
        "coordination.duckdb",
        "proof_scheduler.duckdb",
        "events.jsonl",
    ):
        assert required in basenames, f"missing discovery for {required}"


def test_inventory_document_covers_catalog_and_acceptance_themes(
    report,
    inv,
) -> None:
    text = INVENTORY_DOC.read_text(encoding="utf-8")
    assert "SupervisorStateSinkInventory@1" in text
    assert "Git/source bytes" in text or "Git/source" in text
    assert "Direct DuckDB writers" in text
    assert "Cross-file atomicity gaps" in text
    assert "Reuse candidates" in text
    assert "fails CI" in text.lower() or "unclassified" in text.lower()
    for sink in report.sinks:
        assert f"`{sink.sink_id}`" in text, f"doc missing sink {sink.sink_id}"

    # Document should stay aligned with the rendered catalog (allow hand edits
    # only when they preserve every sink id and the acceptance sections).
    rendered = inv.render_markdown(report)
    for section in (
        "## Authority boundary: Git/source bytes vs supervisor state",
        "## Direct DuckDB writers",
        "## Cross-file atomicity gaps",
        "## Reuse candidates",
        "## Full sink catalog",
    ):
        assert section in text
        assert section in rendered


def test_dependency_lockfiles_are_not_supervisor_state_sinks(inv) -> None:
    """Package-manager and Git index locks are source inputs, not sinks."""

    for name in (
        "cargo.lock",
        "Cargo.lock",
        "poetry.lock",
        "yarn.lock",
        "uv.lock",
        "index.lock",
        "Pipfile.lock",
        "HEAD.lock",
        "config.lock",
        "packed-refs.lock",
        "shallow.lock",
    ):
        assert inv._basename_of_literal(name) is None
        assert inv._basename_of_literal(f"vendor/{name}") is None


def test_source_scan_globs_are_not_mutable_sinks(inv) -> None:
    """Arbitrary source-tree globs must not trip the fail-closed inventory gate."""

    for name in (
        "*.json",
        "*.py",
        "*.pem",
        "*.key",
        "*.p12",
        "*.pfx",
        "*.safetensors",
        "*.gguf",
        "*.pt",
        "*.pkp",
        "*.pkv",
        "*.min.js",
        "*.generated.*",
        "pytorch_model*.bin",
    ):
        assert inv._basename_of_literal(name) is None, name

    # Sink-family globs remain discoverable and classifiable.
    for name in ("*.duckdb", "*.jsonl", "*.pid", "*.lock", "*.todo.md"):
        assert inv._basename_of_literal(name) == name
        media = inv._media_for_basename(name)
        assert media is not None


def test_cli_check_passes_and_is_deterministic() -> None:
    cmd = [sys.executable, str(SCANNER), "--check", "--json"]
    # Discovery walks the whole agent_supervisor tree; keep a generous bound
    # for constrained validation runners while still failing closed on hangs.
    timeout_seconds = 300
    first = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    second = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    assert first.returncode == 0, first.stderr or first.stdout
    assert second.returncode == 0, second.stderr or second.stdout
    payload_a = json.loads(first.stdout)
    payload_b = json.loads(second.stdout)
    assert payload_a == payload_b
    assert payload_a["ok"] is True
    assert payload_a["interface_id"] == "SupervisorStateSinkInventory@1"
    assert payload_a["unclassified_count"] == 0
    assert payload_a["direct_duckdb_writers"]
    assert payload_a["cross_file_atomicity_gaps"]
    assert payload_a["reuse_candidates"]
    assert payload_a["git_source_distinctions"]
