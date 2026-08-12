"""IVP-017: incremental-verification benchmark artifact and harness.

Validates that:

* a freshly generated artifact binds the canonical source snapshot, corpus
  CID/evaluated count, policy, effective environment, commands, measurement
  schema, and status while Git HEAD remains diagnostic only;
* metrics cover cache hit rate, tests selected/full, ground-truth FN/FP,
  outcome discrepancies, static/proof execution, wall samples, paired or
  estimated reused time, route, frontier escalation, counterexample context,
  and estimator-bound token savings;
* zero stale/simulated acceptance is hard while target misses are recorded
  rather than blocking artifact creation;
* deterministic commitments and old-key historical preservation hold;
* incompatible cross-tree unaffected reuse is explicitly unmet;
* small route appears in at least one and 20% of measured localized fixtures
  or the target is red;
* missing canonical fixtures or real provers are typed unavailable/not_measured.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.verification import (
    source_snapshot as snapshot_module,
)
from ipfs_accelerate_py.agent_supervisor.verification.evaluation import (
    MeasurementStatus,
    default_fixture_root,
)
from ipfs_accelerate_py.agent_supervisor.verification.model_route import (
    ModelRoute,
)
from ipfs_accelerate_py.agent_supervisor.verification.source_snapshot import (
    SOURCE_SNAPSHOT_DOMAIN,
    SOURCE_SNAPSHOT_EXCLUDED_PATHS,
    SOURCE_SNAPSHOT_SCHEMA,
    SourceSnapshot,
    SourceSnapshotEntry,
    SourceSnapshotError,
    build_source_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_MODULE = (
    REPO_ROOT / "benchmarks" / "agent_supervisor" / "incremental_verification.py"
)


def _load_benchmark_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "ivp_incremental_verification_benchmark",
        BENCHMARK_MODULE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BENCH = _load_benchmark_module()
BENCHMARK_EVIDENCE = _BENCH.BENCHMARK_EVIDENCE
BENCHMARK_INTERFACE = _BENCH.BENCHMARK_INTERFACE
BENCHMARK_SCHEMA = _BENCH.BENCHMARK_SCHEMA
DEFAULT_OUTPUT_RELPATH = _BENCH.DEFAULT_OUTPUT_RELPATH
GOAL_ID = _BENCH.GOAL_ID
SMALL_ROUTE_MIN_FRACTION = _BENCH.SMALL_ROUTE_MIN_FRACTION
TASK_ID = _BENCH.TASK_ID
TOKENIZER_ID = _BENCH.TOKENIZER_ID
artifacts_structurally_equivalent = _BENCH.artifacts_structurally_equivalent
ensure_corpus_manifest = _BENCH.ensure_corpus_manifest
estimate_tokens = _BENCH.estimate_tokens
run_incremental_verification_benchmark = (
    _BENCH.run_incremental_verification_benchmark
)
write_stable_benchmark_artifact = _BENCH.write_stable_benchmark_artifact

ARTIFACT_PATH = REPO_ROOT / DEFAULT_OUTPUT_RELPATH
FIXTURE_ROOT = default_fixture_root(REPO_ROOT)

REQUIRED_TOP_LEVEL = {
    "schema",
    "interface",
    "evidence",
    "task_id",
    "goal_id",
    "authoritative",
    "status",
    "source_snapshot_id",
    "source_snapshot_schema",
    "source_snapshot_domain",
    "observed_head",
    "corpus",
    "policy",
    "effective_environment",
    "commands",
    "measurement_schema",
    "metrics",
    "targets",
    "target_misses",
    "cases",
    "provers",
    "commitments",
    "historical_preservation",
    "cross_tree_unaffected_reuse",
    "zero_stale_simulated_accepted",
    "content_id",
}

REQUIRED_METRICS = {
    "cache",
    "tests",
    "false_negatives",
    "false_positives",
    "outcome_discrepancies",
    "static_proof_execution",
    "wall_samples",
    "reused_time",
    "routes",
    "frontier_escalation",
    "counterexample_context",
    "token_savings",
}


def _load_artifact() -> dict[str, Any]:
    assert ARTIFACT_PATH.is_file(), f"missing benchmark artifact: {ARTIFACT_PATH}"
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _init_nested_repository(path: Path, label: str) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "--quiet")
    _git(path, "config", "user.email", "ivp@example.invalid")
    _git(path, "config", "user.name", "IVP Test")
    (path / "payload.txt").write_text(f"{label}\n", encoding="utf-8")
    _git(path, "add", "payload.txt")
    _git(path, "commit", "--quiet", "-m", f"initialize {label}")
    return _git(path, "rev-parse", "HEAD")


def _init_snapshot_repository(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, str]]:
    path.mkdir()
    _git(path, "init", "--quiet")
    _git(path, "config", "user.email", "ivp@example.invalid")
    _git(path, "config", "user.name", "IVP Test")

    (path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (path / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    executable = path / "tool.sh"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    (path / "link").symlink_to("source.py")
    todo = path / "docs" / "architecture" / "incremental_verification_planner.todo.md"
    todo.parent.mkdir(parents=True)
    todo.write_text(
        "# Board\n\n## IVP-021 Snapshot\n\n"
        "- Status: todo\n- Note: Status: todo remains source\n",
        encoding="utf-8",
    )
    artifact = (
        path / "artifacts" / "agent_supervisor" / "incremental_verification"
        / "benchmark.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    report = path / "docs" / "architecture" / "INCREMENTAL_VERIFICATION_PLANNER_REPORT.md"
    report.write_text("report\n", encoding="utf-8")

    reviewed = {
        name: _init_nested_repository(path / name, name)
        for name in ("ipfs_kit_py", "ipfs_datasets_py")
    }
    vendor_oid = _init_nested_repository(path / "vendor", "vendor")
    _git(path, "add", ".gitignore", "source.py", "tool.sh", "link", "docs", "artifacts")
    for name, oid in {**reviewed, "vendor": vendor_oid}.items():
        _git(path, "update-index", "--add", "--cacheinfo", f"160000,{oid},{name}")
    _git(path, "commit", "--quiet", "-m", "initial source")
    monkeypatch.setattr(snapshot_module, "_REVIEWED_GITLINKS", reviewed)
    return path, reviewed


# ---------------------------------------------------------------------------
# Canonical source-snapshot contract
# ---------------------------------------------------------------------------


def test_source_snapshot_is_stable_across_provenance_history_and_clone_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, reviewed = _init_snapshot_repository(tmp_path / "source", monkeypatch)
    baseline = build_source_snapshot(root)
    assert baseline.schema == SOURCE_SNAPSHOT_SCHEMA
    assert baseline.domain == SOURCE_SNAPSHOT_DOMAIN
    assert baseline.source_snapshot_id.startswith("sha256:")
    assert len(baseline.source_snapshot_id) == len("sha256:") + 64
    assert set(SOURCE_SNAPSHOT_EXCLUDED_PATHS).isdisjoint(
        entry.path for entry in baseline.entries
    )

    # Adding the same effective path to the index and then committing it does
    # not add provenance to the manifest.
    loose = root / "loose.txt"
    loose.write_text("same bytes\n", encoding="utf-8")
    untracked = build_source_snapshot(root)
    _git(root, "add", "loose.txt")
    staged = build_source_snapshot(root)
    assert staged.source_snapshot_id == untracked.source_snapshot_id
    _git(root, "commit", "--quiet", "-m", "track existing bytes")
    committed = build_source_snapshot(root)
    assert committed.source_snapshot_id == staged.source_snapshot_id
    assert committed.observed_head != staged.observed_head

    # HEAD, branch, exact IVP lifecycle values, ignored files, and the two
    # closed self-referential outputs are not identity inputs.
    stable_id = committed.source_snapshot_id
    _git(root, "checkout", "--quiet", "-b", "other-branch")
    _git(root, "commit", "--quiet", "--allow-empty", "-m", "history only")
    todo = root / "docs" / "architecture" / "incremental_verification_planner.todo.md"
    todo.write_text(
        todo.read_text(encoding="utf-8").replace(
            "- Status: todo", "- Status: completed", 1
        ),
        encoding="utf-8",
    )
    (root / "ignored.txt").write_text("ignored churn\n", encoding="utf-8")
    git_dir = Path(_git(root, "rev-parse", "--git-dir"))
    if not git_dir.is_absolute():
        git_dir = root / git_dir
    info = git_dir / "info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "exclude").write_text("info-ignored.txt\n", encoding="utf-8")
    (root / "info-ignored.txt").write_text("ignored by info/exclude\n", encoding="utf-8")
    for relative in SOURCE_SNAPSHOT_EXCLUDED_PATHS:
        (root / relative).write_text("excluded churn\n", encoding="utf-8")
    stable = build_source_snapshot(root)
    assert stable.source_snapshot_id == stable_id
    assert stable.observed_head != committed.observed_head

    # A clone with identical effective paths has the same identity even though
    # its absolute root is different.
    _git(root, "add", todo.relative_to(root).as_posix())
    _git(root, "commit", "--quiet", "-m", "lifecycle transition")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "--quiet", str(root), str(clone))
    for name in reviewed:
        (clone / name).mkdir(exist_ok=True)
        _git(clone / name, "init", "--quiet")
        _git(clone / name, "fetch", "--quiet", str(root / name), reviewed[name])
        _git(clone / name, "checkout", "--quiet", "--detach", reviewed[name])
    assert build_source_snapshot(clone).source_snapshot_id == build_source_snapshot(
        root
    ).source_snapshot_id


def test_source_snapshot_detects_effective_source_mode_symlink_and_gitlink_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _reviewed = _init_snapshot_repository(tmp_path / "source", monkeypatch)
    baseline = build_source_snapshot(root).source_snapshot_id

    source = root / "source.py"
    original = source.read_bytes()
    source.write_bytes(original + b"# drift\n")
    assert build_source_snapshot(root).source_snapshot_id != baseline
    source.write_bytes(original)

    tool = root / "tool.sh"
    tool.chmod(0o644)
    assert build_source_snapshot(root).source_snapshot_id != baseline
    tool.chmod(0o755)

    link = root / "link"
    link.unlink()
    link.symlink_to("tool.sh")
    assert build_source_snapshot(root).source_snapshot_id != baseline
    link.unlink()
    link.symlink_to("source.py")

    untracked = root / "new.py"
    untracked.write_text("new = True\n", encoding="utf-8")
    assert build_source_snapshot(root).source_snapshot_id != baseline
    untracked.unlink()

    # Status-like prose and malformed status rows remain identity-bearing.
    todo = root / "docs" / "architecture" / "incremental_verification_planner.todo.md"
    todo.write_text(
        todo.read_text(encoding="utf-8").replace(
            "Status: todo remains source", "Status: completed remains source"
        ),
        encoding="utf-8",
    )
    assert build_source_snapshot(root).source_snapshot_id != baseline

    todo.write_text(
        todo.read_text(encoding="utf-8").replace(
            "- Status: Todo", "- Status: <ivp-task-lifecycle>"
        ),
        encoding="utf-8",
    )
    assert build_source_snapshot(root).source_snapshot_id != baseline
    _git(root, "checkout", "--", todo.relative_to(root).as_posix())

    # A gitlink is represented by its exact index object, never by a recursive
    # hash of its nested checkout.
    vendor = root / "vendor"
    (vendor / "payload.txt").write_text("vendor changed\n", encoding="utf-8")
    assert build_source_snapshot(root).source_snapshot_id == baseline
    _git(vendor, "add", "payload.txt")
    _git(vendor, "commit", "--quiet", "-m", "vendor drift")
    assert build_source_snapshot(root).source_snapshot_id == baseline
    _git(root, "add", "vendor")
    assert build_source_snapshot(root).source_snapshot_id != baseline


def test_source_snapshot_deletion_fixed_point_and_reviewed_gitlink_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, reviewed = _init_snapshot_repository(tmp_path / "source", monkeypatch)
    source = root / "source.py"
    source.unlink()
    deleted = build_source_snapshot(root)
    _git(root, "add", "--update")
    _git(root, "commit", "--quiet", "-m", "delete source")
    committed = build_source_snapshot(root)
    assert committed.source_snapshot_id == deleted.source_snapshot_id
    assert committed.observed_head != deleted.observed_head

    kit = root / "ipfs_kit_py"
    (kit / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(SourceSnapshotError, match="contains untracked paths"):
        build_source_snapshot(root)
    (kit / "dirty.txt").unlink()

    (kit / "payload.txt").write_text("different HEAD\n", encoding="utf-8")
    _git(kit, "add", "payload.txt")
    _git(kit, "commit", "--quiet", "-m", "unreviewed dependency")
    with pytest.raises(SourceSnapshotError, match="does not equal gitlink"):
        build_source_snapshot(root)
    _git(kit, "checkout", "--quiet", "--detach", reviewed["ipfs_kit_py"])

    monkeypatch.setattr(
        snapshot_module,
        "_REVIEWED_GITLINKS",
        {**reviewed, "ipfs_kit_py": "0" * 40},
    )
    with pytest.raises(SourceSnapshotError, match="must equal reviewed object"):
        build_source_snapshot(root)


def test_source_snapshot_rejects_nested_index_and_filemode_cleanliness_bypasses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, reviewed = _init_snapshot_repository(tmp_path / "source", monkeypatch)
    kit = root / "ipfs_kit_py"

    _git(kit, "update-index", "--assume-unchanged", "payload.txt")
    (kit / "payload.txt").write_text("hidden dirty content\n", encoding="utf-8")
    with pytest.raises(SourceSnapshotError, match="non-normal index flag"):
        build_source_snapshot(root)
    _git(kit, "update-index", "--no-assume-unchanged", "payload.txt")
    _git(kit, "checkout", "--quiet", reviewed["ipfs_kit_py"], "--", "payload.txt")

    _git(kit, "config", "core.fileMode", "false")
    payload = kit / "payload.txt"
    payload.chmod(payload.stat().st_mode | stat.S_IXUSR)
    assert _git(kit, "status", "--porcelain=v1") == ""
    with pytest.raises(SourceSnapshotError, match="mode differs from its index"):
        build_source_snapshot(root)


def test_source_snapshot_rejects_reviewed_gitlink_clean_filter_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, reviewed = _init_snapshot_repository(tmp_path / "source", monkeypatch)
    kit = root / "ipfs_kit_py"
    attributes = kit / ".gitattributes"
    payload = kit / "payload.txt"
    attributes.write_text("payload.txt filter=sealed\n", encoding="utf-8")
    payload.write_bytes(b"sealed")
    _git(kit, "config", "filter.sealed.clean", "printf sealed")
    _git(kit, "config", "filter.sealed.smudge", "cat")
    _git(kit, "config", "filter.sealed.required", "true")
    _git(kit, "add", ".gitattributes", "payload.txt")
    _git(kit, "commit", "--quiet", "-m", "configure clean filter")
    reviewed_head = _git(kit, "rev-parse", "HEAD")
    reviewed["ipfs_kit_py"] = reviewed_head
    _git(
        root,
        "update-index",
        "--cacheinfo",
        f"160000,{reviewed_head},ipfs_kit_py",
    )
    baseline = build_source_snapshot(root).source_snapshot_id

    payload.write_bytes(b"EVIL DIFFERENT PHYSICAL")
    _git(kit, "add", "payload.txt")
    assert _git(kit, "status", "--porcelain=v1") == ""
    with pytest.raises(SourceSnapshotError, match="physical bytes differ"):
        build_source_snapshot(root)
    assert baseline


def test_source_snapshot_does_not_execute_reviewed_gitlink_filter_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, reviewed = _init_snapshot_repository(tmp_path / "source", monkeypatch)
    kit = root / "ipfs_kit_py"
    marker = tmp_path / "filter-executed"
    filter_script = tmp_path / "clean-filter.sh"
    filter_script.write_text(
        f"#!/bin/sh\ntouch {marker}\ncat\n",
        encoding="utf-8",
    )
    filter_script.chmod(0o755)
    (kit / ".gitattributes").write_text(
        "payload.txt filter=probe\n", encoding="utf-8"
    )
    _git(kit, "config", "filter.probe.clean", str(filter_script))
    _git(kit, "config", "filter.probe.smudge", "cat")
    _git(kit, "config", "filter.probe.required", "true")
    _git(kit, "add", ".gitattributes", "payload.txt")
    _git(kit, "commit", "--quiet", "-m", "configure probe filter")
    reviewed_head = _git(kit, "rev-parse", "HEAD")
    reviewed["ipfs_kit_py"] = reviewed_head
    _git(
        root,
        "update-index",
        "--cacheinfo",
        f"160000,{reviewed_head},ipfs_kit_py",
    )
    marker.unlink(missing_ok=True)

    build_source_snapshot(root)
    assert not marker.exists()


def test_source_snapshot_uses_owner_execute_and_rejects_same_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _reviewed = _init_snapshot_repository(tmp_path / "source", monkeypatch)
    source = root / "source.py"
    baseline = build_source_snapshot(root)
    baseline_entry = next(entry for entry in baseline.entries if entry.path == "source.py")

    source.chmod(source.stat().st_mode | stat.S_IXGRP)
    group_only = build_source_snapshot(root)
    group_entry = next(entry for entry in group_only.entries if entry.path == "source.py")
    assert group_entry.mode == baseline_entry.mode == "100644"
    assert group_only.source_snapshot_id == baseline.source_snapshot_id
    source.chmod(0o644)

    original_reader = snapshot_module._stable_regular_bytes
    replaced = False

    def _replace_after_read(path: Path) -> tuple[bytes, int]:
        nonlocal replaced
        result = original_reader(path)
        if path == source and not replaced:
            replacement = source.with_name("replacement.tmp")
            replacement.write_text("VALUE = 2\n", encoding="utf-8")
            replacement.replace(source)
            replaced = True
        return result

    monkeypatch.setattr(snapshot_module, "_stable_regular_bytes", _replace_after_read)
    with pytest.raises(SourceSnapshotError, match="content changed"):
        build_source_snapshot(root)


def test_malformed_task_status_remains_identity_bearing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _reviewed = _init_snapshot_repository(tmp_path / "source", monkeypatch)
    baseline = build_source_snapshot(root).source_snapshot_id
    todo = root / "docs" / "architecture" / "incremental_verification_planner.todo.md"
    todo.write_text(
        todo.read_text(encoding="utf-8").replace("- Status: todo", "- Status: Todo"),
        encoding="utf-8",
    )
    assert build_source_snapshot(root).source_snapshot_id != baseline


def test_source_snapshot_public_records_reject_forged_identity() -> None:
    entry = SourceSnapshotEntry(path="source.py", mode="100644", sha256="0" * 64)
    with pytest.raises(SourceSnapshotError, match="does not match"):
        SourceSnapshot(
            entries=(entry,),
            source_snapshot_id="sha256:" + ("f" * 64),
            observed_head=None,
        )
    with pytest.raises(SourceSnapshotError, match="canonical"):
        SourceSnapshotEntry(path="source.py", mode="100664", sha256="0" * 64)


# ---------------------------------------------------------------------------
# Module / entry points
# ---------------------------------------------------------------------------


def test_benchmark_module_exists_and_exports_runner() -> None:
    assert BENCHMARK_MODULE.is_file()
    assert callable(run_incremental_verification_benchmark)
    assert BENCHMARK_SCHEMA.endswith("incremental-verification-benchmark@2")
    assert BENCHMARK_INTERFACE == "IncrementalVerificationBenchmark@2"
    assert BENCHMARK_EVIDENCE == "ivp/benchmark@2"
    assert TASK_ID == "IVP-017"
    assert GOAL_ID == "IVP-G090"


def test_artifact_exists_and_binds_identity_surfaces() -> None:
    doc = _load_artifact()
    missing = REQUIRED_TOP_LEVEL - set(doc)
    assert not missing, f"artifact missing keys: {sorted(missing)}"
    assert doc["schema"] == BENCHMARK_SCHEMA
    assert doc["interface"] == BENCHMARK_INTERFACE
    assert doc["evidence"] == BENCHMARK_EVIDENCE
    assert doc["task_id"] == TASK_ID
    assert doc["goal_id"] == GOAL_ID
    assert doc["authoritative"] is False
    assert doc["target_success_asserted"] is False
    current = build_source_snapshot(REPO_ROOT)
    assert doc["source_snapshot_id"] == current.source_snapshot_id
    assert doc["source_snapshot_schema"] == SOURCE_SNAPSHOT_SCHEMA
    assert doc["source_snapshot_domain"] == SOURCE_SNAPSHOT_DOMAIN
    # Checked-in evidence survives the commit that introduced it. HEAD is a
    # diagnostic observation only, so it may lag while source identity cannot.
    assert doc["observed_head"] is None or re.fullmatch(
        r"[0-9a-f]{40,64}", str(doc["observed_head"])
    )
    assert doc["status"] in {"green", "red", "yellow", "not_measured"}

    corpus = doc["corpus"]
    assert "corpus_id" in corpus
    assert "evaluated_count" in corpus
    assert "corpus_cid" in corpus or corpus.get("present") is False

    policy = doc["policy"]
    assert policy.get("policy_id")
    assert policy.get("zero_stale_simulated_acceptance_hard") is True

    env = doc["effective_environment"]
    assert env.get("python_version")
    assert env.get("platform")

    commands = doc["commands"]
    assert "generate_artifact" in commands
    assert "validate" in commands

    measurement = doc["measurement_schema"]
    assert measurement.get("version")
    assert TOKENIZER_ID in {
        measurement.get("tokenizer_id"),
        (doc["metrics"]["token_savings"].get("tokenizer_id")),
    }
    for field in (
        "cache_hit_rate",
        "tests_selected_full",
        "ground_truth_false_negatives",
        "ground_truth_false_positives",
        "outcome_discrepancies",
        "static_proof_execution",
        "wall_samples",
        "paired_estimated_reused_time",
        "route",
        "frontier_escalation",
        "counterexample_context",
        "estimator_bound_token_savings",
    ):
        assert field in measurement["fields"]


def test_metrics_cover_required_dimensions() -> None:
    doc = _load_artifact()
    metrics = doc["metrics"]
    missing = REQUIRED_METRICS - set(metrics)
    assert not missing, f"metrics missing: {sorted(missing)}"

    cache = metrics["cache"]
    assert "hit_rate" in cache
    assert 0.0 <= float(cache["hit_rate"]) <= 1.0
    assert cache["zero_stale_simulated_accepted"] is True

    tests = metrics["tests"]
    assert "selected_total" in tests
    assert "full_total" in tests

    assert "ground_truth_total" in metrics["false_negatives"]
    assert "ground_truth_total" in metrics["false_positives"]
    assert "case_count" in metrics["outcome_discrepancies"]

    static_proof = metrics["static_proof_execution"]
    assert "static_checks_executed" in static_proof
    assert "proof_obligations_executed" in static_proof
    assert static_proof["status"] in {
        MeasurementStatus.MEASURED.value,
        MeasurementStatus.NOT_MEASURED.value,
    }

    wall = metrics["wall_samples"]
    assert int(wall["sample_count"]) >= 1 or wall.get("status") == (
        MeasurementStatus.NOT_MEASURED.value
    )
    if int(wall["sample_count"]) >= 1:
        assert "tolerance_ms" in wall
        assert wall["role"] == "observational"
        assert len(wall["samples_ms"]) == int(wall["sample_count"])

    reused = metrics["reused_time"]
    assert reused.get("label") in {"paired", "estimated"}
    paired = reused.get("paired_cache") or {}
    assert paired.get("label") == "paired" or reused.get("label") == "estimated"

    routes = metrics["routes"]
    assert isinstance(routes.get("counts"), dict)
    assert "frontier_escalation_rate" in routes
    assert "rate" in metrics["frontier_escalation"]

    cx = metrics["counterexample_context"]
    assert "total_bytes" in cx
    assert "total_tokens" in cx

    tokens = metrics["token_savings"]
    assert tokens["estimator_bound"] is True
    assert tokens["tokenizer_id"] == TOKENIZER_ID
    assert tokens["tokenizer_version"]
    assert "tokens_saved_total" in tokens
    assert "compared_artifact_bounds" in tokens


def test_zero_stale_simulated_is_hard_and_target_misses_do_not_block() -> None:
    doc = _load_artifact()
    assert doc["zero_stale_simulated_accepted"] is True
    hard = doc["targets"]["zero_stale_simulated_accepted"]
    assert hard["hard"] is True
    assert hard["status"] == "met"
    assert hard["value"] is True

    # Seeded corpus FN makes the soft release target red; artifact still lands.
    assert isinstance(doc["target_misses"], list)
    assert ARTIFACT_PATH.is_file()
    # Creation is never blocked: status is one of the closed vocabulary.
    assert doc["status"] in {"green", "red", "yellow", "not_measured"}
    # Soft FN miss must be recorded when corpus measured a nonzero FN total.
    fn_total = doc["metrics"]["false_negatives"].get("corpus_total")
    if isinstance(fn_total, int) and fn_total > 0:
        assert any(
            item.get("target") == "zero_controlled_false_negatives"
            for item in doc["target_misses"]
        )
        assert doc["targets"]["zero_controlled_false_negatives"]["status"] == "red"


def test_deterministic_commitments_and_historical_preservation() -> None:
    doc = _load_artifact()
    commitments = doc["commitments"]
    assert commitments["deterministic"] is True
    assert commitments.get("commitment_cid")
    assert commitments.get("body", {}).get("source_snapshot_id") == doc[
        "source_snapshot_id"
    ]

    hist = doc["historical_preservation"]
    assert hist["holds"] is True
    assert hist["old_key_reusable"] is True
    assert hist["historical_present"] is True
    assert doc["targets"]["old_key_historical_preservation"]["status"] == "met"
    assert doc["targets"]["deterministic_commitments"]["status"] == "met"


def test_cross_tree_unaffected_reuse_is_explicitly_unmet() -> None:
    doc = _load_artifact()
    cross = doc["cross_tree_unaffected_reuse"]
    assert cross["status"] == "unmet"
    assert cross["explicitly_unmet"] is True
    assert cross["new_tree_reusable"] is False
    assert "exact_full_tree" in str(cross.get("reason") or "")
    target = doc["targets"]["incompatible_cross_tree_unaffected_reuse"]
    assert target["status"] == "unmet"
    assert target["explicitly_unmet"] is True


def test_small_route_distribution_or_red() -> None:
    doc = _load_artifact()
    cases = doc["cases"]
    measured_localized = [
        case
        for case in cases
        if case.get("localized")
        and case.get("measurement_status") == MeasurementStatus.MEASURED.value
    ]
    small = [
        case
        for case in measured_localized
        if (case.get("route") or {}).get("route")
        == ModelRoute.SMALL_LOCAL_MODEL.value
    ]
    target = doc["targets"]["small_route_localized_distribution"]
    if not measured_localized:
        assert target["status"] == "not_measured"
        return
    fraction = len(small) / len(measured_localized)
    if len(small) >= 1 and fraction >= SMALL_ROUTE_MIN_FRACTION:
        assert target["status"] == "met"
    else:
        assert target["status"] == "red"
        assert any(
            item.get("target") == "small_route_localized_distribution"
            for item in doc["target_misses"]
        )


def test_cases_report_per_fixture_metrics() -> None:
    doc = _load_artifact()
    cases = doc["cases"]
    assert isinstance(cases, list)
    if not cases:
        # Corpus absent path — provers / fixtures typed unavailable.
        assert doc["corpus"].get("present") is False or doc["status"] == (
            "not_measured"
        )
        return

    assert len(cases) == int(doc["corpus"].get("evaluated_count") or len(cases))
    for case in cases:
        assert case.get("fixture_id")
        assert "tests" in case
        assert "selected_count" in case["tests"]
        assert "full_count" in case["tests"]
        assert "false_negatives" in case
        assert "false_positives" in case
        assert "outcome_discrepancies" in case
        assert "static_proof_execution" in case
        assert "wall" in case
        assert case["reused_time"]["label"] in {"paired", "estimated"}
        assert case["route"]["route"]
        assert "frontier_escalation" in case["route"]
        assert "counterexample_context" in case
        tokens = case["token_savings"]
        assert tokens["estimator_bound"] is True
        assert tokens["tokenizer_id"] == TOKENIZER_ID


def test_provers_typed_available_or_unavailable() -> None:
    doc = _load_artifact()
    provers = doc["provers"]
    assert "probes" in provers
    for name, probe in provers["probes"].items():
        assert probe["status"] in {"available", "unavailable"}
        if probe["status"] == "unavailable":
            assert probe["measurement_status"] == MeasurementStatus.NOT_MEASURED.value
        else:
            assert probe["measurement_status"] == MeasurementStatus.MEASURED.value
            assert probe.get("path")
    # Missing provers must appear in unavailable list, never fabricated as green wins.
    for name in provers.get("unavailable") or ():
        assert provers["probes"][name]["status"] == "unavailable"


def test_fresh_run_binds_current_source_and_is_schema_stable(
    tmp_path: Path,
) -> None:
    out = tmp_path / "benchmark.json"
    artifact = run_incremental_verification_benchmark(
        repo_root_path=REPO_ROOT,
        wall_samples=2,
        output_path=out,
    )
    current = build_source_snapshot(REPO_ROOT)
    assert artifact["source_snapshot_id"] == current.source_snapshot_id
    assert artifact["observed_head"] == current.observed_head
    assert artifact["schema"] == BENCHMARK_SCHEMA
    assert artifact["authoritative"] is False
    assert "content_id" in artifact
    # Deterministic commitment body includes source snapshot and corpus.
    body = artifact["commitments"]["body"]
    assert body["source_snapshot_id"] == artifact["source_snapshot_id"]
    # Re-run yields the same commitment for the same source/corpus.
    again = run_incremental_verification_benchmark(
        repo_root_path=REPO_ROOT,
        wall_samples=2,
        output_path=out,
    )
    assert again["commitments"]["commitment_cid"] == artifact["commitments"][
        "commitment_cid"
    ]
    assert again["commitments"]["body"]["case_fixture_ids"] == artifact[
        "commitments"
    ]["body"]["case_fixture_ids"]


def test_absent_corpus_is_not_measured_never_zero_fn(tmp_path: Path) -> None:
    empty = tmp_path / "empty_fixtures"
    empty.mkdir()
    artifact = run_incremental_verification_benchmark(
        repo_root_path=REPO_ROOT,
        fixture_root=empty,
        wall_samples=1,
        output_path=tmp_path / "absent.json",
    )
    assert artifact["corpus"]["present"] is False or artifact["corpus"][
        "evaluated_count"
    ] == 0
    # FN totals must not be fabricated as zero when not measured.
    summary = artifact["selection_summary"]
    assert summary["measurement_status"] == MeasurementStatus.NOT_MEASURED.value
    assert summary["total_false_negatives"] is None
    assert summary["total_false_positives"] is None
    assert artifact["authoritative"] is False


def test_estimator_token_savings_bound_to_tokenizer_version() -> None:
    doc = _load_artifact()
    tokens = doc["metrics"]["token_savings"]
    assert tokens["tokenizer_id"] == TOKENIZER_ID
    assert tokens["tokenizer_version"]
    assert tokens["estimator_bound"] is True
    # Estimator is deterministic.
    assert estimate_tokens("abcd") == estimate_tokens("abcd")
    assert estimate_tokens("abcd") > 0


def test_cli_writes_output(tmp_path: Path) -> None:
    out = tmp_path / "cli-benchmark.json"
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    prefix = "ipfs_kit_py:ipfs_datasets_py:."
    env["PYTHONPATH"] = f"{prefix}:{existing}" if existing else prefix
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_MODULE),
            "--output",
            str(out),
            "--wall-samples",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == BENCHMARK_SCHEMA
    assert payload["source_snapshot_id"] == build_source_snapshot(
        REPO_ROOT
    ).source_snapshot_id
    # Ephemeral process noise must not appear in the stable artifact.
    assert "pid" not in (payload.get("effective_environment") or {})
    assert "generated_at_unix_ms" not in payload


def test_stable_write_is_fixed_point_across_measured_reruns(tmp_path: Path) -> None:
    """Re-running the generator must not rewrite when only wall samples change.

    Candidate stabilization re-validates once; nonconvergent wall-sample churn
    previously failed post-validation with candidate_stabilization_nonconvergent.
    """

    out = tmp_path / "fixed-point.json"
    first = run_incremental_verification_benchmark(
        repo_root_path=REPO_ROOT,
        wall_samples=2,
        output_path=out,
    )
    written_first, preserved_first = write_stable_benchmark_artifact(out, first)
    assert preserved_first is False
    assert out.is_file()
    first_bytes = out.read_bytes()

    second = run_incremental_verification_benchmark(
        repo_root_path=REPO_ROOT,
        wall_samples=2,
        output_path=out,
    )
    # Measured timings may differ; structural identity must still hold.
    assert artifacts_structurally_equivalent(written_first, second)
    written_second, preserved_second = write_stable_benchmark_artifact(out, second)
    assert preserved_second is True
    assert out.read_bytes() == first_bytes
    assert written_second["content_id"] == written_first["content_id"]


def test_checked_in_artifact_matches_runner_structural_contract() -> None:
    """Checked-in artifact must remain a valid structural projection."""

    doc = _load_artifact()
    # Ensure corpus was evaluable when artifact was generated, or honestly not.
    corpus = doc["corpus"]
    if corpus.get("present"):
        assert int(corpus.get("evaluated_count") or 0) >= 1
        assert corpus.get("corpus_cid")
        assert len(doc["cases"]) == int(corpus["evaluated_count"])
    else:
        assert doc["status"] in {"not_measured", "red", "yellow"}
        assert corpus.get("measurement_status") == (
            MeasurementStatus.NOT_MEASURED.value
        )


def test_ensure_corpus_manifest_reports_status() -> None:
    info = ensure_corpus_manifest(FIXTURE_ROOT)
    assert "present" in info
    assert "corpus_id" in info
    if info["present"]:
        assert info["corpus_cid"]
        assert int(info["case_count"]) >= 1
