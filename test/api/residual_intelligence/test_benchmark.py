from __future__ import annotations

import copy
import json
import re
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.benchmark import (
    IDENTITY_FIELDS,
    MANIFEST_SCHEMA,
    PARTITIONS,
    REQUIRED_BINDINGS,
    REQUIRED_KINDS,
    PairedBenchmarkRunner,
    ResidualBenchmarkManifest,
    build_frozen_benchmark_contract,
    load_cases,
    load_frozen_benchmark,
    load_manifest,
    sha256_identity,
    validate_frozen_benchmark,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    PROGRAM_ID,
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "benchmarks/agent_supervisor/residual_intelligence/manifest.json"
CASES = ROOT / "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl"
SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")

# BEGIN VRIF-030 PORTAL BASELINE (materializer-owned)
VRIF_PORTAL_BASELINE_COMMIT = "3cf925ca62b583427c2e16843608b688901f6e6e"
VRIF_PORTAL_BASELINE_TREE = "ccafb2d4bf1dc20dc606eea877c080ed383c54b6"
# END VRIF-030 PORTAL BASELINE (materializer-owned)
VRIF_BENCHMARK_ARTIFACT_COMMIT = "0d4fa2bdcd66bac2e5193e8f6e96679433ac322e"


def _strict_json_bytes(raw: bytes, *, noun: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"{noun} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs)
    assert isinstance(value, dict)
    return value


def _strict_json_object(path: Path) -> dict[str, Any]:
    return _strict_json_bytes(path.read_bytes(), noun=str(path))


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )


def _git_blob(commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    return result.stdout


def test_current_artifacts_equal_independent_owner_reconstruction() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", VRIF_PORTAL_BASELINE_COMMIT)
    assert re.fullmatch(r"[0-9a-f]{40}", VRIF_PORTAL_BASELINE_TREE)
    resolved_commit = _git(
        "rev-parse", "--verify", f"{VRIF_PORTAL_BASELINE_COMMIT}^{{commit}}"
    )
    resolved_tree = _git(
        "rev-parse", "--verify", f"{VRIF_PORTAL_BASELINE_COMMIT}^{{tree}}"
    )
    ancestry = _git("merge-base", "--is-ancestor", VRIF_PORTAL_BASELINE_COMMIT, "HEAD")
    assert resolved_commit.returncode == 0
    assert resolved_commit.stdout.strip() == VRIF_PORTAL_BASELINE_COMMIT
    assert resolved_tree.returncode == 0
    assert resolved_tree.stdout.strip() == VRIF_PORTAL_BASELINE_TREE
    assert ancestry.returncode == 0
    artifact_commit = _git(
        "rev-parse",
        "--verify",
        f"{VRIF_BENCHMARK_ARTIFACT_COMMIT}^{{commit}}",
    )
    artifact_ancestry = _git(
        "merge-base",
        "--is-ancestor",
        VRIF_BENCHMARK_ARTIFACT_COMMIT,
        "HEAD",
    )
    assert artifact_commit.returncode == 0
    assert artifact_commit.stdout.strip() == VRIF_BENCHMARK_ARTIFACT_COMMIT
    assert artifact_ancestry.returncode == 0

    manifest = load_manifest(MANIFEST)
    freeze = manifest["benchmark_freeze"]
    source = {
        "commit": VRIF_PORTAL_BASELINE_COMMIT,
        "tree": VRIF_PORTAL_BASELINE_TREE,
    }
    assert manifest["source_revision"] == VRIF_PORTAL_BASELINE_COMMIT
    assert freeze["source"] == source

    objective_paths = (
        "docs/architecture/agent_supervisor_residual_intelligence.objectives.md",
        "docs/architecture/agent_supervisor_residual_intelligence.todo.md",
    )
    operation_path = "ipfs_accelerate_py/agent_supervisor/control/control_plane.py"
    provider_path = "config/agent_supervisor_residual_intelligence_scheduler.json"
    admission_path = (
        "benchmarks/agent_supervisor/residual_intelligence/"
        "synthetic_training_admission.json"
    )
    split_path = (
        "benchmarks/agent_supervisor/residual_intelligence/"
        "synthetic_split_manifest.json"
    )
    inventory_path = (
        "docs/architecture/residual_intelligence_inventory/"
        "residual_model_call_inventory.json"
    )
    artifact_blobs = {
        path: _git_blob(VRIF_BENCHMARK_ARTIFACT_COMMIT, path)
        for path in (
            *objective_paths,
            operation_path,
            provider_path,
            admission_path,
            split_path,
            inventory_path,
            "test/api/residual_intelligence/test_benchmark.py",
        )
    }
    assert MANIFEST.read_bytes() == _git_blob(
        VRIF_BENCHMARK_ARTIFACT_COMMIT,
        "benchmarks/agent_supervisor/residual_intelligence/manifest.json",
    )
    assert CASES.read_bytes() == _git_blob(
        VRIF_BENCHMARK_ARTIFACT_COMMIT,
        "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl",
    )
    admission = _strict_json_bytes(
        artifact_blobs[admission_path],
        noun=admission_path,
    )
    split = _strict_json_bytes(
        artifact_blobs[split_path],
        noun=split_path,
    )
    admission_body = dict(admission)
    admission_id = admission_body.pop("admission_id")
    assert admission_id == content_identity(admission_body)

    base_bindings = {
        "repository_states": sha256_identity(source),
        "objective_revisions": sha256_identity(
            {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "residual-benchmark-objective-revisions@1"
                ),
                "artifacts": {
                    path: sha256_identity(artifact_blobs[path])
                    for path in objective_paths
                },
            }
        ),
        "operation_catalog": sha256_identity(artifact_blobs[operation_path]),
        "provider_policy": sha256_identity(artifact_blobs[provider_path]),
        "tokenizer": sha256_identity(
            {
                "admission_id": admission_id,
                "disposition": "no_learned_tokenizer_admitted",
            }
        ),
        "model_versions": sha256_identity(
            {
                "inventory_blob_identity": sha256_identity(
                    artifact_blobs[inventory_path]
                ),
                "disposition": "training_unavailable",
            }
        ),
        "validation_policy": sha256_identity(
            {
                "argv": [
                    [
                        "python3 -m pytest -q "
                        "test/api/residual_intelligence/test_benchmark.py"
                    ]
                ],
                "test_blob_identity": sha256_identity(
                    artifact_blobs[
                        "test/api/residual_intelligence/test_benchmark.py"
                    ]
                ),
            }
        ),
    }
    task_families = [family.value for family in ResidualTaskFamily]
    expected = build_frozen_benchmark_contract(
        task_families=task_families,
        source_commit=VRIF_PORTAL_BASELINE_COMMIT,
        source_tree=VRIF_PORTAL_BASELINE_TREE,
        split_root=str(split["split_root"]),
        base_bindings=base_bindings,
    )
    expected_manifest = {
        "schema": MANIFEST_SCHEMA,
        "program_identifier": PROGRAM_ID,
        "status": "staged_not_qualified",
        "owner_task": "VRIF-030",
        "source_revision": VRIF_PORTAL_BASELINE_COMMIT,
        "partitions": expected["partitions"],
        "required_case_kinds": expected["case_kinds"],
        "task_families": task_families,
        "training_admission": "training_unavailable",
        "weights_committed": False,
        "large_corpus_committed": False,
        "promotion_evidence": False,
        "benchmark_freeze": expected["benchmark_freeze"],
    }
    assert manifest == expected_manifest
    assert [case.to_dict() for case in load_cases(CASES)] == expected["cases"]


def test_frozen_catalog_covers_every_family_and_owner_scheduled_partition() -> None:
    manifest, cases = load_frozen_benchmark(MANIFEST, CASES)

    assert len(cases) == len(ResidualTaskFamily) * len(PARTITIONS)
    assert [(case.family, case.partition, case.kind) for case in cases] == [
        (family, partition, kind)
        for family in ResidualTaskFamily
        for partition, kind in zip(PARTITIONS, REQUIRED_KINDS, strict=True)
    ]
    assert manifest.frozen_root == manifest.computed_frozen_root
    assert manifest.case_catalog_root == manifest.benchmark_freeze["case_root"]
    assert manifest.training_admission == "training_unavailable"


def test_cases_are_group_lineage_safe_and_hidden_inputs_are_denied() -> None:
    _, cases = load_frozen_benchmark(MANIFEST, CASES)

    assert len({case.group_id for case in cases}) == len(cases)
    for case in cases:
        assert all(SHA256_ID.fullmatch(getattr(case, field)) for field in IDENTITY_FIELDS)
        assert case.hidden_test is (case.partition in {"held_out", "adversarial"})
        assert case.input_disposition == "payload_unavailable_training_unavailable"
        assert case.expected_outcome is ExpertDisposition.CAPABILITY_UNAVAILABLE

    cross_repository = [case for case in cases if case.kind == "cross_repository"]
    unknown_ood = [case for case in cases if case.kind == "unknown_ood"]
    assert len(cross_repository) == len(ResidualTaskFamily)
    assert {case.partition for case in cross_repository} == {"held_out"}
    assert all(case.hidden_test for case in cross_repository)
    assert len(unknown_ood) == len(ResidualTaskFamily)
    assert {case.partition for case in unknown_ood} == {"adversarial"}
    assert all(case.hidden_test for case in unknown_ood)


def test_manifest_freezes_all_required_identity_bindings_and_roots() -> None:
    manifest, cases = load_frozen_benchmark(MANIFEST, CASES)
    freeze = manifest.benchmark_freeze

    assert set(freeze["bindings"]) == set(REQUIRED_BINDINGS)
    assert all(SHA256_ID.fullmatch(value) for value in freeze["bindings"].values())
    assert re.fullmatch(r"[0-9a-f]{40}", freeze["source"]["commit"])
    assert re.fullmatch(r"[0-9a-f]{40}", freeze["source"]["tree"])
    assert freeze["source"]["commit"] == manifest.source_revision
    assert freeze["case_count"] == len(cases) == 96
    assert freeze["fault_schedule"]["entries"] == [
        {
            "family": case.family.value,
            "partition": case.partition,
            "kind": case.kind,
            "hidden_test": case.hidden_test,
            "group_id": case.group_id,
        }
        for case in cases
    ]


def test_frozen_roots_and_group_lineage_fail_closed_on_tampering() -> None:
    manifest, cases = load_frozen_benchmark(MANIFEST, CASES)
    altered_input = replace(cases[0], input_identity="sha256:" + "0" * 64)
    with pytest.raises(ResidualIntelligenceError, match="96-case schedule"):
        validate_frozen_benchmark(manifest, (altered_input, *cases[1:]))

    held_out = next(case for case in cases if case.partition == "held_out")
    training = next(case for case in cases if case.partition == "training")
    mixed_lineage = replace(held_out, group_id=training.group_id)
    replaced = tuple(mixed_lineage if case.case_id == held_out.case_id else case for case in cases)
    with pytest.raises(ResidualIntelligenceError, match="96-case schedule"):
        validate_frozen_benchmark(manifest, replaced)

    altered_manifest = copy.deepcopy(manifest.to_dict())
    altered_manifest["benchmark_freeze"]["bindings"]["validation_policy"] = (
        "sha256:" + "f" * 64
    )
    with pytest.raises(ResidualIntelligenceError, match="binding set"):
        validate_frozen_benchmark(ResidualBenchmarkManifest.from_dict(altered_manifest), cases)


def test_paired_baseline_preserves_complete_frozen_all_abstain_denominators() -> None:
    manifest, cases = load_frozen_benchmark(MANIFEST, CASES)
    expected = manifest.benchmark_freeze["paired_baseline"]
    runner = PairedBenchmarkRunner()

    assert runner.evaluate(manifest, cases) == expected
    assert runner.evaluate(
        manifest,
        cases,
        prior=expected["before"],
        current=expected["after"],
    ) == expected
    assert expected["candidate_only"] is True
    assert expected["training_performed"] is False
    assert expected["case_count"] == len(cases)
    assert expected["before"] == expected["after"]
    assert expected["before"]["accept"] == 0
    assert expected["before"]["abstain"] == len(cases)
    assert set(expected["before"]["denominators_by_family"]) == {
        family.value for family in ResidualTaskFamily
    }
    assert set(expected["before"]["denominators_by_family"].values()) == {len(PARTITIONS)}
    with pytest.raises(ResidualIntelligenceError, match="all-abstain"):
        runner.evaluate(manifest, cases, prior={"accept": 1}, current={"accept": 1})


def test_strict_loader_rejects_duplicate_or_legacy_fields_and_bad_hidden_flags(
    tmp_path: Path,
) -> None:
    raw = load_manifest(MANIFEST)
    assert set(raw) == set(ResidualBenchmarkManifest._FIELDS)
    assert raw["benchmark_freeze"]["freeze_id"]

    duplicate_case = tmp_path / "duplicate.jsonl"
    duplicate_case.write_text(
        '{"family":"TASK_CLASSIFICATION","family":"RISK_CLASSIFICATION"}\\n',
        encoding="utf-8",
    )
    with pytest.raises(ResidualIntelligenceError, match="duplicate key"):
        load_cases(duplicate_case)

    legacy_manifest = dict(raw)
    legacy_manifest["frozen_roots"] = {}
    with pytest.raises(ResidualIntelligenceError, match="unknown fields"):
        ResidualBenchmarkManifest.from_dict(legacy_manifest)

    training = next(case for case in load_cases(CASES) if case.partition == "training")
    with pytest.raises(ResidualIntelligenceError, match="hidden_test"):
        replace(training, hidden_test=True)
