"""Fail-closed tests for the protected LGCVF closeout judge."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/validate_logic_governed_compositional_verification_fabric_closeout.py"


def _load() -> ModuleType:
    name = "lgcvf_closeout_validation_tested"
    specification = importlib.util.spec_from_file_location(name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _qualification(module: ModuleType) -> dict[str, object]:
    observation: dict[str, object] = {
        "schema": "lgcvf-independent-pytest-observation@1",
        "suite_id": "fixed-independent-suite",
        "manifest": {"manifest_cid": "cid:fixed-manifest"},
        "isolation": {
            "profile": "landlock-readonly-seccomp-no-network",
            "checkout_write_permitted": False,
            "network_permitted": False,
            "completion_authoritative": False,
            "landlock_abi": 4,
            "seccomp_denied_syscall_count": 2,
        },
        "collected": 1,
        "passed_count": 1,
        "failed_count": 0,
        "skipped_count": 0,
        "xfailed_count": 0,
        "xpassed_count": 0,
        "error_count": 0,
        "nodeids_cid": "cid:fixed-nodeids",
        "exit_code": 0,
        "passed": True,
        "duration_ms": 1,
        "transcript_sha256": "sha256:" + "a" * 64,
        "failure_tail": "",
    }
    observation["observation_cid"] = module.content_identity(observation)
    value: dict[str, object] = {
        "schema": "lgcvf-independent-hermetic-qualification@1",
        "plan_cid": module.PLAN_CID,
        "predecessor_plan_cid": "cid:predecessor",
        "cohort": "hermetic_local_execution",
        "candidate_suites_are_self_authority": False,
        "independent_fixed_manifest_executed": True,
        "checkout_fingerprint_cid": module.content_identity({"checkout": "fixture"}),
        "checkout_unchanged": True,
        "passed": True,
        "totals": {
            "collected": 1,
            "passed_count": 1,
            "failed_count": 0,
            "skipped_count": 0,
            "xfailed_count": 0,
            "xpassed_count": 0,
            "error_count": 0,
        },
        "suites": [observation],
        "task_implementation_complete": False,
        "test_qualification_complete": True,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
        "production_authoritative": False,
        "limitations": ["hermetic only"],
    }
    value["result_cid"] = module.content_identity(value)
    return value


def _benchmark(module: ModuleType) -> dict[str, object]:
    benchmark: dict[str, object] = {
        "schema": "lgcvf-symbolic-displacement-benchmark@1",
        "interface": "LgcvfSymbolicDisplacementBenchmark@1",
        "cohort": "hermetic_local_execution",
        "production_authoritative": False,
        "overall_disposition": "partial",
        "release_qualified": False,
        "production_authorized": False,
        "execution_evidence": {
            "fresh_execution_receipts_reproducible": False,
            "vertical_result_cid": "cid:fresh-run",
            "artifact_cid": "cid:proof-artifact",
            "artifact_verification_receipt_cid": "cid:artifact-verification",
        },
        "pairing": {"policy_root": "cid:policy", "model_invocation_count": 0},
        "task_class_coverage": {
            "required": ["local_bug_repair", "dynamic_opaque_python_escalation"],
            "observed": ["local_bug_repair"],
            "missing": ["dynamic_opaque_python_escalation"],
        },
        "paired_result": {
            "schema": "lgcvf-paired-benchmark@1",
            "cohort": "hermetic_local_execution",
            "production_authoritative": False,
            "baseline": {"context_bytes": 1000},
            "challenger": {"context_bytes": 400},
            "comparison": {"context_reduction_bps": 6000},
        },
        "thresholds": [
            {
                "threshold_id": "representative_task_class_coverage",
                "target": 2,
                "observed": 1,
                "comparison": "at_least",
                "disposition": "missed",
                "reason": "",
            }
        ],
        "excluded_cohorts": ["production_authoritative_evidence"],
        "limitations": ["hermetic only"],
    }
    benchmark["reproducible_projection_cid"] = module.content_identity(
        module._benchmark_projection(benchmark)
    )
    benchmark["report_cid"] = module.content_identity(benchmark)
    return benchmark


def _write_evidence(
    module: ModuleType, root: Path
) -> tuple[Path, dict[str, object], Path, dict[str, object]]:
    benchmark = _benchmark(module)
    benchmark_path = root / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")

    qualification = _qualification(module)
    qualification_path = root / "qualification.json"
    qualification_path.write_text(json.dumps(qualification), encoding="utf-8")
    return benchmark_path, benchmark, qualification_path, qualification


def _install_replay(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    *,
    benchmark: dict[str, object],
    qualification: dict[str, object],
) -> list[tuple[str, tuple[str, ...]]]:
    calls: list[tuple[str, tuple[str, ...]]] = []
    real_run = module.subprocess.run

    def replay(command: tuple[str, ...], **kwargs: Any) -> Any:
        if command[0] == "git":
            return real_run(command, **kwargs)
        assert command[0] == sys.executable
        assert kwargs == {
            "cwd": module.ROOT,
            "check": False,
            "capture_output": True,
            "text": True,
            "timeout": module.PROTECTED_REPLAY_TIMEOUT_SECONDS,
        }
        script = Path(command[1])
        if script == module.QUALIFICATION_VALIDATOR_PATH:
            label = "qualification"
        else:
            assert script == module.BENCHMARK_VALIDATOR_PATH
            label = "benchmark"
        arguments = tuple(command[2:])
        calls.append((label, arguments))
        value = qualification if label == "qualification" else benchmark
        return module.subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(value),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", replay)
    return calls


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _clone(value: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(value))


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
    )


def _rehash_benchmark(module: ModuleType, value: dict[str, object]) -> None:
    value["reproducible_projection_cid"] = module.content_identity(
        module._benchmark_projection(value)
    )
    value["report_cid"] = module.content_identity(
        {key: item for key, item in value.items() if key != "report_cid"}
    )


def _rehash_qualification(module: ModuleType, value: dict[str, object]) -> None:
    for observation in value["suites"]:  # type: ignore[union-attr]
        observation["observation_cid"] = module.content_identity(
            {key: item for key, item in observation.items() if key != "observation_cid"}
        )
    value["result_cid"] = module.content_identity(
        {key: item for key, item in value.items() if key != "result_cid"}
    )


def _rehash_successors(module: ModuleType, value: dict[str, object]) -> None:
    for task in value["tasks"]:  # type: ignore[union-attr]
        task["task_cid"] = module.content_identity(
            {key: item for key, item in task.items() if key != "task_cid"}
        )
    value["successor_tasks_cid"] = module.content_identity(
        {key: item for key, item in value.items() if key != "successor_tasks_cid"}
    )


def _release(
    module: ModuleType,
    benchmark: dict[str, object],
    qualification: dict[str, object],
) -> str:
    nodeids = [item["nodeids_cid"] for item in qualification["suites"]]  # type: ignore[index]
    qualification_authority_cid = module.content_identity(
        module._qualification_authority_evidence(qualification)
    )
    benchmark_authority_cid = module.content_identity(
        module._benchmark_authority_evidence(benchmark)
    )
    disposition = benchmark["overall_disposition"]
    if disposition == "development_targets_met":
        disposition = "partial"
    return f"""# LGCVF release disposition

## Evidence

- Formal plan CID: {module.PLAN_CID}
- Qualification result CID: {qualification['result_cid']}
- Qualification authority CID: {qualification_authority_cid}
- Qualification suite node IDs: {_json(nodeids)}
- Benchmark result CID: {benchmark['report_cid']}
- Benchmark authority CID: {benchmark_authority_cid}
- Evidence cohort: hermetic_local_execution

## Disposition

- Disposition: {disposition}
- Task implementation: incomplete
- Test success: passed_hermetic
- Objective completion: incomplete
- Release qualification: not_qualified
- Production authorization: not_authorized
- Threshold comparison: {_json(benchmark['thresholds'])}

## Blockers

- External authority gate: blocked_external_authority
- Manual authority gate: blocked_manual

## Limitations

- Limitations: {_json(['Hermetic evidence is not production authority.'])}
"""


def _task(module: ModuleType, **fields: object) -> dict[str, object]:
    value = dict(fields)
    value["task_cid"] = module.content_identity(value)
    return value


def _successors(
    module: ModuleType,
    *,
    benchmark_cid: str,
    qualification_cid: str,
    release_sha256: str,
) -> dict[str, object]:
    tasks = [
        _task(
            module,
            task_id="LGCVF-S001",
            title="Obtain independent external qualification evidence",
            status="blocked_external_authority",
            owning_repository="ipfs_accelerate_py",
            depends_on=[],
            outputs=["data/agent_supervisor/lgcvf/external_qualification.json"],
            validation=["python -m json.tool data/agent_supervisor/lgcvf/external_qualification.json"],
            acceptance=["Independent external authority validates the current plan and roots."],
            reason_codes=["blocked_external_authority"],
        ),
        _task(
            module,
            task_id="LGCVF-S002",
            title="Obtain explicit operator production authorization",
            status="blocked_manual",
            owning_repository="ipfs_accelerate_py",
            depends_on=["LGCVF-S001"],
            outputs=["data/agent_supervisor/lgcvf/production_authorization.json"],
            validation=["python -m json.tool data/agent_supervisor/lgcvf/production_authorization.json"],
            acceptance=["A current operator receipt authorizes the exact qualified source roots."],
            reason_codes=["blocked_manual"],
        ),
    ]
    value: dict[str, object] = {
        "schema": "lgcvf-successor-tasks@1",
        "plan_cid": module.PLAN_CID,
        "benchmark_cid": benchmark_cid,
        "qualification_cid": qualification_cid,
        "release_report_sha256": release_sha256,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
        "tasks": tasks,
    }
    value["successor_tasks_cid"] = module.content_identity(value)
    return value


def _implementation(
    module: ModuleType,
    *,
    benchmark: dict[str, object],
    qualification: dict[str, object],
    release_sha256: str,
    successors: dict[str, object],
) -> str:
    headings = module._IMPLEMENTATION_HEADINGS
    execution = benchmark["execution_evidence"]
    comparison = benchmark["paired_result"]["comparison"]  # type: ignore[index]
    displacement = {
        "model_invocation_count": benchmark["pairing"]["model_invocation_count"],  # type: ignore[index]
        "context_comparison": comparison,
    }
    task_ids = [task["task_id"] for task in successors["tasks"]]  # type: ignore[index]
    completion = {
        "task_implementation_complete": False,
        "test_qualification_complete": True,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
    }
    qualification_authority_cid = module.content_identity(
        module._qualification_authority_evidence(qualification)
    )
    benchmark_authority_cid = module.content_identity(
        module._benchmark_authority_evidence(benchmark)
    )
    revisions, topology = module._current_repository_truth(qualification)
    changed_files = {
        "ipfs_accelerate_py": [
            "scripts/validate_logic_governed_compositional_verification_fabric_closeout.py"
        ],
        "ipfs_datasets_py": ["ipfs_datasets_py/logic/verification.py"],
    }
    sections = [
        f"## {headings[0]}\n\n"
        f"- Source revisions: {_json(revisions)}\n"
        f"- Repository topology: {_json(topology)}",
        f"## {headings[1]}\n\n"
        f"- Reused capabilities: {_json(['Content-addressed validation contracts'])}",
        f"## {headings[2]}\n\n"
        f"- Verified gaps: {_json(['External qualification remains unavailable'])}",
        f"## {headings[3]}\n\n- Completion states: {_json(completion)}",
        f"## {headings[4]}\n\n"
        f"- Files changed by repository: {_json(changed_files)}",
        f"## {headings[5]}\n\n"
        f"- Public interfaces: {_json(['Lgcvf closeout validation JSON interface'])}",
        f"## {headings[6]}\n\n"
        f"- Test commands: {_json(['python -m pytest -q test/api/test_agent_supervisor_lgcvf_closeout_validation.py'])}\n"
        f"- Exact test results: {_json(qualification['totals'])}",
        f"## {headings[7]}\n\n- Vertical receipt identities: {_json(execution)}",
        f"## {headings[8]}\n\n"
        f"- Benchmark disposition: {benchmark['overall_disposition']}\n"
        f"- Thresholds: {_json(benchmark['thresholds'])}",
        f"## {headings[9]}\n\n- Displacement evidence: {_json(displacement)}",
        f"## {headings[10]}\n\n"
        f"- Remaining risks: {_json(['External and operator authority remain unavailable'])}\n"
        f"- Production blockers: {_json(['blocked_external_authority', 'blocked_manual'])}",
        f"## {headings[11]}\n\n"
        f"- Successor task IDs: {_json(task_ids)}\n"
        f"- Successor tasks CID: {successors['successor_tasks_cid']}",
    ]
    return f"""# LGCVF implementation report

- Formal plan CID: {module.PLAN_CID}
- Qualification result CID: {qualification['result_cid']}
- Qualification authority CID: {qualification_authority_cid}
- Benchmark result CID: {benchmark['report_cid']}
- Benchmark authority CID: {benchmark_authority_cid}
- Release report SHA256: {release_sha256}
- Task implementation: incomplete
- Test success: passed_hermetic
- Objective completion: incomplete
- Release qualification: not_qualified
- Production authorization: not_authorized

{chr(10).join(chr(10) + section for section in sections)}
"""


def test_release_reconstructs_evidence_and_rejects_authority_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    benchmark_path, benchmark, qualification_path, qualification = _write_evidence(
        module, tmp_path
    )
    release_path = tmp_path / "release.md"
    release_path.write_text(_release(module, benchmark, qualification), encoding="utf-8")
    monkeypatch.setattr(module, "BENCHMARK_PATH", benchmark_path)
    monkeypatch.setattr(module, "QUALIFICATION_PATH", qualification_path)
    calls = _install_replay(
        module,
        monkeypatch,
        benchmark=benchmark,
        qualification=qualification,
    )

    result = module.validate_release(release_path)
    assert result["valid"] is True
    assert result["release_qualified"] is False
    assert calls == [
        ("qualification", ("--check",)),
        (
            "benchmark",
            ("--check", "--output", str(benchmark_path), "--json"),
        ),
    ]

    release_path.write_text(
        _release(module, benchmark, qualification).replace(
            "Production authorization: not_authorized",
            "Production authorization: authorized",
        ),
        encoding="utf-8",
    )
    with pytest.raises(module.CloseoutValidationError, match="Production authorization"):
        module.validate_release(release_path)


def test_implementation_report_binds_successor_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    benchmark_path, benchmark, qualification_path, qualification = _write_evidence(
        module, tmp_path
    )
    benchmark_cid = str(benchmark["report_cid"])
    qualification_cid = str(qualification["result_cid"])
    release_path = tmp_path / "release.md"
    release_path.write_text(_release(module, benchmark, qualification), encoding="utf-8")
    monkeypatch.setattr(module, "BENCHMARK_PATH", benchmark_path)
    monkeypatch.setattr(module, "QUALIFICATION_PATH", qualification_path)
    monkeypatch.setattr(module, "RELEASE_PATH", release_path)
    calls = _install_replay(
        module,
        monkeypatch,
        benchmark=benchmark,
        qualification=qualification,
    )
    release_sha256 = "sha256:" + hashlib.sha256(release_path.read_bytes()).hexdigest()

    successors = _successors(
        module,
        benchmark_cid=benchmark_cid,
        qualification_cid=qualification_cid,
        release_sha256=release_sha256,
    )
    successors_path = tmp_path / "successors.json"
    successors_path.write_text(json.dumps(successors), encoding="utf-8")

    report = _implementation(
        module,
        benchmark=benchmark,
        qualification=qualification,
        release_sha256=release_sha256,
        successors=successors,
    )
    report_path = tmp_path / "implementation.md"
    report_path.write_text(report, encoding="utf-8")
    result = module.validate_implementation(report_path, successors_path)
    assert result["valid"] is True
    assert result["task_implementation_complete"] is False
    assert result["test_qualification_complete"] is True
    assert [label for label, _arguments in calls] == ["qualification", "benchmark"]

    source_head = module._current_repository_truth(qualification)[0][
        "ipfs_accelerate_py"
    ]["head"]
    report_path.write_text(
        report.replace(source_head, "0" * len(source_head), 1),
        encoding="utf-8",
    )
    with pytest.raises(module.CloseoutValidationError, match="differ from Git truth"):
        module.validate_implementation(report_path, successors_path)

    report_path.write_text(
        report.replace(
            f"## {module._IMPLEMENTATION_HEADINGS[1]}\n",
            f"## {module._IMPLEMENTATION_HEADINGS[1]}\n\nAll tasks implemented.\n",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(module.CloseoutValidationError, match="contains untyped content"):
        module.validate_implementation(report_path, successors_path)

    trivial = re.sub(
        rf"(## {re.escape(module._IMPLEMENTATION_HEADINGS[0])})\n.*?"
        rf"(?=\n## {re.escape(module._IMPLEMENTATION_HEADINGS[1])})",
        r"\1\n\nEvidence.\n",
        report,
        count=1,
        flags=re.DOTALL,
    )
    report_path.write_text(trivial, encoding="utf-8")
    with pytest.raises(module.CloseoutValidationError, match="typed fields differ"):
        module.validate_implementation(report_path, successors_path)

    report_path.write_text(
        report.replace("Test success: passed_hermetic", "Test success: failed"),
        encoding="utf-8",
    )
    with pytest.raises(module.CloseoutValidationError, match="Test success"):
        module.validate_implementation(report_path, successors_path)

    report_path.write_text(report, encoding="utf-8")
    successors["tasks"] = []
    successors["successor_tasks_cid"] = module.content_identity(
        {key: item for key, item in successors.items() if key != "successor_tasks_cid"}
    )
    successors_path.write_text(json.dumps(successors), encoding="utf-8")
    with pytest.raises(module.CloseoutValidationError, match="task list is empty"):
        module.validate_implementation(report_path, successors_path)


def test_self_hashed_minimal_qualification_is_rejected_by_protected_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    benchmark_path, benchmark, qualification_path, qualification = _write_evidence(
        module, tmp_path
    )
    forged: dict[str, object] = {
        "schema": "lgcvf-independent-hermetic-qualification@1",
        "plan_cid": module.PLAN_CID,
        "passed": True,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
    }
    forged["result_cid"] = module.content_identity(forged)
    qualification_path.write_text(json.dumps(forged), encoding="utf-8")
    monkeypatch.setattr(module, "BENCHMARK_PATH", benchmark_path)
    monkeypatch.setattr(module, "QUALIFICATION_PATH", qualification_path)
    calls = _install_replay(
        module,
        monkeypatch,
        benchmark=benchmark,
        qualification=qualification,
    )

    with pytest.raises(
        module.CloseoutValidationError,
        match="qualification fields differ from the closed schema",
    ):
        module.validate_release(tmp_path / "not-reached.md")
    assert calls == [("qualification", ("--check",))]


def test_self_hashed_minimal_benchmark_is_rejected_by_protected_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    benchmark_path, benchmark, qualification_path, qualification = _write_evidence(
        module, tmp_path
    )
    forged: dict[str, object] = {
        "schema": "lgcvf-symbolic-displacement-benchmark@1",
        "cohort": "hermetic_local_execution",
        "production_authoritative": False,
        "overall_disposition": "partial",
        "release_qualified": False,
        "production_authorized": False,
    }
    forged["report_cid"] = module.content_identity(forged)
    benchmark_path.write_text(json.dumps(forged), encoding="utf-8")
    monkeypatch.setattr(module, "BENCHMARK_PATH", benchmark_path)
    monkeypatch.setattr(module, "QUALIFICATION_PATH", qualification_path)
    calls = _install_replay(
        module,
        monkeypatch,
        benchmark=benchmark,
        qualification=qualification,
    )

    with pytest.raises(
        module.CloseoutValidationError,
        match="benchmark fields differ from the closed schema",
    ):
        module.validate_release(tmp_path / "not-reached.md")
    assert [label for label, _arguments in calls] == ["qualification", "benchmark"]


def test_benchmark_execution_authority_cids_cannot_hide_behind_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    benchmark_path, benchmark, qualification_path, qualification = _write_evidence(
        module, tmp_path
    )
    forged = _clone(benchmark)
    forged["execution_evidence"]["artifact_cid"] = "cid:forged-artifact"  # type: ignore[index]
    forged["execution_evidence"][  # type: ignore[index]
        "artifact_verification_receipt_cid"
    ] = "cid:forged-verifier-receipt"
    _rehash_benchmark(module, forged)
    assert module._benchmark_projection(forged) == module._benchmark_projection(benchmark)
    benchmark_path.write_text(json.dumps(forged), encoding="utf-8")
    monkeypatch.setattr(module, "BENCHMARK_PATH", benchmark_path)
    monkeypatch.setattr(module, "QUALIFICATION_PATH", qualification_path)
    _install_replay(
        module,
        monkeypatch,
        benchmark=benchmark,
        qualification=qualification,
    )

    with pytest.raises(
        module.CloseoutValidationError,
        match="benchmark authority reconstruction differs",
    ):
        module.validate_release(tmp_path / "not-reached.md")


def test_release_binds_distinct_full_stored_and_fresh_qualification_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    benchmark_path, benchmark, qualification_path, qualification = _write_evidence(
        module, tmp_path
    )
    replayed_qualification = _clone(qualification)
    replayed_qualification["suites"][0]["duration_ms"] = 9  # type: ignore[index]
    replayed_qualification["suites"][0][  # type: ignore[index]
        "transcript_sha256"
    ] = "sha256:" + "b" * 64
    _rehash_qualification(module, replayed_qualification)
    assert replayed_qualification["result_cid"] != qualification["result_cid"]
    release_path = tmp_path / "release.md"
    release_path.write_text(
        _release(module, benchmark, qualification),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "BENCHMARK_PATH", benchmark_path)
    monkeypatch.setattr(module, "QUALIFICATION_PATH", qualification_path)
    _install_replay(
        module,
        monkeypatch,
        benchmark=benchmark,
        qualification=replayed_qualification,
    )

    result = module.validate_release(release_path)
    assert result["qualification_cid"] == qualification["result_cid"]
    assert result["qualification_replay_cid"] == replayed_qualification["result_cid"]
    qualification_authority_cid = module.content_identity(
        module._qualification_authority_evidence(qualification)
    )
    assert result["qualification_authority_cid"] == qualification_authority_cid

    release_path.write_text(
        release_path.read_text(encoding="utf-8").replace(
            f"Qualification authority CID: {qualification_authority_cid}",
            f"Qualification authority CID: {qualification['result_cid']}",
        ),
        encoding="utf-8",
    )
    with pytest.raises(module.CloseoutValidationError, match="Qualification authority CID"):
        module.validate_release(release_path)


def test_successor_contract_is_closed_dependency_complete_and_preserves_blockers() -> None:
    module = _load()
    release_sha256 = "sha256:" + "c" * 64
    successors = _successors(
        module,
        benchmark_cid="cid:benchmark",
        qualification_cid="cid:qualification",
        release_sha256=release_sha256,
    )
    assert (
        module._validate_successors(
            successors,
            benchmark_cid="cid:benchmark",
            qualification_cid="cid:qualification",
            release_sha256=release_sha256,
        )
        == successors["successor_tasks_cid"]
    )

    open_schema = _clone(successors)
    open_schema["tasks"][0]["production_authorized"] = True  # type: ignore[index]
    _rehash_successors(module, open_schema)
    with pytest.raises(module.CloseoutValidationError, match="closed schema"):
        module._validate_successors(
            open_schema,
            benchmark_cid="cid:benchmark",
            qualification_cid="cid:qualification",
            release_sha256=release_sha256,
        )

    foreign_owner = _clone(successors)
    foreign_owner["tasks"][0]["owning_repository"] = "parallel_authority"  # type: ignore[index]
    _rehash_successors(module, foreign_owner)
    with pytest.raises(module.CloseoutValidationError, match="owning_repository is invalid"):
        module._validate_successors(
            foreign_owner,
            benchmark_cid="cid:benchmark",
            qualification_cid="cid:qualification",
            release_sha256=release_sha256,
        )

    dangling = _clone(successors)
    dangling["tasks"][1]["depends_on"] = ["LGCVF-S999"]  # type: ignore[index]
    _rehash_successors(module, dangling)
    with pytest.raises(module.CloseoutValidationError, match="outside the successor task closure"):
        module._validate_successors(
            dangling,
            benchmark_cid="cid:benchmark",
            qualification_cid="cid:qualification",
            release_sha256=release_sha256,
        )

    cyclic = _clone(successors)
    cyclic["tasks"][0]["depends_on"] = ["LGCVF-S002"]  # type: ignore[index]
    _rehash_successors(module, cyclic)
    with pytest.raises(module.CloseoutValidationError, match="contain a cycle"):
        module._validate_successors(
            cyclic,
            benchmark_cid="cid:benchmark",
            qualification_cid="cid:qualification",
            release_sha256=release_sha256,
        )

    no_manual_blocker = _clone(successors)
    no_manual_blocker["tasks"][1]["status"] = "todo"  # type: ignore[index]
    no_manual_blocker["tasks"][1]["reason_codes"] = [  # type: ignore[index]
        "manual_authority_pending"
    ]
    _rehash_successors(module, no_manual_blocker)
    with pytest.raises(module.CloseoutValidationError, match="preserve the blocked_manual"):
        module._validate_successors(
            no_manual_blocker,
            benchmark_cid="cid:benchmark",
            qualification_cid="cid:qualification",
            release_sha256=release_sha256,
        )


def test_truthful_no_go_release_remains_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    benchmark_path, benchmark, qualification_path, qualification = _write_evidence(
        module, tmp_path
    )
    benchmark["overall_disposition"] = "no_go"
    _rehash_benchmark(module, benchmark)
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    release_path = tmp_path / "release.md"
    release_path.write_text(_release(module, benchmark, qualification), encoding="utf-8")
    monkeypatch.setattr(module, "BENCHMARK_PATH", benchmark_path)
    monkeypatch.setattr(module, "QUALIFICATION_PATH", qualification_path)
    _install_replay(
        module,
        monkeypatch,
        benchmark=benchmark,
        qualification=qualification,
    )

    result = module.validate_release(release_path)
    assert result["disposition"] == "no_go"
    assert result["release_qualified"] is False
    assert result["production_authorized"] is False


def test_source_baseline_survives_only_evidence_report_commit(tmp_path: Path) -> None:
    module = _load()
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    _git(repository, "config", "user.name", "LGCVF Fixture")
    source = repository / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", "source.py")
    _git(repository, "commit", "-qm", "semantic source")
    baseline = module._semantic_source_revision(repository)

    evidence_relative = (
        "docs/architecture/"
        "LOGIC_GOVERNED_COMPOSITIONAL_VERIFICATION_FABRIC_IMPLEMENTATION_REPORT.md"
    )
    evidence = repository / evidence_relative
    evidence.parent.mkdir(parents=True)
    evidence.write_text("# Partial report\n", encoding="utf-8")
    _git(repository, "add", evidence_relative)
    _git(repository, "commit", "-qm", "evidence-only report")
    assert module._semantic_source_revision(repository) == baseline

    source.write_text("VALUE = 2\n", encoding="utf-8")
    _git(repository, "add", "source.py")
    _git(repository, "commit", "-qm", "semantic source update")
    assert module._semantic_source_revision(repository) != baseline
