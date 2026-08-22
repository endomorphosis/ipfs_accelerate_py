from __future__ import annotations

import json
import os
import subprocess

import pytest
import scripts.materialize_agent_supervisor_autonomous_meta_controller_board as materializer_module
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from scripts.materialize_agent_supervisor_autonomous_meta_controller_board import (
    BASELINE_COMPLETION_RECEIPT_SCHEMA,
    BASELINE_VALIDATION_EVIDENCE_KIND,
    BASELINE_VALIDATION_SET_EVIDENCE_KIND,
    MAX_VALIDATION_OUTPUT_BYTES,
    PRELAUNCH_COMPLETED_TASK_IDS,
    PRELAUNCH_QUALIFIED_TASK_IDS,
    PRELAUNCH_READY_TASK_IDS,
    PRELAUNCH_VALIDATION_SET_EVIDENCE_KIND,
    TRUSTED_PYTHON,
    TRUSTED_VALIDATION_PATH,
    MaterializationError,
    _run_validation,
    _verify_source,
    _write_runtime_receipt,
    build_population,
    materialize,
)


def _population() -> dict[str, object]:
    return build_population(
        source_head="a" * 40,
        source_tree="b" * 40,
    )


def _identity() -> tuple[str, str]:
    return "a" * 40, "b" * 40


def test_qualified_materialization_completes_baseline_from_exact_validation_evidence(
    tmp_path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def validation_runner(argv):
        normalized = tuple(str(item) for item in argv)
        calls.append(normalized)
        return subprocess.CompletedProcess(
            args=list(normalized),
            returncode=0,
            stdout=f"passed:{len(calls)}".encode(),
            stderr=b"",
        )

    database = tmp_path / "control.duckdb"
    receipt = materialize(
        database,
        _population(),
        validation_runner=validation_runner,
        source_identity_reader=_identity,
    )

    assert receipt["baseline_qualified"] is True
    assert receipt["baseline_completion_receipt_id"].startswith("bagu")
    assert receipt["prelaunch_qualified"] is True
    assert receipt["prelaunch_completed_task_aliases"] == list(PRELAUNCH_COMPLETED_TASK_IDS)
    assert receipt["ready_task_aliases"] == list(PRELAUNCH_READY_TASK_IDS)
    expected_tasks = {item["task_alias"]: item for item in _population()["tasks"]}
    expected_calls = [
        tuple(validation["argv"])
        for task_alias in PRELAUNCH_QUALIFIED_TASK_IDS
        for validation in expected_tasks[task_alias]["validations"]
    ]
    assert calls == expected_calls
    expected_validations = expected_tasks["APMC-000"]["validations"]

    source = DatabaseTaskSource(database, install_schema=False)
    try:
        baseline = source.get_task("APMC-000")
        assert baseline is not None
        assert baseline.status == "completed"
        assert baseline.revision == 2
        assert baseline.body["completion_receipt"]["schema"] == BASELINE_COMPLETION_RECEIPT_SCHEMA
        completion = baseline.body["completion_receipt"]
        assert completion["validation_count"] == len(expected_validations)
        assert len(set(completion["validation_evidence_digests"])) == len(expected_validations)
        evidence = source.current_evidence_for_task(baseline.task_cid)
        assert sum(
            item["evidence_kind"] == BASELINE_VALIDATION_EVIDENCE_KIND for item in evidence
        ) == len(expected_validations)
        assert (
            sum(item["evidence_kind"] == BASELINE_VALIDATION_SET_EVIDENCE_KIND for item in evidence)
            == 1
        )
        assert [item.task_alias for item in source.ready_tasks(limit=100).tasks] == [
            *PRELAUNCH_READY_TASK_IDS,
        ]
        for task_alias in PRELAUNCH_COMPLETED_TASK_IDS:
            qualified = source.get_task(task_alias)
            assert qualified is not None
            assert qualified.status == "completed"
            assert qualified.revision == 2
        benchmark = source.get_task("APMC-018")
        assert benchmark is not None
        assert benchmark.body["completion_receipt"]["benchmark_measurement_status"] == "not_run"
        assert benchmark.body["completion_receipt"]["promotion_eligible"] is False
    finally:
        source.close()


def test_prelaunch_qualification_fails_closed_before_any_completion(
    tmp_path,
) -> None:
    population = _population()
    validation_count = sum(
        len(task["validations"])
        for task in population["tasks"]
        if task["task_alias"] in PRELAUNCH_QUALIFIED_TASK_IDS
    )
    calls = 0

    def validation_runner(argv):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            args=list(argv),
            returncode=1 if calls == validation_count else 0,
            stdout=b"",
            stderr=b"validation failed",
        )

    database = tmp_path / "control.duckdb"
    with pytest.raises(MaterializationError, match="current-tree validation .* failed"):
        materialize(
            database,
            population,
            validation_runner=validation_runner,
            source_identity_reader=_identity,
        )
    assert calls == validation_count

    source = DatabaseTaskSource(database, install_schema=False)
    try:
        for task_alias in PRELAUNCH_COMPLETED_TASK_IDS:
            task = source.get_task(task_alias)
            assert task is not None
            assert task.status == "todo"
            assert task.revision == 1
            assert source.current_evidence_for_task(task.task_cid) == ()
    finally:
        source.close()


def test_source_identity_change_rejects_before_any_evidence_write(tmp_path) -> None:
    identity_calls = 0

    def identity_reader():
        nonlocal identity_calls
        identity_calls += 1
        if identity_calls == 1:
            return _identity()
        return "c" * 40, "d" * 40

    def validation_runner(argv):
        return subprocess.CompletedProcess(list(argv), 0, stdout=b"ok", stderr=b"")

    database = tmp_path / "control.duckdb"
    with pytest.raises(MaterializationError, match="commit/tree changed"):
        materialize(
            database,
            _population(),
            validation_runner=validation_runner,
            source_identity_reader=identity_reader,
        )

    source = DatabaseTaskSource(database, install_schema=False)
    try:
        baseline = source.get_task("APMC-000")
        assert baseline is not None
        assert baseline.status == "todo"
        assert source.current_evidence_for_task(baseline.task_cid) == ()
    finally:
        source.close()


def test_unqualified_population_is_not_a_runtime_valid_board(tmp_path) -> None:
    population = _population()
    database = tmp_path / "control.duckdb"
    source = DatabaseTaskSource(
        database,
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        install_schema=True,
    )
    try:
        source.materialize(
            population,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        )
        with pytest.raises(MaterializationError, match="not evidence-qualified"):
            _verify_source(source, population)
    finally:
        source.close()


def test_validation_process_has_bounded_output_and_exact_scrubbed_environment(
    monkeypatch,
    tmp_path,
) -> None:
    shadow = tmp_path / "python3"
    shadow.write_text("#!/bin/sh\necho forged-shadow\n", encoding="utf-8")
    shadow.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "state-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    result = _run_validation(
        (
            "python3",
            "-c",
            (
                "import os; print(int('IPFS_ACCELERATE_AGENT_QUACK_TOKEN' in os.environ), "
                "int('OPENAI_API_KEY' in os.environ), os.environ.get('PATH'))"
            ),
        )
    )
    assert result.returncode == 0
    assert result.args[0] == TRUSTED_PYTHON
    assert result.stdout.strip() == f"0 0 {TRUSTED_VALIDATION_PATH}".encode()

    with pytest.raises(MaterializationError, match="output exceeds"):
        _run_validation(
            (
                "python3",
                "-c",
                f"import sys; sys.stdout.buffer.write(b'x' * {MAX_VALIDATION_OUTPUT_BYTES + 1})",
            )
        )


def test_verifier_rejects_live_declaration_reduction_and_extra_evidence(tmp_path) -> None:
    population = _population()

    def validation_runner(argv):
        return subprocess.CompletedProcess(list(argv), 0, stdout=b"ok", stderr=b"")

    database = tmp_path / "control.duckdb"
    materialize(
        database,
        population,
        validation_runner=validation_runner,
        source_identity_reader=_identity,
    )
    source = DatabaseTaskSource(
        database,
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        evidence_freshness_seconds=0,
        install_schema=False,
    )
    try:
        # Exact-tree qualification is content invalidated, not age invalidated.
        with source.intent._connection(write=True) as connection:  # noqa: SLF001
            connection.execute(
                "UPDATE evidence_nodes SET created_at = ?",
                ["2000-01-01T00:00:00+00:00"],
            )
        assert _verify_source(source, population)["prelaunch_qualified"] is True

        task = source.get_task("APMC-001")
        assert task is not None
        with source.intent._connection(write=True) as connection:  # noqa: SLF001
            connection.execute(
                "UPDATE task_acceptance SET evidence_policy_json = ? "
                "WHERE task_cid = ? AND ordinal = 0",
                [json.dumps({"criterion": "weakened"}), task.task_cid],
            )
            connection.execute(
                "UPDATE task_validations SET argv_json = ?, policy_json = ? "
                "WHERE task_cid = ? AND ordinal = 0",
                [json.dumps(["true"]), json.dumps({}), task.task_cid],
            )
            connection.execute(
                "DELETE FROM validation_results WHERE task_cid = ?",
                [task.task_cid],
            )
            connection.execute(
                "DELETE FROM validation_runs WHERE task_cid = ?",
                [task.task_cid],
            )
            connection.execute(
                "DELETE FROM completion_receipts WHERE task_cid = ?",
                [task.task_cid],
            )
            connection.execute(
                "UPDATE tasks SET identity_json = ?, extension_schema = ?, "
                "extension_json = ? WHERE task_cid = ?",
                [
                    json.dumps({"forged": True}),
                    "forged@1",
                    json.dumps({"authority": True}),
                    task.task_cid,
                ],
            )
            connection.execute(
                "UPDATE goals SET title = ? WHERE goal_cid = ?",
                ["forged goal title", task.goal_cid],
            )
            connection.execute(
                "INSERT INTO objectives (objective_id, objective_alias, "
                "parent_objective_id, title, status, priority, created_at, "
                "updated_at, revision, body_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    "forged-objective",
                    "forged-objective",
                    "",
                    "forged",
                    "open",
                    "P0",
                    "2000-01-01T00:00:00+00:00",
                    "2000-01-01T00:00:00+00:00",
                    1,
                    json.dumps({}),
                ],
            )
            connection.execute(
                "INSERT INTO plans (plan_cid, goal_cid, plan_alias, status, "
                "created_at, updated_at, revision, body_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    "forged-plan",
                    task.goal_cid,
                    "forged-plan",
                    "active",
                    "2000-01-01T00:00:00+00:00",
                    "2000-01-01T00:00:00+00:00",
                    1,
                    json.dumps({}),
                ],
            )
        source.record_evidence(
            task_cid=task.task_cid,
            evidence_kind=PRELAUNCH_VALIDATION_SET_EVIDENCE_KIND,
            digest="sha256:" + ("c" * 64),
            body={"forged": True},
        )
        source.record_evidence(
            task_cid=task.task_cid,
            evidence_kind="validation",
            digest="sha256:" + ("d" * 64),
            body={"forged": True},
        )

        with pytest.raises(MaterializationError) as raised:
            _verify_source(source, population)
        message = str(raised.value)
        assert "APMC-001 acceptance declarations changed" in message
        assert "APMC-001 validation declarations changed" in message
        assert "APMC-001 current evidence population is not exact" in message
        assert "APMC-001 validation-set evidence is not current and exact" in message
        assert "APMC-001 identity/extension authority changed" in message
        assert "APMC-001 canonical validation authority is incomplete" in message
        assert "APMC-001 canonical completion authority is not exact" in message
        assert "APMC-G020 goal projection changed" in message
        assert "objective/goal/plan/task population count changed" in message
    finally:
        source.close()


def test_runtime_receipt_publish_rejects_parent_and_final_symlinks(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(materializer_module, "REPO_ROOT", tmp_path)
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir(mode=0o700)

    (state / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(MaterializationError, match="parent may not traverse"):
        _write_runtime_receipt("state/link/escaped.json", {"accepted": False})
    assert not (outside / "escaped.json").exists()

    escaped_target = outside / "final-target.json"
    (state / "final.json").symlink_to(escaped_target)
    with pytest.raises(MaterializationError, match="target already exists"):
        _write_runtime_receipt("state/final.json", {"accepted": False})
    assert not escaped_target.exists()

    receipt = _write_runtime_receipt("state/receipt.json", {"accepted": True})
    assert json.loads(receipt.read_text(encoding="utf-8")) == {"accepted": True}
    assert os.stat(receipt).st_mode & 0o777 == 0o600
    with pytest.raises(MaterializationError, match="target already exists"):
        _write_runtime_receipt("state/receipt.json", {"accepted": False})
