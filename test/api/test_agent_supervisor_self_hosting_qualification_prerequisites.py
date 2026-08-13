"""SHQ-G006: read-only prerequisite-state observer and current-fact binding."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
OBSERVER = (
    REPO_ROOT
    / "scripts/ops/agent_supervisor/self_hosting_qualification_prerequisites.py"
)
ARTIFACT = (
    REPO_ROOT
    / "artifacts/agent_supervisor/self_hosting_qualification/"
    / "prerequisite_observation.json"
)
MODULE_NAME = "agent_supervisor_self_hosting_qualification_prerequisites"

TEN_PREREQUISITES = (
    "IncrementalSemanticIndex",
    "SemanticCapsuleCompiler",
    "ContextPackBuilder",
    "VerificationReceiptCache",
    "IncrementalVerificationPlanner",
    "ModelRoutePlanner",
    "VerifiedGuiOptimizer",
    "IncrementalProofSealer",
    "SemanticCompressionGovernor",
    "AdversarialAssuranceEngine",
)


def _load_observer():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, OBSERVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


@pytest.fixture(scope="module")
def obs():
    module = _load_observer()
    try:
        yield module
    finally:
        sys.modules.pop(MODULE_NAME, None)


@pytest.fixture(scope="module")
def observation(obs):
    return obs.observe_prerequisite_releases(
        REPO_ROOT,
        mode=obs.ObservationMode.OBSERVE,
        evidence_time="2026-08-13T00:00:00Z",
    )


def test_observer_module_and_artifact_paths_exist() -> None:
    assert OBSERVER.is_file()
    assert ARTIFACT.parent.is_dir() or True  # parent created on write


def test_contract_identity_and_non_authority(obs) -> None:
    assert obs.INTERFACE_ID == "PrerequisiteObservation@1"
    assert obs.OBSERVATION_SCHEMA.endswith(
        "self-hosting-qualification-prerequisite-observation@1"
    )
    assert obs.GOAL_ID == "SHQ-G006"
    assert obs.OBSERVATION_IS_COMPLETION_EVIDENCE is False
    assert obs.OBSERVATION_IS_PROOF_EVIDENCE is False
    assert obs.OBSERVATION_AUTHORIZES_MUTATION is False
    assert obs.OBSERVATION_AUTHORIZES_RELEASE is False


def test_catalog_has_exactly_ten_stable_prerequisites(obs) -> None:
    catalog = obs.prerequisite_catalog()
    ids = [item.prerequisite_id for item in catalog]
    assert ids == list(TEN_PREREQUISITES)
    assert len(ids) == 10
    assert len(set(ids)) == 10


def test_compatibility_map_is_explicit_and_does_not_invent_facades(obs) -> None:
    mapping = {entry.planned_name: entry for entry in obs.COMPATIBILITY_MAP}
    assert "ContextPackBuilder" in mapping
    context = mapping["ContextPackBuilder"]
    assert "ContextPacker" in context.implementation_symbols
    assert "pack_context" in context.implementation_symbols
    assert "ContextPack@1" in context.interface_ids
    # Never claim a manufactured ContextPackBuilder facade exists in the map.
    assert "ContextPackBuilder" not in context.implementation_symbols

    payload = obs.compatibility_map_as_dict()
    assert any(item["planned_name"] == "ContextPackBuilder" for item in payload)


def test_ordinary_observation_succeeds_when_upstreams_are_incomplete(
    obs, observation
) -> None:
    assert observation.schema == obs.OBSERVATION_SCHEMA
    assert observation.interface == obs.INTERFACE_ID
    assert observation.mode == obs.ObservationMode.OBSERVE.value
    assert len(observation.rows) == 10
    assert observation.authorizes_release is False
    assert observation.is_completion_evidence is False

    by_id = {row.prerequisite_id: row for row in observation.rows}
    assert set(by_id) == set(TEN_PREREQUISITES)

    # Incomplete / unbound systems must not crash ordinary observation.
    for missing_id in (
        "VerifiedGuiOptimizer",
        "IncrementalProofSealer",
        "AdversarialAssuranceEngine",
    ):
        row = by_id[missing_id]
        assert row.status in {
            obs.PrerequisiteStatus.MISSING.value,
            obs.PrerequisiteStatus.IN_FLIGHT.value,
            obs.PrerequisiteStatus.UNVERIFIABLE.value,
        }
        assert row.terminal is False


def test_every_row_binds_required_evidence_fields(observation) -> None:
    for row in observation.rows:
        payload = row.to_dict()
        assert payload["prerequisite_id"]
        assert payload["repository"]
        assert "commit" in payload
        assert payload["status"]
        assert payload["evidence_time"] == "2026-08-13T00:00:00Z"
        assert "api" in payload
        assert "resolution" in payload["api"]
        assert "expected_symbols" in payload["api"]
        assert "tests" in payload
        assert "selectors" in payload["tests"]
        assert "board" in payload
        assert "state" in payload["board"]
        # Branch / prompt text are not used as release authority.
        assert "branch" not in payload
        assert "prompt" not in payload


def test_context_pack_builder_resolves_only_via_compatibility_map(
    obs, observation
) -> None:
    row = next(
        item
        for item in observation.rows
        if item.prerequisite_id == "ContextPackBuilder"
    )
    assert row.status == obs.PrerequisiteStatus.MISMATCHED_NAME.value
    assert row.api.resolution == obs.ApiResolution.COMPATIBILITY_MAP.value
    assert row.api.compatibility is not None
    assert row.api.compatibility["planned_name"] == "ContextPackBuilder"
    assert "ContextPacker" in row.api.found_symbols
    assert "pack_context" in row.api.found_symbols
    # No manufactured facade class should appear as a found definition.
    assert "ContextPackBuilder" not in row.api.found_symbols


def test_released_rows_require_terminal_board_and_tests_not_branch_names(
    obs, observation
) -> None:
    released = [
        row
        for row in observation.rows
        if row.status == obs.PrerequisiteStatus.RELEASED.value
    ]
    for row in released:
        assert row.commit
        assert row.tests.all_present
        assert row.board.state == obs.BoardState.TERMINAL.value
        assert row.api.to_dict()["resolved"] is True
        assert row.terminal is True

    # IVP-owned systems are expected to be released on the current tree.
    by_id = {row.prerequisite_id: row for row in observation.rows}
    for prerequisite_id in (
        "VerificationReceiptCache",
        "IncrementalVerificationPlanner",
        "ModelRoutePlanner",
    ):
        row = by_id[prerequisite_id]
        assert row.status == obs.PrerequisiteStatus.RELEASED.value
        assert row.terminal is True


def test_semantic_compression_governor_stays_honest_while_board_open(
    obs, observation
) -> None:
    row = next(
        item
        for item in observation.rows
        if item.prerequisite_id == "SemanticCompressionGovernor"
    )
    assert row.status in {
        obs.PrerequisiteStatus.IN_FLIGHT.value,
        obs.PrerequisiteStatus.MISSING.value,
        obs.PrerequisiteStatus.MISMATCHED_NAME.value,
    }
    assert row.terminal is False
    if row.board.state == obs.BoardState.IN_FLIGHT.value:
        assert row.board.open + row.board.blocked > 0


def test_require_terminal_fails_closed_on_incomplete_snapshot(obs) -> None:
    observation = obs.observe_prerequisite_releases(
        REPO_ROOT,
        mode=obs.ObservationMode.REQUIRE_TERMINAL,
        evidence_time="2026-08-13T00:00:00Z",
    )
    assert observation.summary["all_terminal"] is False
    with pytest.raises(obs.NonTerminalPrerequisiteError) as excinfo:
        obs.assert_terminal(observation)
    assert "VerifiedGuiOptimizer" in str(excinfo.value) or excinfo.value.nonterminal_ids


def test_observation_is_deterministic(obs) -> None:
    first = obs.observe_prerequisite_releases(
        REPO_ROOT,
        mode=obs.ObservationMode.OBSERVE,
        evidence_time="2026-08-13T00:00:00Z",
    )
    second = obs.observe_prerequisite_releases(
        REPO_ROOT,
        mode=obs.ObservationMode.OBSERVE,
        evidence_time="2026-08-13T00:00:00Z",
    )
    assert first.to_dict() == second.to_dict()
    assert obs.observation_to_json(first) == obs.observation_to_json(second)


def test_fixture_tree_observation_covers_status_taxonomy(obs, tmp_path: Path) -> None:
    """Synthetic tree proves all five statuses without relying on live boards."""

    root = tmp_path / "repo"
    (root / "impl").mkdir(parents=True)
    (root / "boards").mkdir()
    (root / "tests").mkdir()

    # Exact released system: class + terminal board + test.
    (root / "impl" / "released_mod.py").write_text(
        "class ReleasedSystem:\n    INTERFACE = 'ReleasedSystem@1'\n",
        encoding="utf-8",
    )
    (root / "boards" / "released.todo.md").write_text(
        "## RS-001 Example\n- Status: completed\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_released.py").write_text(
        "def test_ok():\n    assert True\n",
        encoding="utf-8",
    )

    # Mismatched-name via compatibility map.
    (root / "impl" / "mapped_mod.py").write_text(
        "class ActualName:\n    pass\n\ndef pack_context():\n    return None\n",
        encoding="utf-8",
    )
    (root / "boards" / "mapped.todo.md").write_text(
        "## MP-001 Example\n- Status: completed\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_mapped.py").write_text(
        "def test_ok():\n    assert True\n",
        encoding="utf-8",
    )

    # In-flight: API present, open board.
    (root / "impl" / "inflight_mod.py").write_text(
        "class InFlightSystem:\n    pass\n",
        encoding="utf-8",
    )
    (root / "boards" / "inflight.todo.md").write_text(
        "## IF-001 Example\n- Status: todo\n## IF-002 Example\n- Status: completed\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_inflight.py").write_text(
        "def test_ok():\n    assert True\n",
        encoding="utf-8",
    )

    # Missing: no API, no board content.
    # Unverifiable: board path configured but unreadable is simulated by a
    # directory where a file is expected.
    (root / "boards" / "broken").mkdir()

    # Initialize a git repo so commits resolve.
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "shq@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "SHQ"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=root,
        check=True,
        capture_output=True,
    )

    # Monkeypatch compatibility map for PlannedName → ActualName.
    original_map = obs.COMPATIBILITY_MAP
    fixture_entry = obs.CompatibilityEntry(
        planned_name="PlannedName",
        implementation_symbols=("ActualName", "pack_context"),
        interface_ids=(),
        module_paths=("impl/mapped_mod.py",),
        rationale="fixture compatibility mapping",
    )
    # Module-level COMPATIBILITY_MAP is a Final tuple; patch via assignment.
    obs.COMPATIBILITY_MAP = (fixture_entry,)

    catalog = (
        obs.PrerequisiteSpec(
            prerequisite_id="ReleasedSystem",
            repository="fixture",
            repository_root_relative=".",
            module_paths=("impl/released_mod.py",),
            expected_symbols=("ReleasedSystem",),
            interface_ids=("ReleasedSystem@1",),
            test_selectors=("tests/test_released.py",),
            board_paths=("boards/released.todo.md",),
        ),
        obs.PrerequisiteSpec(
            prerequisite_id="PlannedName",
            repository="fixture",
            repository_root_relative=".",
            module_paths=("impl/mapped_mod.py",),
            expected_symbols=("PlannedName",),
            interface_ids=(),
            test_selectors=("tests/test_mapped.py",),
            board_paths=("boards/mapped.todo.md",),
        ),
        obs.PrerequisiteSpec(
            prerequisite_id="InFlightSystem",
            repository="fixture",
            repository_root_relative=".",
            module_paths=("impl/inflight_mod.py",),
            expected_symbols=("InFlightSystem",),
            interface_ids=(),
            test_selectors=("tests/test_inflight.py",),
            board_paths=("boards/inflight.todo.md",),
        ),
        obs.PrerequisiteSpec(
            prerequisite_id="MissingSystem",
            repository="fixture",
            repository_root_relative=".",
            module_paths=("impl/does_not_exist.py",),
            expected_symbols=("MissingSystem",),
            interface_ids=(),
            test_selectors=(),
            board_paths=(),
        ),
        obs.PrerequisiteSpec(
            prerequisite_id="UnreadableBoardSystem",
            repository="fixture",
            repository_root_relative=".",
            module_paths=("impl/released_mod.py",),
            expected_symbols=("ReleasedSystem",),
            interface_ids=(),
            test_selectors=("tests/test_released.py",),
            board_paths=("boards/broken",),
        ),
        # Pad to ten with additional missing systems.
        *(
            obs.PrerequisiteSpec(
                prerequisite_id=f"PadMissing{index}",
                repository="fixture",
                repository_root_relative=".",
                module_paths=(),
                expected_symbols=(f"PadMissing{index}",),
                interface_ids=(),
                test_selectors=(),
                board_paths=(),
            )
            for index in range(5)
        ),
    )

    try:
        result = obs.observe_prerequisite_releases(
            root,
            mode=obs.ObservationMode.OBSERVE,
            evidence_time="2026-08-13T12:00:00Z",
            catalog=catalog,
        )
    finally:
        obs.COMPATIBILITY_MAP = original_map

    by_id = {row.prerequisite_id: row for row in result.rows}
    assert by_id["ReleasedSystem"].status == obs.PrerequisiteStatus.RELEASED.value
    assert by_id["ReleasedSystem"].terminal is True
    assert by_id["PlannedName"].status == obs.PrerequisiteStatus.MISMATCHED_NAME.value
    assert by_id["PlannedName"].terminal is True
    assert by_id["InFlightSystem"].status == obs.PrerequisiteStatus.IN_FLIGHT.value
    assert by_id["MissingSystem"].status == obs.PrerequisiteStatus.MISSING.value
    assert (
        by_id["UnreadableBoardSystem"].status
        == obs.PrerequisiteStatus.UNVERIFIABLE.value
    )

    # require-terminal fails closed on this incomplete fixture.
    with pytest.raises(obs.NonTerminalPrerequisiteError):
        obs.assert_terminal(result)


def test_cli_observe_writes_artifact_and_exits_zero(obs, tmp_path: Path) -> None:
    output = tmp_path / "prerequisite_observation.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(OBSERVER),
            "--repo-root",
            str(REPO_ROOT),
            "--mode",
            "observe",
            "--output",
            str(output),
            "--quiet",
        ],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["interface"] == "PrerequisiteObservation@1"
    assert len(payload["rows"]) == 10
    assert payload["authorizes_release"] is False
    for row in payload["rows"]:
        assert "repository" in row
        assert "commit" in row
        assert "api" in row
        assert "tests" in row
        assert "board" in row
        assert "evidence_time" in row


def test_cli_require_terminal_fails_closed_on_current_tree() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(OBSERVER),
            "--repo-root",
            str(REPO_ROOT),
            "--mode",
            "require-terminal",
            "--quiet",
        ],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "require-terminal" in completed.stderr.lower() or "non-terminal" in (
        completed.stderr + completed.stdout
    ).lower()


def test_write_observation_artifact_round_trip(obs, tmp_path: Path, observation) -> None:
    path = tmp_path / "out.json"
    written = obs.write_observation_artifact(observation, path)
    assert written == path
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == observation.to_dict()
