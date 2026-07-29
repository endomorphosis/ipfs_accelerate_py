from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.contract_analysis.execution_profile import (
    AnalysisExecutionProfile,
    CapabilitySnapshot,
    ExecutionProfileError,
    PROFILE_SCHEMA,
    RESOURCE_BOUNDS_SCHEMA,
    ResourceBudget,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "datasets_contract_analysis"
    / "policy"
    / "analyzer-profile-v1.json"
)
RESOURCE_BOUNDS_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "datasets_contract_analysis"
    / "policy"
    / "resource-bounds-v1.json"
)


def _profile() -> AnalysisExecutionProfile:
    return AnalysisExecutionProfile.load(
        POLICY_PATH, repository_root=REPOSITORY_ROOT
    )


def _matching_snapshot(
    profile: AnalysisExecutionProfile, repository: Path
) -> CapabilitySnapshot:
    return CapabilitySnapshot(
        tool_identities={tool.name: tool.identity for tool in profile.tools},
        lock_identities={lock.path: lock.identity for lock in profile.locks},
        environment_names=("LANG", "PATH", "PYTHONHASHSEED", "TZ"),
        read_paths=(str(repository / "ipfs_datasets_py"),),
        write_paths=(
            str(repository / "data/datasets_contract_analysis/runtime/result.json"),
        ),
    )


def test_reviewed_policy_is_strict_canonical_and_binds_every_required_identity() -> None:
    profile = _profile()

    assert profile.goal_id == "DSCON-G050"
    assert profile.to_dict()["schema"] == PROFILE_SCHEMA
    assert profile.to_dict()["resource_bounds"]["schema"] == RESOURCE_BOUNDS_SCHEMA
    assert (
        profile.resource_bounds_evidence
        == "data/datasets_contract_analysis/policy/resource-bounds-v1.json"
    )
    assert {
        "python",
        "node",
        "parser",
        "typescript",
        "solver",
        "proof",
    }.issubset({role for tool in profile.tools for role in tool.roles})
    assert all(tool.version and tool.identity.startswith("sha256:") for tool in profile.tools)
    assert all(lock.identity.startswith("sha256:") for lock in profile.locks)
    assert profile.sandbox.network == "deny"
    assert profile.sandbox.auto_install == "deny"
    assert profile.sandbox.home_cache == "deny"
    assert profile.sandbox.credentials == "deny"

    restored = AnalysisExecutionProfile.from_json(profile.to_json())
    assert restored == profile
    assert restored.content_identity == profile.content_identity
    with pytest.raises(FrozenInstanceError):
        profile.profile_id = "forged"  # type: ignore[misc]


def test_resource_profile_enforces_every_objective_dimension_and_fails_closed() -> None:
    profile = _profile()
    budget = profile.resources
    expected = {
        "max_blob_bytes",
        "max_files",
        "max_ast_nodes",
        "max_edges",
        "max_scc_nodes",
        "max_recursion_depth",
        "max_timeout_ms",
        "max_memory_bytes",
        "max_proof_bytes",
        "max_receipt_bytes",
        "max_findings",
        "max_tasks",
        "max_prompt_bytes",
        "max_prompt_tokens",
    }
    assert expected.issubset(profile.to_dict()["resource_bounds"])
    assert budget.max_file_count == budget.max_files
    assert budget.max_graph_edges == budget.max_edges

    usage = {
        name.removeprefix("max_"): getattr(budget, name)
        for name in expected
        if name not in {"max_files", "max_edges"}
    }
    usage.update(files=budget.max_files, edges=budget.max_edges)
    assert profile.validate_usage(usage).ok

    for usage_name, field_name in ResourceBudget._USAGE_ALIASES.items():
        result = profile.validate_usage(
            {usage_name: getattr(budget, field_name) + 1}
        )
        assert not result.ok
        assert result.disposition == "incomplete"
        assert result.exhausted_resources == (field_name,)
        proof_result = profile.validate_usage(
            {usage_name: getattr(budget, field_name) + 1},
            proof_required=True,
        )
        assert proof_result.disposition == "unknown"
        assert not proof_result.complete

    with pytest.raises(ExecutionProfileError, match="unsupported resource"):
        profile.validate_usage({"unbounded_magic": 1})
    with pytest.raises(ExecutionProfileError, match="non-negative"):
        profile.validate_usage({"files": -1})


def test_standalone_resource_evidence_is_closed_and_exactly_bound() -> None:
    profile = _profile()
    payload = json.loads(RESOURCE_BOUNDS_PATH.read_text(encoding="utf-8"))

    assert payload == profile.to_dict()["resource_bounds"]
    assert (
        profile.validate_resource_bounds_evidence(
            repository_root=REPOSITORY_ROOT
        )
        == profile.resources
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"max_memory_bytes": 1}, "exactly match"),
        ({"class": "unreviewed"}, "class must match"),
        ({"unbounded_fallback": True}, "unsupported fields"),
    ],
)
def test_resource_evidence_drift_is_rejected(
    tmp_path: Path, mutation: dict[str, object], error: str
) -> None:
    profile = _profile()
    evidence = tmp_path / profile.resource_bounds_evidence
    evidence.parent.mkdir(parents=True)
    payload = json.loads(RESOURCE_BOUNDS_PATH.read_text(encoding="utf-8"))
    payload.update(mutation)
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExecutionProfileError, match=error):
        profile.validate_resource_bounds_evidence(repository_root=tmp_path)


def test_resource_evidence_symlink_cannot_escape_repository(tmp_path: Path) -> None:
    profile = _profile()
    repository = tmp_path / "repo"
    evidence = repository / profile.resource_bounds_evidence
    outside = tmp_path / "outside"
    evidence.parent.mkdir(parents=True)
    outside.mkdir()
    evidence.symlink_to(outside / "resource-bounds.json")

    with pytest.raises(ExecutionProfileError, match="escapes the repository"):
        profile.validate_resource_bounds_evidence(repository_root=repository)


def test_matching_attested_capabilities_are_accepted_without_tool_execution(
    tmp_path: Path,
) -> None:
    profile = _profile()
    repository = tmp_path / "repo"
    for relative in (
        "ipfs_datasets_py",
        "data/datasets_contract_analysis/runtime",
        "data/datasets_contract_analysis/scans",
        "data/datasets_contract_analysis/receipts",
    ):
        (repository / relative).mkdir(parents=True, exist_ok=True)

    result = profile.validate(_matching_snapshot(profile, repository), repository_root=repository)

    assert result.ok
    assert result.disposition == "pass"
    assert result.violations == ()
    assert result.unavailable_capabilities == ()


@pytest.mark.parametrize(
    ("override", "violation"),
    [
        ({"network_enabled": True}, "network_enabled"),
        ({"auto_install_enabled": True}, "auto_install_enabled"),
        ({"home_cache_enabled": True}, "home_cache_enabled"),
        ({"credential_names": ("OPENAI_API_KEY",)}, "credentials_present"),
        (
            {"environment_names": ("LANG", "PATH", "AWS_SECRET_ACCESS_KEY")},
            "credential_environment_present",
        ),
        (
            {"environment_names": ("LANG", "PATH", "HOME")},
            "home_cache_environment_present",
        ),
        (
            {"environment_names": ("LANG", "PATH", "UNREVIEWED_AMBIENT")},
            "ambient_environment_present",
        ),
    ],
)
def test_unsafe_ambient_capabilities_are_rejected_without_leaking_values(
    tmp_path: Path, override: dict[str, object], violation: str
) -> None:
    profile = _profile()
    repository = tmp_path / "repo"
    (repository / "data/datasets_contract_analysis/runtime").mkdir(
        parents=True, exist_ok=True
    )
    baseline = {
        "tool_identities": {tool.name: tool.identity for tool in profile.tools},
        "lock_identities": {lock.path: lock.identity for lock in profile.locks},
        "environment_names": ("LANG", "PATH"),
        "read_paths": (),
        "write_paths": (),
    }
    baseline.update(override)

    result = profile.validate(
        CapabilitySnapshot(**baseline), repository_root=repository
    )

    assert not result.safe
    assert not result.ok
    assert result.disposition == "rejected"
    assert violation in result.violations
    assert "secret-value" not in repr(result)


def test_missing_tool_is_a_capability_fact_and_never_requests_auto_install(
    tmp_path: Path,
) -> None:
    profile = _profile()
    repository = tmp_path / "repo"
    snapshot = _matching_snapshot(profile, repository)
    identities = dict(snapshot.tool_identities)
    identities.pop("typescript-parser")

    result = profile.validate(
        CapabilitySnapshot(
            tool_identities=identities,
            lock_identities=snapshot.lock_identities,
            unavailable_tools=("typescript-parser",),
            environment_names=snapshot.environment_names,
        ),
        repository_root=repository,
        proof_required=True,
    )

    assert result.safe
    assert not result.complete
    assert not result.ok
    assert result.disposition == "unknown"
    assert result.unavailable_capabilities == ("tool:typescript-parser",)
    assert "auto_install" not in result.unavailable_capabilities


def test_identity_mismatch_and_write_root_escape_are_rejected(tmp_path: Path) -> None:
    profile = _profile()
    repository = tmp_path / "repo"
    outside = tmp_path / "outside"
    outside.mkdir()
    snapshot = _matching_snapshot(profile, repository)
    tools = dict(snapshot.tool_identities)
    tools["cvc5"] = "sha256:" + "0" * 64

    result = profile.validate(
        CapabilitySnapshot(
            tool_identities=tools,
            lock_identities=snapshot.lock_identities,
            environment_names=snapshot.environment_names,
            write_paths=(str(outside / "escaped.json"),),
        ),
        repository_root=repository,
    )

    assert result.disposition == "rejected"
    assert "tool_identity_mismatch:cvc5" in result.violations
    assert "write_root_escape" in result.violations


def test_existing_symlink_cannot_escape_a_write_root(tmp_path: Path) -> None:
    profile = _profile()
    repository = tmp_path / "repo"
    permitted = repository / "data/datasets_contract_analysis/runtime"
    outside = tmp_path / "outside"
    permitted.mkdir(parents=True)
    outside.mkdir()
    escape = permitted / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    snapshot = _matching_snapshot(profile, repository)

    result = profile.validate(
        CapabilitySnapshot(
            tool_identities=snapshot.tool_identities,
            lock_identities=snapshot.lock_identities,
            environment_names=snapshot.environment_names,
            write_paths=(str(escape / "result.json"),),
        ),
        repository_root=repository,
    )

    assert result.disposition == "rejected"
    assert "write_root_escape" in result.violations


def test_observer_only_records_capabilities_and_never_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _profile()
    executable = tmp_path / "tool"
    executable.write_bytes(b"reviewed tool")
    lock_path = tmp_path / profile.locks[0].path
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(b"reviewed lock")
    calls: list[str] = []

    def fake_which(locator: str) -> str | None:
        calls.append(locator)
        return str(executable) if locator == "python3.12" else None

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.contract_analysis.execution_profile.shutil.which",
        fake_which,
    )
    snapshot = CapabilitySnapshot.observe(
        profile,
        repository_root=tmp_path,
        environment={"LANG": "C.UTF-8", "OPENAI_API_KEY": "secret-value"},
    )

    assert calls == [
        tool.locator for tool in profile.tools if tool.kind == "executable"
    ]
    assert snapshot.tool_identities["cpython"].startswith("sha256:")
    assert "node" in snapshot.unavailable_tools
    assert "typescript-parser" in snapshot.unavailable_tools
    assert snapshot.credential_names == ("OPENAI_API_KEY",)
    assert "secret-value" not in repr(snapshot)


def test_closed_profile_decoder_rejects_unknown_fields_and_unbounded_limits() -> None:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload["allow_network_for_missing_tools"] = True
    with pytest.raises(ExecutionProfileError, match="unsupported fields"):
        AnalysisExecutionProfile.from_dict(payload)

    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload["resource_bounds"]["max_tasks"] = 0
    with pytest.raises(ExecutionProfileError, match="positive integer"):
        AnalysisExecutionProfile.from_dict(payload)

    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    payload["sandbox"]["network"] = "allow"
    with pytest.raises(ExecutionProfileError, match="must be 'deny'"):
        AnalysisExecutionProfile.from_dict(payload)
