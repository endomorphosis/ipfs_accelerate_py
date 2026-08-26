from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    eaaef_reconciliation_lifecycle as lifecycle,
)

_REAL_INSPECT_CURRENT_REPOSITORY_FOREST = lifecycle.inspect_current_repository_forest
_REAL_VERIFY_IMPORTED_CASF_SOURCE = lifecycle.verify_imported_casf_source
_TEST_IMPORT_EVIDENCE_CID = "sha256:" + "9" * 64


def _deterministic_did(fill: int) -> str:
    return ed25519_did_key(bytes([fill]) * 32)


def _base58btc(value: bytes) -> str:
    alphabet = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    zeroes = len(value) - len(value.lstrip(b"\0"))
    integer = int.from_bytes(value, "big")
    encoded = bytearray()
    while integer:
        integer, remainder = divmod(integer, 58)
        encoded.append(alphabet[remainder])
    return (b"1" * zeroes + bytes(reversed(encoded))).decode("ascii")


OWNER_IDENTITY_DID = _deterministic_did(1)
OPERATOR_IDENTITY_DID = _deterministic_did(2)
SECURITY_REVIEWER_IDENTITY_DID = _deterministic_did(3)
CAPABILITY_REVIEWER_IDENTITY_DID = _deterministic_did(4)
REMOTE_REVIEWER_IDENTITY_DID = _deterministic_did(5)
MALFORMED_ED25519_DIDS = (
    pytest.param("did:key:z0", id="invalid-base58"),
    pytest.param(
        "did:key:z" + _base58btc(b"\xec\x01" + bytes([6]) * 32),
        id="wrong-multicodec",
    ),
    pytest.param(
        "did:key:z" + _base58btc(b"\xed\x01" + bytes([7]) * 31),
        id="wrong-length",
    ),
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_test_git(repo: Path, *arguments: str) -> str:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout.strip()


def _tiny_git_repository(tmp_path: Path) -> Path:
    repo = tmp_path / "sealed-git-fixture"
    repo.mkdir()
    _run_test_git(repo, "init", "--initial-branch=main")
    _run_test_git(repo, "config", "user.name", "EAAEF Test")
    _run_test_git(repo, "config", "user.email", "eaaef@example.invalid")
    (repo / "tracked.txt").write_text("first\n", encoding="ascii")
    _run_test_git(repo, "add", "tracked.txt")
    _run_test_git(repo, "commit", "-m", "first")
    (repo / "tracked.txt").write_text("second\n", encoding="ascii")
    _run_test_git(repo, "add", "tracked.txt")
    _run_test_git(repo, "commit", "-m", "second")
    return repo


def _sealed_forest(*, accelerator_commit: str = "1" * 40) -> dict[str, Any]:
    repositories = [
        {
            "name": "ipfs_accelerate_py",
            "relative_path": ".",
            "commit": accelerator_commit,
            "tree": "a" * 40,
            "gitlink": False,
            "initialized": True,
            "clean": True,
        },
        {
            "name": "ipfs_datasets_py",
            "relative_path": "ipfs_datasets_py",
            "commit": "2" * 40,
            "tree": "b" * 40,
            "gitlink": True,
            "initialized": True,
            "clean": True,
        },
        {
            "name": "ipfs_kit_py",
            "relative_path": "ipfs_kit_py",
            "commit": "3" * 40,
            "tree": "c" * 40,
            "gitlink": True,
            "initialized": True,
            "clean": True,
        },
        {
            "name": "mcpplusplus",
            "relative_path": "ipfs_accelerate_py/mcplusplus",
            "commit": "4" * 40,
            "tree": "d" * 40,
            "gitlink": True,
            "initialized": True,
            "clean": True,
        },
    ]
    board_bytes = (
        Path(__file__).resolve().parents[2] / lifecycle.EAAEF_BOARD_PATH
    ).read_bytes()
    blob_oid = hashlib.sha1(
        b"blob " + str(len(board_bytes)).encode("ascii") + b"\0" + board_bytes,
        usedforsecurity=False,
    ).hexdigest()
    board_source = lifecycle._board_source_binding(
        board_bytes,
        source_head=repositories[0]["commit"],
        source_tree=repositories[0]["tree"],
        git_mode="100644",
        blob_oid=blob_oid,
    )
    identity = {
        "schema": lifecycle.EAAEF_FOREST_SCHEMA,
        "repositories": repositories,
        "board_source": board_source,
    }
    root = lifecycle._cid(identity)
    return {
        **identity,
        "valid": True,
        "blockers": [],
        "source_head": repositories[0]["commit"],
        "source_tree": repositories[0]["tree"],
        "source_forest_root": root,
        "source_generation_cid": root,
        "binding_cid": lifecycle._cid({**identity, "source_forest_root": root}),
    }


@pytest.fixture(autouse=True)
def _explicit_trusted_forest_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        lifecycle,
        "inspect_current_repository_forest",
        lambda _root: _sealed_forest(),
    )
    monkeypatch.setattr(
        lifecycle,
        "_require_production_source_policy",
        lambda _root, *, forest: {
            "source_head": forest["source_head"],
            "source_tree": forest["source_tree"],
            "source_forest_root": forest["source_forest_root"],
            "import_evidence_cid": _TEST_IMPORT_EVIDENCE_CID,
        },
    )


def _board(repo_root: Path) -> dict[str, Any]:
    return json.loads((repo_root / lifecycle.EAAEF_BOARD_PATH).read_text(encoding="utf-8"))


def _population(repo_root: Path) -> lifecycle.CompiledEAAEFPopulation:
    return lifecycle.compile_fresh_eaaef_population(
        _board(repo_root),
        forest=_sealed_forest(),
        repo_root=repo_root,
    )


def _bootstrap_snapshot(
    population: lifecycle.CompiledEAAEFPopulation,
) -> dict[str, Any]:
    value = {
        "schema": lifecycle.EAAEF_BOOTSTRAP_SNAPSHOT_SCHEMA,
        "source_head": population.source_head,
        "source_tree": population.source_tree,
        "source_forest_root": population.source_forest_root,
        "board_cid": population.board_cid,
        "reconciliation_population_cid": population.population_cid,
        "bootstrap_population_cid": population.bootstrap_population_cid,
        "bootstrap_task_count": lifecycle.EAAEF_BOOTSTRAP_TASK_COUNT,
        "held_task_count": lifecycle.EAAEF_PLAN_R2_TASK_COUNT,
        "terminal_statuses_imported": 0,
        "bootstrap_materialization_mode": "offline_before_exclusive_owner_start",
        "bootstrap_owner_absent_during_materialization": True,
        "owner_started_after_bootstrap": True,
        "direct_database_mutation_after_owner_start": False,
        "bootstrap_admission_cid": "sha256:" + "a" * 64,
        "r1_launch_capsule_cid": "sha256:" + "b" * 64,
        "quack_owner_qualification_cid": "sha256:" + "c" * 64,
        "quack_command_fabric_qualification_cid": "sha256:" + "d" * 64,
        "owner_principal_did": OWNER_IDENTITY_DID,
        "shard_id": "fresh-shard",
        "store_id": "fresh-store",
        "owner_generation": 1,
        "expected_epoch": 1,
        "fencing_token": 1,
        "lease_id": "fresh-lease",
        "expected_version": 1,
        "expected_active_plan_cid": population.plan_r1_cid,
        "expected_active_plan_root_cid": population.plan_r1_cid,
        "expected_active_plan_revision": 1,
        "expected_event_cursor": "0",
        "expected_semantic_root_cid": population.source_forest_root,
        "request_id": "fresh-request",
        "idempotency_key": "fresh-idempotency",
        "deadline_ms": 200_000,
        "issued_at_ms": 100_000,
        "expires_at_ms": 300_000,
        "one_use_nonce": "fresh-nonce",
    }
    value["snapshot_cid"] = lifecycle._cid(value)
    return value


def _qualification(source_forest_root: str) -> dict[str, Any]:
    value = {
        "schema": lifecycle.EAAEF_OWNER_QUALIFICATION_SCHEMA,
        "interface": lifecycle.EAAEF_RECONCILIATION_OWNER_INTERFACE,
        "source_forest_root": source_forest_root,
        "bootstrap_materialization_mode": "offline_before_exclusive_owner_start",
        "bootstrap_materialization_before_owner_start": True,
        "offline_population_includes_execution_contracts": True,
        "direct_database_mutation_after_owner_start": False,
        "typed_task_source_interface": lifecycle.EAAEF_TYPED_TASK_SOURCE_INTERFACE,
        "plan_r2_repository_interface": lifecycle.AUTHORIZED_PLAN_R2_REPOSITORY_INTERFACE,
        "plan_r2_remote_gateway_interface": (lifecycle.PLAN_R2_REMOTE_CLIENT_GATEWAY_INTERFACE),
        "plan_r2_wire_channel_interface": lifecycle.PLAN_R2_REMOTE_WIRE_CHANNEL_INTERFACE,
        "plan_r2_remote_runtime_qualification_status": "production_qualified",
        "plan_r2_remote_runtime_blockers": [],
        "status_operation": "status.snapshot",
        "stop_tracks_operation": "stop_tracks",
        "launch_modes": ["paused", "plan_r2"],
        "database_authority_crossing_allowed": False,
        "filesystem_path_authority_crossing_allowed": False,
        "transport_token_authority_crossing_allowed": False,
        "sql_crossing_allowed": False,
        "provider_launch_allowed": True,
    }
    value["qualification_cid"] = lifecycle._cid(value)
    return value


def _bootstrap_qualification(source_forest_root: str) -> dict[str, Any]:
    value = {
        "schema": lifecycle.EAAEF_BOOTSTRAP_OWNER_QUALIFICATION_SCHEMA,
        "interface": lifecycle.EAAEF_BOOTSTRAP_RECONCILIATION_OWNER_INTERFACE,
        "source_forest_root": source_forest_root,
        "materialization_operation": (
            "materialize_offline_22_plus_94_then_start_owner"
        ),
        "bootstrap_materialization_mode": "offline_before_exclusive_owner_start",
        "bootstrap_materialization_before_owner_start": True,
        "offline_population_includes_execution_contracts": True,
        "direct_database_mutation_after_owner_start": False,
        "exclusive_owner_lifecycle_interface": (
            lifecycle.EAAEF_CASF_BOOTSTRAP_OWNER_LIFECYCLE_INTERFACE
        ),
        "exclusive_owner_lifecycle_qualification_status": (
            lifecycle.EAAEF_CASF_PERSISTENT_BOOTSTRAP_QUALIFICATION_STATUS
        ),
        "bootstrap_owner_ready": True,
        "bootstrap_owner_blockers": [],
        "database_authority_crossing_allowed": False,
        "filesystem_path_authority_crossing_allowed": False,
        "transport_token_authority_crossing_allowed": False,
        "sql_crossing_allowed": False,
        "provider_launch_allowed": False,
    }
    value["qualification_cid"] = lifecycle._cid(value)
    return value


class _FakeOwner:
    INTERFACE = lifecycle.EAAEF_RECONCILIATION_OWNER_INTERFACE
    BOOTSTRAP_INTERFACE = lifecycle.EAAEF_BOOTSTRAP_RECONCILIATION_OWNER_INTERFACE

    def __init__(
        self,
        source_forest_root: str,
        *,
        birth: lifecycle.ProcessBirth | None = None,
        omit_stopped_birth: bool = False,
    ) -> None:
        self.source_forest_root = source_forest_root
        self.birth = birth
        self.omit_stopped_birth = omit_stopped_birth
        self.stopped = False
        self.offline_request: dict[str, Any] | None = None

    def reconciliation_qualification(self) -> Mapping[str, Any]:
        return _qualification(self.source_forest_root)

    def bootstrap_reconciliation_qualification(self) -> Mapping[str, Any]:
        return _bootstrap_qualification(self.source_forest_root)

    def materialize_offline_population(
        self,
        request: Mapping[str, Any],
        *,
        population: lifecycle.CompiledEAAEFPopulation,
    ) -> Mapping[str, Any]:
        self.offline_request = dict(request)
        snapshot = _bootstrap_snapshot(population)
        value = {
            "schema": lifecycle.EAAEF_OFFLINE_POPULATION_RECEIPT_SCHEMA,
            "interface": lifecycle.EAAEF_BOOTSTRAP_RECONCILIATION_OWNER_INTERFACE,
            "request_cid": request["request_cid"],
            "generation_id": request["generation_id"],
            "source_forest_root": population.source_forest_root,
            "population_cid": population.population_cid,
            "goal_population_cid": population.goal_population_cid,
            "execution_contract_population_cid": (
                population.execution_contract_population_cid
            ),
            "bootstrap_population_cid": population.bootstrap_population_cid,
            "held_plan_r2_population_cid": population.plan_r2_population_cid,
            "plan_r1_cid": population.plan_r1_cid,
            "task_count": lifecycle.EAAEF_TASK_COUNT,
            "goal_count": lifecycle.EAAEF_GOAL_COUNT,
            "goal_edge_count": lifecycle.EAAEF_GOAL_EDGE_COUNT,
            "plan_count": 1,
            "bootstrap_task_count": lifecycle.EAAEF_BOOTSTRAP_TASK_COUNT,
            "held_task_count": lifecycle.EAAEF_PLAN_R2_TASK_COUNT,
            "task_status_counts": {
                "blocked": lifecycle.EAAEF_PLAN_R2_TASK_COUNT,
                "todo": lifecycle.EAAEF_BOOTSTRAP_TASK_COUNT,
            },
            "execution_contract_counts": population.execution_contract_counts,
            "execution_contracts_materialized": True,
            "terminal_statuses_imported": 0,
            "bootstrap_materialization_mode": "offline_before_exclusive_owner_start",
            "bootstrap_owner_absent_during_materialization": True,
            "owner_started_after_bootstrap": True,
            "direct_database_mutation_after_owner_start": False,
            "provider_process_started": False,
            "bootstrap_snapshot": snapshot,
        }
        value["receipt_cid"] = lifecycle._cid(value)
        return value

    def apply_signed_plan_r2(
        self,
        request: Mapping[str, Any],
        *,
        population: lifecycle.CompiledEAAEFPopulation,
        authority: lifecycle.VerifiedFreshEAAEFAuthority,
    ) -> Mapping[str, Any]:
        raise AssertionError("signed authority is not created or applied by these tests")

    def launch_reconciliation_supervisor(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        raise AssertionError("no supervisor is launched by these tests")

    def reconciliation_status_snapshot(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        active = not self.stopped
        value = {
            "schema": lifecycle.EAAEF_OWNER_STATUS_RECEIPT_SCHEMA,
            "interface": lifecycle.EAAEF_RECONCILIATION_OWNER_INTERFACE,
            "request_cid": request["request_cid"],
            "active": active,
            "generation_id": "eaaef-test-generation" if active else "",
            "phase": "launched_paused" if active else "absent",
            "source_head": "1" * 40 if active else "",
            "source_forest_root": self.source_forest_root if active else "",
            "task_count": lifecycle.EAAEF_TASK_COUNT if active else 0,
            "task_status_counts": {"todo": lifecycle.EAAEF_TASK_COUNT} if active else {},
            "supervisor_birth": self.birth.to_dict() if active and self.birth else None,
            "provider_process_started": False,
        }
        value["receipt_cid"] = lifecycle._cid(value)
        return value

    def stop_reconciliation_tracks(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        stopped_births = []
        if self.birth is not None and not self.omit_stopped_birth:
            stopped_births.append(self.birth.to_dict())
        self.stopped = True
        value = {
            "schema": lifecycle.EAAEF_OWNER_STOP_RECEIPT_SCHEMA,
            "interface": lifecycle.EAAEF_RECONCILIATION_OWNER_INTERFACE,
            "request_cid": request["request_cid"],
            "generation_id": request["generation_id"],
            "stopped": True,
            "remaining_track_count": 0,
            "stopped_process_births": stopped_births,
            "provider_processes_stopped": True,
            "task_state_mutated": False,
        }
        value["receipt_cid"] = lifecycle._cid(value)
        return value


class _FakeBootstrapOwner:
    INTERFACE = lifecycle.EAAEF_BOOTSTRAP_RECONCILIATION_OWNER_INTERFACE
    BOOTSTRAP_INTERFACE = lifecycle.EAAEF_BOOTSTRAP_RECONCILIATION_OWNER_INTERFACE

    def __init__(self, source_forest_root: str) -> None:
        self._delegate = _FakeOwner(source_forest_root)
        self.source_forest_root = source_forest_root

    @property
    def offline_request(self) -> dict[str, Any] | None:
        return self._delegate.offline_request

    def bootstrap_reconciliation_qualification(self) -> Mapping[str, Any]:
        return _bootstrap_qualification(self.source_forest_root)

    def materialize_offline_population(
        self,
        request: Mapping[str, Any],
        *,
        population: lifecycle.CompiledEAAEFPopulation,
    ) -> Mapping[str, Any]:
        return self._delegate.materialize_offline_population(
            request,
            population=population,
        )


def _state(
    population: lifecycle.CompiledEAAEFPopulation,
    *,
    phase: str = "launched_paused",
) -> dict[str, Any]:
    value = {
        "schema": lifecycle.EAAEF_STATE_SCHEMA,
        "interface": lifecycle.EAAEF_RECONCILIATION_LIFECYCLE_INTERFACE,
        "generation_id": "eaaef-test-generation",
        "phase": phase,
        "source_head": population.source_head,
        "source_tree": population.source_tree,
        "source_forest_root": population.source_forest_root,
        "casf_import_evidence_cid": _TEST_IMPORT_EVIDENCE_CID,
        "population": population.public_dict(),
        "supervisor_birth": None,
        "provider_process_started": False,
        "updated_at_ms": 1,
    }
    value["state_cid"] = lifecycle._cid(value)
    return value


def _parser_destinations(parser: argparse.ArgumentParser) -> set[str]:
    result = {action.dest for action in parser._actions}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                result.update(_parser_destinations(child))
    return result


def test_fresh_population_is_exact_22_plus_94_and_plan_r2_releases_all(
    repo_root: Path,
) -> None:
    population = _population(repo_root)

    assert population.task_count == 116
    assert len(population.bootstrap_tasks) == 22
    assert len(population.plan_r2_tasks) == 94
    assert len(population.dependencies) == 270
    assert Counter(item["status"] for item in population.bootstrap_tasks) == {"todo": 22}
    assert Counter(item["status"] for item in population.plan_r2_tasks) == {"blocked": 94}
    assert population.execution_contract_counts == {
        "task_dependencies": 270,
        "task_outputs": 430,
        "task_validations": 117,
        "task_acceptance": 116,
    }

    statement = lifecycle.build_unsigned_fresh_plan_r2_statement(
        population=population,
        bootstrap_snapshot=_bootstrap_snapshot(population),
    )
    transition = population.plan_r2_transition_tasks(
        plan_cid=str(statement["new_plan"]["plan_cid"])
    )

    assert len(statement["tasks"]) == 116
    assert len(statement["dependencies"]) == 270
    assert statement["protected_tasks"] == []
    assert all(item["status"] == "todo" for item in transition)
    assert all(item["body"]["is_schedulable"] is True for item in transition)
    assert all(item["body"]["blocked_reason"] == "" for item in transition)
    assert all(item["revision"] == 2 for item in transition)
    assert len(lifecycle._canonical_bytes(statement)) <= (
        lifecycle.MAX_PLAN_R2_REMOTE_REQUEST_BYTES
        - lifecycle._PLAN_R2_REMOTE_REQUEST_OVERHEAD_RESERVE
    )
    assert "operator_signature" not in statement
    assert "security_reviewer_signature" not in statement


def test_signing_request_projection_is_deterministic_current_and_has_no_effects(
    repo_root: Path,
) -> None:
    population = _population(repo_root)
    arguments = {
        "population": population,
        "bootstrap_snapshot": _bootstrap_snapshot(population),
        "operator_identity_did": OPERATOR_IDENTITY_DID,
        "security_reviewer_identity_did": SECURITY_REVIEWER_IDENTITY_DID,
        "capability_reviewer_identity_did": CAPABILITY_REVIEWER_IDENTITY_DID,
        "issued_at_ms": 100_500,
        "expires_at_ms": 250_000,
    }
    request = lifecycle.build_fresh_plan_r2_signing_request_projection(**arguments)

    assert request == lifecycle.build_fresh_plan_r2_signing_request_projection(
        **arguments
    )
    assert request["schema"] == lifecycle.EAAEF_PLAN_R2_SIGNING_REQUEST_SCHEMA
    assert request["source_head"] == population.source_head
    assert request["source_tree"] == population.source_tree
    assert request["source_forest_root"] == population.source_forest_root
    assert request["statement_cid"] == request["unsigned_plan_r2_statement"][
        "statement_cid"
    ]
    assert request["request_cid"] == lifecycle._cid(
        {key: value for key, value in request.items() if key != "request_cid"}
    )
    assert set(request["signing_payloads"]) == {
        "independent_operator",
        "independent_security_reviewer",
        "independent_plan_r2_capability_reviewer",
    }
    assert all(
        "signature" not in payload and "reviewer_signature" not in payload
        for payload in request["signing_payloads"].values()
    )
    assert request["deferred_external_signature"].startswith(
        "independent_plan_r2_remote_transport_reviewer"
    )
    for field in (
        "authority_valid",
        "launch_allowed",
        "trust_roots_read",
        "signing_key_read",
        "signature_created",
        "authority_mutated",
        "provider_process_started",
    ):
        assert request[field] is False

    stale_snapshot = json.loads(json.dumps(arguments["bootstrap_snapshot"]))
    stale_snapshot["source_head"] = "9" * 40
    stale_snapshot.pop("snapshot_cid")
    stale_snapshot["snapshot_cid"] = lifecycle._cid(stale_snapshot)
    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="bootstrap owner snapshot differs",
    ):
        lifecycle.build_fresh_plan_r2_signing_request_projection(
            **{
                **arguments,
                "bootstrap_snapshot": stale_snapshot,
            }
        )
    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="identities are not independent",
    ):
        lifecycle.build_fresh_plan_r2_signing_request_projection(
            **{
                **arguments,
                "security_reviewer_identity_did": arguments[
                    "operator_identity_did"
                ],
            }
        )


def test_signing_request_cli_has_only_the_documented_read_only_qualification_effects(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    population = _population(repo_root)
    snapshot_path = tmp_path / "bootstrap-snapshot.json"
    snapshot_path.write_text(
        json.dumps(_bootstrap_snapshot(population), sort_keys=True),
        encoding="ascii",
    )

    allowed_effects: list[str] = []

    def captured_git_qualification(_repo_root: Path) -> dict[str, Any]:
        allowed_effects.append("captured_read_only_git_children")
        return _sealed_forest()

    def exact_structural_qualification(
        _repo_root: Path,
        *,
        forest: Mapping[str, Any],
    ) -> dict[str, Any]:
        allowed_effects.extend(
            (
                "exact_isolated_structural_validator_child",
                "cleaned_transient_validator_storage",
            )
        )
        return {
            "source_head": forest["source_head"],
            "source_tree": forest["source_tree"],
            "source_forest_root": forest["source_forest_root"],
            "import_evidence_cid": _TEST_IMPORT_EVIDENCE_CID,
        }

    def forbidden_effect(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("signing-request crossed its forbidden effect boundary")

    monkeypatch.setattr(
        lifecycle,
        "inspect_current_repository_forest",
        captured_git_qualification,
    )
    monkeypatch.setattr(
        lifecycle,
        "_require_production_source_policy",
        exact_structural_qualification,
    )
    for name in (
        "resolve_production_reconciliation_owner",
        "_authority_from_args",
        "load_fresh_authority_artifacts",
        "load_fresh_trust_roots",
        "preflight_reconciliation",
        "prepare_fresh_generation",
        "materialize_fresh_generation",
        "launch_reconciliation_supervisor",
        "reconciliation_status",
        "stop_reconciliation_generation",
        "ReconciliationStateStore",
    ):
        monkeypatch.setattr(lifecycle, name, forbidden_effect)
    before = set(tmp_path.iterdir())
    result = lifecycle.main(
        [
            "--repo-root",
            str(repo_root),
            "--state-root",
            str(tmp_path / "unused-state"),
            "signing-request",
            "--bootstrap-snapshot",
            str(snapshot_path),
            "--operator-identity-did",
            OPERATOR_IDENTITY_DID,
            "--security-reviewer-identity-did",
            SECURITY_REVIEWER_IDENTITY_DID,
            "--plan-r2-capability-reviewer-identity-did",
            CAPABILITY_REVIEWER_IDENTITY_DID,
            "--issued-at-ms",
            "100500",
            "--expires-at-ms",
            "250000",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["schema"] == lifecycle.EAAEF_PLAN_R2_SIGNING_REQUEST_SCHEMA
    assert output["signing_key_read"] is False
    assert output["signature_created"] is False
    assert output["provider_process_started"] is False
    assert allowed_effects == [
        "captured_read_only_git_children",
        "exact_isolated_structural_validator_child",
        "cleaned_transient_validator_storage",
        "captured_read_only_git_children",
    ]
    assert set(tmp_path.iterdir()) == before
    assert not (tmp_path / "unused-state").exists()

    signing_source = inspect.getsource(
        lifecycle.build_fresh_plan_r2_signing_request_projection
    )
    signing_source += inspect.getsource(lifecycle.main)
    assert "Ed25519PrivateKey" not in signing_source
    assert ".sign(" not in signing_source
    help_text = lifecycle._argument_parser().format_help()
    subparsers = next(
        action
        for action in lifecycle._argument_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    signing_help = " ".join(
        subparsers.choices["signing-request"].format_help().split()
    )
    assert "captured Git children" in signing_help
    assert "isolated structural-validator child" in signing_help
    assert "cleaned transient storage" in signing_help
    assert "durable artifacts/state" in signing_help
    assert "providers/supervisors" in signing_help
    assert "signing-request" in help_text


@pytest.mark.parametrize(
    "identity_field",
    (
        "operator_identity_did",
        "security_reviewer_identity_did",
        "capability_reviewer_identity_did",
        "owner_principal_did",
    ),
)
@pytest.mark.parametrize("malformed_did", MALFORMED_ED25519_DIDS)
def test_signing_request_rejects_malformed_dids_before_any_payload_is_emitted(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    identity_field: str,
    malformed_did: str,
) -> None:
    population = _population(repo_root)
    snapshot = _bootstrap_snapshot(population)
    if identity_field == "owner_principal_did":
        snapshot[identity_field] = malformed_did
        snapshot.pop("snapshot_cid")
        snapshot["snapshot_cid"] = lifecycle._cid(snapshot)
    snapshot_path = tmp_path / "bootstrap-snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, sort_keys=True), encoding="ascii")

    def forbidden_payload(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("malformed DID reached a signing payload builder")

    monkeypatch.setattr(
        lifecycle,
        "prepare_plan_r2_transition_approval",
        forbidden_payload,
    )
    monkeypatch.setattr(
        lifecycle,
        "plan_r2_operational_capability_signing_payload",
        forbidden_payload,
    )
    identities = {
        "operator_identity_did": OPERATOR_IDENTITY_DID,
        "security_reviewer_identity_did": SECURITY_REVIEWER_IDENTITY_DID,
        "capability_reviewer_identity_did": CAPABILITY_REVIEWER_IDENTITY_DID,
    }
    if identity_field in identities:
        identities[identity_field] = malformed_did

    result = lifecycle.main(
        [
            "--repo-root",
            str(repo_root),
            "--state-root",
            str(tmp_path / "unused-state"),
            "signing-request",
            "--bootstrap-snapshot",
            str(snapshot_path),
            "--operator-identity-did",
            identities["operator_identity_did"],
            "--security-reviewer-identity-did",
            identities["security_reviewer_identity_did"],
            "--plan-r2-capability-reviewer-identity-did",
            identities["capability_reviewer_identity_did"],
            "--issued-at-ms",
            "100500",
            "--expires-at-ms",
            "250000",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert result == 2
    assert output["error_code"] == "EAAEFReconciliationIdentityError"
    assert output["command"] == "signing-request"
    assert "signing_payloads" not in output
    assert "unsigned_plan_r2_statement" not in output
    assert "request_cid" not in output
    assert output["authority_mutated"] is False
    assert output["provider_process_started"] is False
    assert not (tmp_path / "unused-state").exists()


def test_stale_forest_and_bootstrap_bindings_fail_closed(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _sealed_forest()
    stale = json.loads(json.dumps(original))
    stale["repositories"][0]["tree"] = "e" * 40
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="forest"):
        lifecycle.compile_fresh_eaaef_population(
            _board(repo_root),
            forest=stale,
            repo_root=repo_root,
        )

    original_population = lifecycle.compile_fresh_eaaef_population(
        _board(repo_root),
        forest=original,
        repo_root=repo_root,
    )
    fresh_forest = _sealed_forest(accelerator_commit="9" * 40)
    with monkeypatch.context() as fresh_inspection:
        fresh_inspection.setattr(
            lifecycle,
            "inspect_current_repository_forest",
            lambda _root: fresh_forest,
        )
        fresh_population = lifecycle.compile_fresh_eaaef_population(
            _board(repo_root),
            forest=fresh_forest,
            repo_root=repo_root,
        )
    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="bootstrap owner snapshot differs",
    ):
        lifecycle.build_unsigned_fresh_plan_r2_statement(
            population=fresh_population,
            bootstrap_snapshot=_bootstrap_snapshot(original_population),
        )


def test_board_source_binding_is_read_from_exact_git_tree_blob(repo_root: Path) -> None:
    head = lifecycle._git(repo_root, "rev-parse", "HEAD")
    tree = lifecycle._git(repo_root, "rev-parse", f"{head}^{{tree}}")
    binding = lifecycle._git_board_source(
        repo_root,
        source_head=head,
        source_tree=tree,
    )
    board_bytes = (repo_root / lifecycle.EAAEF_BOARD_PATH).read_bytes()

    assert binding["relative_path"] == lifecycle.EAAEF_BOARD_PATH
    assert binding["source_head"] == head
    assert binding["source_tree"] == tree
    assert binding["git_mode"] == "100644"
    assert binding["object_type"] == "blob"
    assert binding["byte_count"] == len(board_bytes)
    assert binding["bytes_cid"] == lifecycle._cid(board_bytes)
    assert binding["canonical_json_cid"] == lifecycle._eaaef_source_cid(_board(repo_root))


def test_git_identity_reads_disable_replacements_and_ambient_redirection(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = {
        "GIT_COMMON_DIR": "/forged/common-dir",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.worktree",
        "GIT_CONFIG_VALUE_0": "/forged/config-work-tree",
        "GIT_DIR": "/forged/git-dir",
        "GIT_INDEX_FILE": "/forged/index",
        "GIT_WORK_TREE": "/forged/work-tree",
        "GIT_OBJECT_DIRECTORY": "/forged/objects",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": "/forged/alternates",
        "GIT_REPLACE_REF_BASE": "refs/forged/replace/",
    }
    for name, value in hostile.items():
        monkeypatch.setenv(name, value)
    real_run = lifecycle.subprocess.run
    observed: list[tuple[list[str], dict[str, str]]] = []

    def audited_run(arguments: list[str], **kwargs: Any) -> Any:
        if arguments and arguments[0] == "git":
            environment = dict(kwargs.get("env") or {})
            observed.append((list(arguments), environment))
            assert arguments[1] == "--no-replace-objects"
            assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
            assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
            assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
            assert environment["GIT_CONFIG_SYSTEM"] == os.devnull
            assert set(environment).isdisjoint(hostile)
            flattened = "\0".join(arguments)
            for sealed_config in (
                "core.worktree=",
                "extensions.worktreeConfig=false",
                "core.fsmonitor=false",
                "core.untrackedCache=false",
                "core.sparseCheckout=false",
                "core.ignoreStat=false",
                "core.commitGraph=false",
            ):
                assert sealed_config in flattened
        return real_run(arguments, **kwargs)

    monkeypatch.setattr(lifecycle.subprocess, "run", audited_run)
    head = lifecycle._git(repo_root, "rev-parse", "HEAD")
    blob_oid = lifecycle._git(
        repo_root,
        "rev-parse",
        f"{head}:{lifecycle.EAAEF_CASF_IMPORT_MANIFEST_PATH}",
    )
    assert lifecycle._git_blob(repo_root, blob_oid, maximum_bytes=128 * 1024)
    assert len(observed) == 4


def test_sealed_git_command_overrides_repository_config_redirection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _tiny_git_repository(tmp_path)
    forged_worktree = tmp_path / "forged-worktree"
    forged_worktree.mkdir()
    sentinel = tmp_path / "fsmonitor-was-executed"
    monitor = tmp_path / "hostile-fsmonitor.sh"
    monitor.write_text(
        "#!/bin/sh\n" f": > {sentinel}\n" "exit 0\n",
        encoding="ascii",
    )
    monitor.chmod(0o700)
    git_dir = repo / ".git"
    for key, value in (
        ("core.worktree", str(forged_worktree)),
        ("core.fsmonitor", str(monitor)),
        ("core.untrackedCache", "true"),
        ("core.sparseCheckout", "true"),
        ("core.ignoreStat", "true"),
    ):
        result = subprocess.run(
            ["git", f"--git-dir={git_dir}", "config", key, value],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
    forged_index = tmp_path / "forged-index"
    forged_index.write_bytes(b"not an index")
    monkeypatch.setenv("GIT_INDEX_FILE", str(forged_index))
    monkeypatch.setenv("GIT_WORK_TREE", str(forged_worktree))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.worktree")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(forged_worktree))

    assert Path(lifecycle._git(repo, "rev-parse", "--show-toplevel")) == repo
    assert lifecycle._git(repo, "status", "--porcelain=v1", "--untracked-files=all") == ""
    lifecycle._require_sealed_git_repository(repo)
    assert not sentinel.exists()


@pytest.mark.parametrize(
    ("flag", "message"),
    (
        ("--assume-unchanged", "assume-unchanged"),
        ("--skip-worktree", "skip-worktree"),
    ),
)
def test_sealed_git_repository_rejects_hidden_index_flags(
    tmp_path: Path,
    flag: str,
    message: str,
) -> None:
    repo = _tiny_git_repository(tmp_path)
    lifecycle._require_sealed_git_repository(repo)
    _run_test_git(repo, "update-index", flag, "tracked.txt")

    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match=message):
        lifecycle._require_sealed_git_repository(repo)


def test_sealed_git_repository_rejects_fsmonitor_index_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _tiny_git_repository(tmp_path)
    real_git = lifecycle._git

    def tagged_git(repo_root: Path, *arguments: str, check: bool = True) -> str:
        if arguments == ("ls-files", "-f", "--cached"):
            return "h tracked.txt"
        return real_git(repo_root, *arguments, check=check)

    monkeypatch.setattr(lifecycle, "_git", tagged_git)
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="fsmonitor"):
        lifecycle._require_sealed_git_repository(repo)


def test_checkout_blob_rejects_assume_unchanged_worktree_drift(tmp_path: Path) -> None:
    repo = _tiny_git_repository(tmp_path)
    blob_oid = _run_test_git(repo, "rev-parse", "HEAD:tracked.txt")
    _run_test_git(repo, "update-index", "--assume-unchanged", "tracked.txt")
    (repo / "tracked.txt").write_text("hidden drift\n", encoding="ascii")
    assert _run_test_git(repo, "status", "--porcelain=v1", "--untracked-files=all") == ""

    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="differs from sealed blob",
    ):
        lifecycle._require_checkout_blob(repo, "tracked.txt", blob_oid)


@pytest.mark.parametrize(
    "indirection",
    ("grafts", "grafts_symlink", "alternates", "http_alternates", "replace"),
)
def test_sealed_git_repository_rejects_ancestry_and_object_indirection(
    tmp_path: Path,
    indirection: str,
) -> None:
    repo = _tiny_git_repository(tmp_path)
    lifecycle._require_sealed_git_repository(repo)
    git_dir = repo / ".git"
    if indirection == "grafts":
        (git_dir / "info/grafts").write_text(
            _run_test_git(repo, "rev-parse", "HEAD") + "\n",
            encoding="ascii",
        )
    elif indirection == "grafts_symlink":
        target = tmp_path / "external-grafts"
        target.write_text(_run_test_git(repo, "rev-parse", "HEAD") + "\n", encoding="ascii")
        (git_dir / "info/grafts").symlink_to(target)
    elif indirection in {"alternates", "http_alternates"}:
        alternate_objects = tmp_path / "alternate-objects"
        alternate_objects.mkdir()
        name = "alternates" if indirection == "alternates" else "http-alternates"
        (git_dir / f"objects/info/{name}").write_text(
            str(alternate_objects) + "\n",
            encoding="ascii",
        )
    else:
        _run_test_git(
            repo,
            "replace",
            _run_test_git(repo, "rev-parse", "HEAD"),
            _run_test_git(repo, "rev-parse", "HEAD~1"),
        )

    expected = "replacement refs" if indirection == "replace" else "graft or alternate"
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match=expected):
        lifecycle._require_sealed_git_repository(repo)


def test_canonical_casf_import_is_exact_and_structurally_valid(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as trusted:
        trusted.setattr(
            lifecycle,
            "inspect_current_repository_forest",
            _REAL_INSPECT_CURRENT_REPOSITORY_FOREST,
        )
        forest = _REAL_INSPECT_CURRENT_REPOSITORY_FOREST(repo_root)
        evidence = _REAL_VERIFY_IMPORTED_CASF_SOURCE(repo_root, forest=forest)

    assert evidence["canonical_tip"] == "4030ab14cbe084ee4716a904fc37677aafd168d3"
    assert evidence["canonical_tree"] == "7aa6f3c7c2bd90a046bb1514d7b3a92c7af55714"
    assert evidence["structural_validation_report_cid"] == (
        "sha256:ed81aec7bad2b030325fe998d187f32e757c3514d35eef01a7d1a91ba4d98c67"
    )
    assert evidence["standalone_operator_policy_unchanged"] is True
    assert set(evidence["selected_blobs"]) == lifecycle._EAAEF_CASF_IMPORT_BLOB_PATHS
    assert len(evidence["selected_blobs"]) == 17
    assert lifecycle._EAAEF_CASF_IMPORT_OVERRIDE_PATHS == {
        "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    }


def test_casf_import_rejects_config_binding_and_current_blob_drift(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_git_tree_json_object = lifecycle._git_tree_json_object
    real_git_tree_blob_oid = lifecycle._git_tree_blob_oid
    real_git_blob = lifecycle._git_blob
    with monkeypatch.context() as trusted:
        trusted.setattr(
            lifecycle,
            "inspect_current_repository_forest",
            _REAL_INSPECT_CURRENT_REPOSITORY_FOREST,
        )
        forest = _REAL_INSPECT_CURRENT_REPOSITORY_FOREST(repo_root)
        source_tree = str(forest["source_tree"])
        canonical_manifest, manifest_oid = real_git_tree_json_object(
            repo_root,
            source_tree,
            lifecycle.EAAEF_CASF_IMPORT_MANIFEST_PATH,
            noun="test CASF import manifest",
        )
        canonical_config, config_oid = real_git_tree_json_object(
            repo_root,
            source_tree,
            lifecycle.EAAEF_CONFIG_PATH,
            noun="test EAAEF config",
        )
        protected_override_manifest = json.loads(json.dumps(canonical_manifest))
        protected_override_manifest["eaaef_import_overrides"][
            "scripts/validate_agent_supervisor_causal_event_federation_board.py"
        ] = protected_override_manifest["selected_blobs"][
            "scripts/validate_agent_supervisor_causal_event_federation_board.py"
        ]
        protected_override_manifest.pop("manifest_cid")
        protected_override_manifest["manifest_cid"] = lifecycle._cid(
            protected_override_manifest
        )
        protected_override_config = json.loads(json.dumps(canonical_config))
        protected_override_config["casf_import_binding"]["manifest_cid"] = (
            protected_override_manifest["manifest_cid"]
        )

        def protected_override(
            root: Path,
            tree: str,
            path: str,
            *,
            noun: str,
            maximum_bytes: int = 32 * 1024 * 1024,
        ) -> tuple[dict[str, Any], str]:
            del root, tree, noun, maximum_bytes
            if path == lifecycle.EAAEF_CONFIG_PATH:
                return json.loads(json.dumps(protected_override_config)), config_oid
            if path == lifecycle.EAAEF_CASF_IMPORT_MANIFEST_PATH:
                return json.loads(json.dumps(protected_override_manifest)), manifest_oid
            raise AssertionError(f"unexpected sealed JSON path: {path}")

        trusted.setattr(lifecycle, "_git_tree_json_object", protected_override)
        with pytest.raises(
            lifecycle.EAAEFReconciliationIdentityError,
            match="override inventory differs",
        ):
            _REAL_VERIFY_IMPORTED_CASF_SOURCE(repo_root, forest=forest)

        trusted.setattr(lifecycle, "_git_tree_json_object", real_git_tree_json_object)

        def mismatched_config(
            root: Path,
            tree: str,
            path: str,
            *,
            noun: str,
            maximum_bytes: int = 32 * 1024 * 1024,
        ) -> tuple[dict[str, Any], str]:
            value, oid = real_git_tree_json_object(
                root,
                tree,
                path,
                noun=noun,
                maximum_bytes=maximum_bytes,
            )
            if path == lifecycle.EAAEF_CONFIG_PATH:
                value["casf_import_binding"]["manifest_cid"] = "sha256:" + "0" * 64
            return value, oid

        trusted.setattr(lifecycle, "_git_tree_json_object", mismatched_config)
        with pytest.raises(
            lifecycle.EAAEFReconciliationIdentityError,
            match="scheduler canonical CASF import binding differs",
        ):
            _REAL_VERIFY_IMPORTED_CASF_SOURCE(repo_root, forest=forest)

        trusted.setattr(lifecycle, "_git_tree_json_object", real_git_tree_json_object)

        def divergent_sealed_validator(
            root: Path,
            blob_oid: str,
            *,
            maximum_bytes: int,
        ) -> bytes:
            if blob_oid == "ae25600a5a3cc866b8b94d262b60df5f62f60a78":
                return b"raise RuntimeError('divergent sealed validator sentinel')\n"
            return real_git_blob(root, blob_oid, maximum_bytes=maximum_bytes)

        trusted.setattr(lifecycle, "_git_blob", divergent_sealed_validator)
        with pytest.raises(
            lifecycle.EAAEFReconciliationIdentityError,
            match="divergent sealed validator sentinel",
        ):
            _REAL_VERIFY_IMPORTED_CASF_SOURCE(repo_root, forest=forest)

        trusted.setattr(lifecycle, "_git_blob", real_git_blob)

        def mismatched_current_blob(root: Path, tree: str, path: str) -> str:
            if path == "ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py":
                return "0" * 40
            return real_git_tree_blob_oid(root, tree, path)

        trusted.setattr(lifecycle, "_git_tree_blob_oid", mismatched_current_blob)
        with pytest.raises(
            lifecycle.EAAEFReconciliationIdentityError,
            match="canonical CASF import blob differs",
        ):
            _REAL_VERIFY_IMPORTED_CASF_SOURCE(repo_root, forest=forest)


def _current_head_forest(repo_root: Path) -> dict[str, Any]:
    return _sealed_forest(accelerator_commit=lifecycle._git(repo_root, "rev-parse", "HEAD"))


def _fresh_trust_roots() -> dict[str, Any]:
    value = {
        "schema": lifecycle.EAAEF_FRESH_TRUST_SCHEMA,
        "remote_reviewer_dids": [REMOTE_REVIEWER_IDENTITY_DID],
        "plan_r2_capability_reviewer_dids": [CAPABILITY_REVIEWER_IDENTITY_DID],
        "operator_dids": [OPERATOR_IDENTITY_DID],
        "security_reviewer_dids": [SECURITY_REVIEWER_IDENTITY_DID],
    }
    value["trust_bundle_cid"] = lifecycle._cid(value)
    return value


def test_preflight_accepts_current_head_over_tracked_predecessor_policy(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forest = _current_head_forest(repo_root)
    observed: list[str] = []
    monkeypatch.setattr(
        lifecycle,
        "inspect_current_repository_forest",
        lambda _root: forest,
    )

    def verify_current_authority(
        _authority: Mapping[str, Any],
        *,
        population: lifecycle.CompiledEAAEFPopulation,
        trust_roots: Mapping[str, Any],
        now_ms: int | None = None,
    ) -> object:
        del trust_roots, now_ms
        observed.append(population.source_forest_root)
        return object()

    monkeypatch.setattr(
        lifecycle,
        "verify_fresh_authority_bundle",
        verify_current_authority,
    )
    result = lifecycle.preflight_reconciliation(
        repo_root,
        authority={"fresh": True},
        trust_roots={"independent": True},
        owner=_FakeOwner(str(forest["source_forest_root"])),
    )

    assert result["valid"] is True
    assert result["bootstrap_owner_ready"] is True
    assert result["production_owner_ready"] is True
    assert result["stale_bindings"] == []
    assert observed == [forest["source_forest_root"]]
    assert result["population"]["source_head"] == forest["source_head"]


def test_changed_forest_rejects_stale_external_authority(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_forest = _sealed_forest()
    current_forest = _current_head_forest(repo_root)
    assert prior_forest["source_forest_root"] != current_forest["source_forest_root"]
    monkeypatch.setattr(
        lifecycle,
        "inspect_current_repository_forest",
        lambda _root: current_forest,
    )
    stale_authorization = {
        "board_namespace": lifecycle.EAAEF_BOARD_NAMESPACE,
        "source_head": prior_forest["source_head"],
        "source_tree": prior_forest["source_tree"],
        "source_generation_cid": prior_forest["source_forest_root"],
        "new_plan": {},
    }
    authority = lifecycle.assemble_fresh_authority_bundle(
        authorization=stale_authorization,
        plan_r2_operational_capability={},
        plan_r2_remote_owner_capability={},
    )

    result = lifecycle.preflight_reconciliation(
        repo_root,
        authority=authority,
        trust_roots=_fresh_trust_roots(),
        owner=_FakeOwner(str(current_forest["source_forest_root"])),
        now_ms=1,
    )

    assert result["valid"] is False
    assert result["stale_bindings"] == []
    assert any(
        blocker.startswith(
            "fresh_authority_rejected:EAAEFReconciliationIdentityError:"
            "Plan-R2 authorization is stale or belongs to another source"
        )
        for blocker in result["blockers"]
    )


def test_historical_tracked_host_admission_is_ignored(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forest = _current_head_forest(repo_root)
    original_json_object = lifecycle._json_object
    historical_path = (repo_root / lifecycle.EAAEF_ADMISSION_BUNDLE_PATH).resolve()
    monkeypatch.setattr(
        lifecycle,
        "inspect_current_repository_forest",
        lambda _root: forest,
    )

    def reject_historical_read(
        path: Path,
        *,
        noun: str,
        maximum_bytes: int = 32 * 1024 * 1024,
    ) -> dict[str, Any]:
        assert Path(path).resolve() != historical_path
        return original_json_object(path, noun=noun, maximum_bytes=maximum_bytes)

    monkeypatch.setattr(lifecycle, "_json_object", reject_historical_read)
    result = lifecycle.preflight_reconciliation(repo_root)

    assert "historical_host_admission" not in result["stale_bindings"]
    assert all("historical_host_admission" not in item for item in result["blockers"])
    assert "scheduler_source_policy" not in result["stale_bindings"]


def test_bootstrap_preflight_readiness_is_distinct_from_production_readiness(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = _population(repo_root)
    bootstrap_owner = _FakeBootstrapOwner(population.source_forest_root)
    monkeypatch.setattr(
        lifecycle,
        "verify_fresh_authority_bundle",
        lambda *_args, **_kwargs: object(),
    )

    result = lifecycle.preflight_reconciliation(
        repo_root,
        authority={"fresh": True},
        trust_roots={"independent": True},
        bootstrap_owner=bootstrap_owner,
    )

    assert result["bootstrap_owner_ready"] is True
    assert result["production_owner_ready"] is False
    assert result["valid"] is False
    assert result["bootstrap_owner_qualification"]["bootstrap_owner_ready"] is True
    assert (
        "bootstrap_portfolio_materialization_owner_unavailable_until_casf_binding"
        not in result["blockers"]
    )
    assert (
        "typed_portfolio_materialization_owner_unavailable_until_final_casf_adapter"
        in result["blockers"]
    )


def test_bootstrap_resolver_requires_an_explicit_exact_binding(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        typed_eaaef_reconciliation_owner as owner_module,
    )

    monkeypatch.delattr(
        owner_module,
        "open_eaaef_bootstrap_reconciliation_owner",
        raising=False,
    )
    with pytest.raises(lifecycle.EAAEFReconciliationBlocked, match="opener is absent"):
        lifecycle.resolve_bootstrap_reconciliation_owner(repo_root)

    population = _population(repo_root)
    bound = _FakeBootstrapOwner(population.source_forest_root)
    monkeypatch.setattr(
        owner_module,
        "open_eaaef_bootstrap_reconciliation_owner",
        lambda *, repo_root: bound,
        raising=False,
    )
    assert lifecycle.resolve_bootstrap_reconciliation_owner(repo_root) is bound


def test_one_shot_materialize_cli_does_not_resolve_or_orphan_bootstrap_owner(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def production_unavailable(_repo_root: Path) -> object:
        raise lifecycle.EAAEFReconciliationBlocked("full production owner absent")

    def forbidden_bootstrap_resolution(_repo_root: Path) -> object:
        raise AssertionError("one-shot materialize resolved a bootstrap broker")

    monkeypatch.setattr(
        lifecycle,
        "resolve_production_reconciliation_owner",
        production_unavailable,
    )
    monkeypatch.setattr(
        lifecycle,
        "resolve_bootstrap_reconciliation_owner",
        forbidden_bootstrap_resolution,
    )
    state_root = tmp_path / "unused-state"

    exit_code = lifecycle.main(
        [
            "--repo-root",
            str(repo_root),
            "--state-root",
            str(state_root),
            "materialize",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert output["error"] == "full production owner absent"
    assert not state_root.exists()


def test_only_prepare_uses_the_bootstrap_gate() -> None:
    prepare_source = inspect.getsource(lifecycle.prepare_fresh_generation)
    assert "require_bootstrap_reconciliation_owner(" in prepare_source
    assert "require_typed_reconciliation_owner(" not in prepare_source
    for operation in (
        lifecycle.materialize_fresh_generation,
        lifecycle.launch_reconciliation_supervisor,
        lifecycle.reconciliation_status,
        lifecycle.stop_reconciliation_generation,
    ):
        assert "require_typed_reconciliation_owner(" in inspect.getsource(operation)


def test_prepare_materializes_offline_contracts_and_stops_before_authority(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = _population(repo_root)
    owner = _FakeBootstrapOwner(population.source_forest_root)
    monkeypatch.setattr(
        lifecycle,
        "inspect_current_repository_forest",
        lambda _root: _sealed_forest(),
    )

    result = lifecycle.prepare_fresh_generation(
        repo_root=repo_root,
        state_root=tmp_path / "state",
        owner=owner,
        generation_id="eaaef-prepare-generation",
        now_ms=1,
    )

    assert result["phase"] == "awaiting_external_authority"
    assert result["provider_process_started"] is False
    assert result["unsigned_authority_request"]["unsigned_plan_r2_statement"] is not None
    assert owner.offline_request is not None
    assert owner.offline_request["expected_task_count"] == 116
    assert owner.offline_request["expected_goal_count"] == 20
    assert owner.offline_request["expected_goal_edge_count"] == 18
    assert owner.offline_request["expected_plan_count"] == 1
    assert owner.offline_request["bootstrap_task_count"] == 22
    assert owner.offline_request["held_task_count"] == 94
    assert owner.offline_request["owner_must_be_absent_during_population_write"] is True
    assert owner.offline_request["expected_execution_contract_counts"] == (
        population.execution_contract_counts
    )
    assert owner.offline_request["interface"] == (
        lifecycle.EAAEF_BOOTSTRAP_RECONCILIATION_OWNER_INTERFACE
    )


@pytest.mark.parametrize("operation", ["prepare", "materialize"])
def test_source_policy_failure_precedes_every_owner_materialization_effect(
    operation: str,
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = _population(repo_root)
    owner = _FakeOwner(population.source_forest_root)

    def reject_source_policy(
        _root: Path,
        *,
        forest: Mapping[str, Any],
    ) -> dict[str, Any]:
        del forest
        raise lifecycle.EAAEFReconciliationIdentityError("canonical CASF import differs")

    monkeypatch.setattr(lifecycle, "_require_production_source_policy", reject_source_policy)
    state_root = tmp_path / "state"
    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="canonical CASF import differs",
    ):
        if operation == "prepare":
            lifecycle.prepare_fresh_generation(
                repo_root=repo_root,
                state_root=state_root,
                owner=owner,
                generation_id="eaaef-policy-failure",
            )
        else:
            lifecycle.materialize_fresh_generation(
                repo_root=repo_root,
                state_root=state_root,
                authority={},
                trust_roots={},
                owner=owner,
                generation_id="eaaef-policy-failure",
            )

    assert owner.offline_request is None
    assert not state_root.exists()


def test_launch_rejects_changed_forest_before_owner_effect(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = _population(repo_root)
    owner = _FakeOwner(population.source_forest_root)
    state = _state(population, phase="materialized")
    store = lifecycle.ReconciliationStateStore(tmp_path / "state")
    store.create_generation("eaaef-test-generation", state)
    store.activate("eaaef-test-generation", state_cid=str(state["state_cid"]))
    changed_forest = _sealed_forest(accelerator_commit="8" * 40)
    monkeypatch.setattr(
        lifecycle,
        "inspect_current_repository_forest",
        lambda _root: changed_forest,
    )

    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="generation source is stale",
    ):
        lifecycle.launch_reconciliation_supervisor(
            repo_root=repo_root,
            state_root=store.root,
            owner=owner,
        )


def test_launch_accepts_exact_current_source_and_records_typed_receipt(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    population = _population(repo_root)
    owner = _FakeOwner(population.source_forest_root)
    birth = lifecycle.ProcessBirth(
        pid=47,
        start_time_ticks=101,
        parent_pid=2,
        boot_id="boot-launch",
        argv_sha256="sha256:" + "7" * 64,
    )
    observed_requests: list[dict[str, Any]] = []

    def launch(request: Mapping[str, Any]) -> Mapping[str, Any]:
        observed_requests.append(dict(request))
        value = {
            "schema": lifecycle.EAAEF_LAUNCH_RECEIPT_SCHEMA,
            "interface": lifecycle.EAAEF_RECONCILIATION_OWNER_INTERFACE,
            "request_cid": request["request_cid"],
            "generation_id": request["generation_id"],
            "source_forest_root": request["source_forest_root"],
            "population_cid": request["population_cid"],
            "launch_mode": "paused",
            "implementation_enabled": False,
            "provider_process_started": False,
            "typed_task_source_interface": lifecycle.EAAEF_TYPED_TASK_SOURCE_INTERFACE,
            "process_birth": birth.to_dict(),
        }
        value["receipt_cid"] = lifecycle._cid(value)
        return value

    owner.launch_reconciliation_supervisor = launch  # type: ignore[method-assign]
    state = _state(population, phase="materialized")
    state["owner_receipt"] = {"receipt_cid": "sha256:" + "8" * 64}
    state.pop("state_cid")
    state["state_cid"] = lifecycle._cid(state)
    store = lifecycle.ReconciliationStateStore(tmp_path / "state")
    store.create_generation("eaaef-test-generation", state)
    store.activate("eaaef-test-generation", state_cid=str(state["state_cid"]))

    launched = lifecycle.launch_reconciliation_supervisor(
        repo_root=repo_root,
        state_root=store.root,
        owner=owner,
        process_probe=lambda pid: birth if pid == birth.pid else None,
    )

    assert launched["phase"] == "launched_paused"
    assert launched["supervisor_birth"] == birth.to_dict()
    assert launched["provider_process_started"] is False
    assert len(observed_requests) == 1
    assert observed_requests[0]["source_forest_root"] == population.source_forest_root
    assert observed_requests[0]["provider_launch_allowed"] is False


@pytest.mark.parametrize(
    "forbidden",
    [
        {"database_path": "/tmp/control.duckdb"},
        {"raw_token": "secret"},
        {"statement_sql": "SELECT * FROM tasks"},
    ],
)
def test_typed_boundary_rejects_database_token_and_sql(forbidden: dict[str, str]) -> None:
    with pytest.raises(lifecycle.EAAEFReconciliationIdentityError, match="exposes"):
        lifecycle._assert_no_boundary_authority(forbidden)


def test_public_cli_and_source_have_no_raw_authority_or_historical_run_surface() -> None:
    destinations = _parser_destinations(lifecycle._argument_parser())
    assert destinations.isdisjoint(
        {
            "database",
            "database_path",
            "duckdb_path",
            "sql",
            "token",
            "credential",
            "key_path",
            "output",
            "output_path",
            "private_key",
            "signing_key",
            "skip_source_check",
            "branch",
            "manifest",
            "source_policy",
        }
    )
    parsed = lifecycle._argument_parser().parse_args(["launch", "--plan-r2"])
    assert parsed.plan_r2 is True
    source = inspect.getsource(lifecycle)
    assert "duckdb.connect(" not in source
    assert "os.kill(" not in source
    assert "SIGTERM" not in source
    assert "SIGKILL" not in source
    assert "run-v14" not in source


def test_owner_qualification_rejects_stale_forest_and_unclosed_remote_blockers(
    repo_root: Path,
) -> None:
    population = _population(repo_root)
    owner = _FakeOwner(population.source_forest_root)
    assert (
        lifecycle.require_typed_reconciliation_owner(
            owner, source_forest_root=population.source_forest_root
        )
        is owner
    )

    stale = _FakeOwner("sha256:" + "f" * 64)
    with pytest.raises(lifecycle.EAAEFReconciliationBlocked, match="source_forest_root"):
        lifecycle.require_typed_reconciliation_owner(
            stale, source_forest_root=population.source_forest_root
        )

    qualification = _qualification(population.source_forest_root)
    qualification["plan_r2_remote_runtime_blockers"] = ["wire_not_qualified"]
    qualification.pop("qualification_cid")
    qualification["qualification_cid"] = lifecycle._cid(qualification)
    owner.reconciliation_qualification = lambda: qualification  # type: ignore[method-assign]
    with pytest.raises(
        lifecycle.EAAEFReconciliationBlocked,
        match="plan_r2_remote_runtime_blockers",
    ):
        lifecycle.require_typed_reconciliation_owner(owner)


def test_status_and_stop_use_owner_receipts_and_exact_birth_cleanup(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population = _population(repo_root)
    birth = lifecycle.ProcessBirth(
        pid=42,
        start_time_ticks=100,
        parent_pid=2,
        boot_id="boot-one",
        argv_sha256="sha256:" + "5" * 64,
    )
    reused = lifecycle.ProcessBirth(
        pid=42,
        start_time_ticks=101,
        parent_pid=2,
        boot_id="boot-one",
        argv_sha256="sha256:" + "6" * 64,
    )
    owner = _FakeOwner(population.source_forest_root, birth=birth)
    store = lifecycle.ReconciliationStateStore(tmp_path / "state")
    state = _state(population)
    store.create_generation("eaaef-test-generation", state)
    store.activate("eaaef-test-generation", state_cid=str(state["state_cid"]))
    for artifact in ("owner.sock", "supervisor.pid.json", "stop.request"):
        (store.generation_dir("eaaef-test-generation") / artifact).write_text(
            "test", encoding="utf-8"
        )

    def probe(pid: int) -> lifecycle.ProcessBirth | None:
        assert pid == birth.pid
        return reused if owner.stopped else birth

    def forbidden_signal(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("client lifecycle must never signal a process directly")

    monkeypatch.setattr(os, "kill", forbidden_signal)
    status = lifecycle.reconciliation_status(
        state_root=store.root,
        owner=owner,
        process_probe=probe,
    )
    assert status["active"] is True
    assert status["local_birth_corroborated"] is True
    assert status["owner_supervisor_birth"] == birth.to_dict()

    stopped = lifecycle.stop_reconciliation_generation(
        state_root=store.root,
        owner=owner,
        process_probe=probe,
    )
    assert stopped["stopped"] is True
    assert stopped["stopped_process_count"] == 1
    assert set(stopped["removed_runtime_artifacts"]) == {
        "owner.sock",
        "supervisor.pid.json",
        "stop.request",
    }
    assert store.active_generation() == ""
    assert store.read_state("eaaef-test-generation")["phase"] == "stopped"


def test_status_rejects_unknown_typed_task_status(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    population = _population(repo_root)
    owner = _FakeOwner(population.source_forest_root)
    original_status = owner.reconciliation_status_snapshot

    def unknown_status(request: Mapping[str, Any]) -> Mapping[str, Any]:
        value = dict(original_status(request))
        value.pop("receipt_cid")
        value["task_status_counts"] = {"todo": 115, "invented_status": 1}
        value["receipt_cid"] = lifecycle._cid(value)
        return value

    owner.reconciliation_status_snapshot = unknown_status  # type: ignore[method-assign]
    store = lifecycle.ReconciliationStateStore(tmp_path / "state")
    state = _state(population)
    store.create_generation("eaaef-test-generation", state)
    store.activate("eaaef-test-generation", state_cid=str(state["state_cid"]))

    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="status counts are malformed",
    ):
        lifecycle.reconciliation_status(state_root=store.root, owner=owner)


def test_stop_rejects_receipt_that_omits_status_bound_birth(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    population = _population(repo_root)
    birth = lifecycle.ProcessBirth(
        pid=43,
        start_time_ticks=100,
        parent_pid=2,
        boot_id="boot-one",
        argv_sha256="sha256:" + "7" * 64,
    )
    owner = _FakeOwner(
        population.source_forest_root,
        birth=birth,
        omit_stopped_birth=True,
    )
    store = lifecycle.ReconciliationStateStore(tmp_path / "state")
    state = _state(population)
    store.create_generation("eaaef-test-generation", state)
    store.activate("eaaef-test-generation", state_cid=str(state["state_cid"]))
    artifact = store.generation_dir("eaaef-test-generation") / "owner.sock"
    artifact.write_text("test", encoding="utf-8")

    with pytest.raises(
        lifecycle.EAAEFReconciliationIdentityError,
        match="omits the status-bound supervisor birth",
    ):
        lifecycle.stop_reconciliation_generation(
            state_root=store.root,
            owner=owner,
            process_probe=lambda _pid: birth,
        )
    assert artifact.exists()
    assert store.active_generation() == "eaaef-test-generation"
