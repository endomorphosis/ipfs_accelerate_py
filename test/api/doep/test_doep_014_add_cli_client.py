"""Independent current-tree checks for the DOEP-014 CLI client adapter."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints import cli as supervisor_cli
from ipfs_accelerate_py.agent_supervisor.entrypoints.intent_service import (
    CANONICAL_OBJECTIVE_SUBMISSION_ENTRYPOINT,
    CANONICAL_OBJECTIVE_SUBMISSION_SERVICE,
    ObjectiveSubmissionContractError,
    ObjectiveSubmissionPolicyError,
    submit_objective,
)
from ipfs_datasets_py.logic.intent_ir.schema import (
    OBJECTIVE_MATERIALIZATION_RECEIPT_FORBIDDEN_FIELDS,
    ObjectiveMaterializationReceipt,
    SupervisorObjectiveIntent,
    SupervisorObjectiveSubmitterKind,
    idea_text_sha256,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
SERVICE_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "entrypoints"
    / "cli.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "outputs"
    / "DOEP-014.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "receipts"
    / "DOEP-014.json"
)

OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py",
    "test/api/doep/test_doep_014_add_cli_client.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-014.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-014.json",
)
TASK_CID = "sha256:c7bc65d22f429107910d86e863dc8e8df261dc35c40e349cc52d24590e21e18e"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _minimal_intent(**overrides: Any) -> SupervisorObjectiveIntent:
    idea = "Submit one bounded high-level idea through the existing CLI client."
    values: dict[str, Any] = {
        "intent_id": "doep.objective.intent.cli.example",
        "idea_text": idea,
        "idea_sha256": idea_text_sha256(idea),
        "submitter_kind": SupervisorObjectiveSubmitterKind.HUMAN,
        "caller": "caller:cli-example-principal",
        "repository_id": "repository:sha256:example",
        "board_namespace": "agent-supervisor-direct-objective-and-event-driven-planning-v1",
        "title_hint": "Example CLI direct objective",
        "tags": ("direct-objective", "doep", "cli"),
    }
    values.update(overrides)
    return SupervisorObjectiveIntent(**values)


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_canonical_cli_client_extends_existing_entrypoints_cli() -> None:
    assert supervisor_cli.CANONICAL_CLI_CLIENT == "SupervisorCLI@1"
    assert (
        supervisor_cli.CANONICAL_CLI_OBJECTIVE_SUBMISSION_COMMAND == "submit-objective"
    )
    assert (
        supervisor_cli.CANONICAL_CLI_OBJECTIVE_SUBMISSION_ENTRYPOINT
        == "submit_objective_cli"
    )
    assert (
        supervisor_cli.submit_objective_cli.__module__
        == "ipfs_accelerate_py.agent_supervisor.entrypoints.cli"
    )
    assert CANONICAL_OBJECTIVE_SUBMISSION_SERVICE == "SupervisorIntentService@1"
    assert CANONICAL_OBJECTIVE_SUBMISSION_ENTRYPOINT == "submit_objective"
    source = SERVICE_PATH.read_text(encoding="utf-8")
    assert "def submit_objective_cli" in source
    assert "def register_supervisor_cli" in source
    assert "def run_supervisor_cli" in source
    assert "submit-objective" in source
    assert "from .intent_service import submit_objective" in source
    assert "class CanonicalCLIClient" not in source
    assert "class ObjectiveCLIService" not in source
    assert "FormalPlanCompiler" not in source
    assert "PromptProgramMaterializer" not in source
    # ASE3 prompt-lifecycle vocabulary remains MCP-parity stable.
    assert "init" in supervisor_cli.SUPERVISOR_COMMANDS
    assert "run" in supervisor_cli.SUPERVISOR_COMMANDS
    assert (
        supervisor_cli.CANONICAL_CLI_OBJECTIVE_SUBMISSION_COMMAND
        not in supervisor_cli.SUPERVISOR_COMMANDS
    )
    assert (
        supervisor_cli.CANONICAL_CLI_OBJECTIVE_SUBMISSION_COMMAND
        in supervisor_cli.OBJECTIVE_SUBMISSION_CLI_COMMANDS
    )


def test_register_and_discovery_are_cold_and_advertise_submission() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    group = supervisor_cli.register_supervisor_cli(sub)
    help_io = io.StringIO()
    group.print_help(help_io)
    text = help_io.getvalue()
    assert "submit-objective" in text
    assert "run" in text

    parsed = parser.parse_args(
        [
            "supervisor",
            "submit-objective",
            "--intent-json",
            '{"intent_id":"x"}',
            "--output-json",
        ]
    )
    assert parsed.supervisor_command == "submit-objective"

    manifest = supervisor_cli.supervisor_cli_discovery_manifest()
    assert manifest["canonical_cli_client"] == supervisor_cli.CANONICAL_CLI_CLIENT
    assert manifest["objective_submission_commands"] == list(
        supervisor_cli.OBJECTIVE_SUBMISSION_CLI_COMMANDS
    )
    assert (
        manifest["objective_submission_entrypoint"]
        == supervisor_cli.CANONICAL_CLI_OBJECTIVE_SUBMISSION_ENTRYPOINT
    )
    assert manifest["callers_supply_authoritative_policy"] is False
    assert manifest["completion_authority"] is False
    assert manifest["competing_subsystem_created"] is False
    assert manifest["cold_help"] is True
    assert manifest["side_effect_free_parse"] is True
    assert set(manifest["commands"]) == set(supervisor_cli.SUPERVISOR_COMMANDS)


def test_submit_objective_cli_delegates_to_canonical_service() -> None:
    intent = _minimal_intent()
    expected = submit_objective(intent).to_dict()
    via_cli = supervisor_cli.submit_objective_cli(intent.to_dict())
    assert via_cli == expected
    assert via_cli["intent_id"] == intent.intent_id
    assert set(via_cli).isdisjoint(OBJECTIVE_MATERIALIZATION_RECEIPT_FORBIDDEN_FIELDS)
    assert "policy" not in via_cli
    assert "plan_root_cid" not in via_cli

    class _Receipt:
        def to_dict(self) -> dict[str, Any]:
            return {"receipt_id": "cid:test", "intent_id": intent.intent_id}

    calls: list[Any] = []

    def _fake_submit(payload: Any) -> _Receipt:
        calls.append(payload)
        return _Receipt()

    injected = supervisor_cli.submit_objective_cli(
        intent.to_dict(), submit_objective=_fake_submit
    )
    assert injected["receipt_id"] == "cid:test"
    assert len(calls) == 1


def test_run_supervisor_cli_submit_objective_json_envelope(tmp_path: Path) -> None:
    intent = _minimal_intent()
    intent_path = tmp_path / "intent.json"
    intent_path.write_text(
        json.dumps(intent.to_dict(), sort_keys=True), encoding="utf-8"
    )
    args = SimpleNamespace(
        supervisor_command="submit-objective",
        intent_file=intent_path,
        intent_json=None,
        intent_stdin=False,
        output_json=True,
        repository=None,
        state_root=None,
    )
    out = io.StringIO()
    code = supervisor_cli.run_supervisor_cli(args, stdout=out)
    assert code == supervisor_cli.EXIT_SUCCESS
    payload = json.loads(out.getvalue())
    assert payload["ok"] is True
    assert payload["command"] == "submit-objective"
    assert payload["result"]["intent_id"] == intent.intent_id
    assert payload["result"]["receipt_id"]
    assert isinstance(
        ObjectiveMaterializationReceipt.from_dict(
            {
                key: value
                for key, value in payload["result"].items()
                if key != "summary"
            }
        ),
        ObjectiveMaterializationReceipt,
    )


def test_cli_rejects_policy_overrides_and_bad_intent_sources() -> None:
    intent = _minimal_intent()
    poisoned = {**intent.to_dict(), "policy_id": "policy:forbidden"}
    args = SimpleNamespace(
        supervisor_command="submit-objective",
        intent_file=None,
        intent_json=json.dumps(poisoned),
        intent_stdin=False,
        output_json=True,
        repository=None,
        state_root=None,
    )
    out = io.StringIO()
    code = supervisor_cli.run_supervisor_cli(args, stdout=out)
    assert code == supervisor_cli.EXIT_INVALID
    payload = json.loads(out.getvalue())
    assert payload["ok"] is False
    assert "policy_id" in payload["error"]

    missing = SimpleNamespace(
        supervisor_command="submit-objective",
        intent_file=None,
        intent_json=None,
        intent_stdin=False,
        output_json=True,
        repository=None,
        state_root=None,
    )
    out2 = io.StringIO()
    code2 = supervisor_cli.run_supervisor_cli(missing, stdout=out2)
    assert code2 == supervisor_cli.EXIT_INVALID

    with pytest.raises(ObjectiveSubmissionPolicyError):
        supervisor_cli.submit_objective_cli(intent.to_dict(), submit_objective=lambda *_a, **_k: (_ for _ in ()).throw(ObjectiveSubmissionPolicyError("denied")))

    with pytest.raises(ObjectiveSubmissionContractError):
        supervisor_cli.submit_objective_cli("not-an-intent")


def test_submit_objective_does_not_open_facade_or_duckdb() -> None:
    intent = _minimal_intent()
    args = SimpleNamespace(
        supervisor_command="submit-objective",
        intent_file=None,
        intent_json=json.dumps(intent.to_dict()),
        intent_stdin=False,
        output_json=True,
        repository="/tmp/should-not-open",
        state_root="/tmp/should-not-open-state",
    )

    class _BoomSupervisor:
        @staticmethod
        def open(**_kwargs: Any) -> Any:
            raise AssertionError("facade must not open for submit-objective")

        @staticmethod
        def init_local(**_kwargs: Any) -> Any:
            raise AssertionError("init_local must not run for submit-objective")

    out = io.StringIO()
    # Injecting supervisor must be ignored for submit-objective.
    code = supervisor_cli.run_supervisor_cli(
        args, stdout=out, supervisor=_BoomSupervisor()
    )
    assert code == supervisor_cli.EXIT_SUCCESS
    payload = json.loads(out.getvalue())
    assert payload["ok"] is True
    assert payload["result"]["intent_id"] == intent.intent_id


def test_manifest_and_candidate_receipt_bind_the_exact_current_tree_outputs() -> None:
    manifest = _load_json(OUTPUT_PATH)
    receipt = _load_json(RECEIPT_PATH)

    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-014"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["plan_revision"] == "DOEP-PLAN-V5"
        assert payload["board_namespace"] == (
            "agent-supervisor-direct-objective-and-event-driven-planning-v1"
        )
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True

    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    client = manifest["canonical_cli_client"]
    assert client["identifier"] == supervisor_cli.CANONICAL_CLI_CLIENT
    assert client["module"] == "ipfs_accelerate_py.agent_supervisor.entrypoints.cli"
    assert (
        client["entrypoint"]
        == supervisor_cli.CANONICAL_CLI_OBJECTIVE_SUBMISSION_ENTRYPOINT
    )
    assert (
        client["command"]
        == supervisor_cli.CANONICAL_CLI_OBJECTIVE_SUBMISSION_COMMAND
    )
    assert client["carrier"] == "SupervisorCLI"
    assert client["submission_delegate"] == (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.intent_service.submit_objective"
    )
    assert client["callers_supply_authoritative_policy"] is False
    assert client["completion_authority"] is False
    assert client["competing_subsystem_created"] is False
    assert client["ase310_prompt_lifecycle_preserved"] is True
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["expected_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["write_scope"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["base_repositories"] == BASE_REPOSITORIES
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(SERVICE_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
    assert receipt["required_evidence"]["source_commit_tree_gitlinks"] == BASE_REPOSITORIES
