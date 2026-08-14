"""IPS-043: zk-seal CLI operations, JSON statuses, and cold-help hermeticity."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.cli import (
    CLI_COMMANDS,
    CLI_PROG,
    CLI_SUBSET,
    build_parser,
    main,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    RepositoryStateView,
    RequiredUnitEvidence,
    VerificationPolicyView,
    create_full_checkpoint,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    ProofMode,
    ProofTerminalStatus,
    SealStatus,
)

_DIGEST_A = "sha256:" + ("aa" * 32)
_DIGEST_B = "sha256:" + ("bb" * 32)
_DIGEST_C = "sha256:" + ("cc" * 32)
_DIGEST_D = "sha256:" + ("dd" * 32)
_DIGEST_E = "sha256:" + ("ee" * 32)
_DIGEST_F = "sha256:" + ("ff" * 32)
_PARENT = "sha256:" + ("99" * 32)
_VK = "vk/prod-1"


def _state_payload() -> dict:
    return {
        "repository_id": "repo/accelerate",
        "revision": "rev-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "source_root_cid": _DIGEST_A,
        "repository_state_cid": _DIGEST_B,
        "environment_cid": _DIGEST_C,
        "parent_revision_ids": [],
    }


def _policy_payload() -> dict:
    return {
        "policy_cid": _DIGEST_D,
        "proof_schema_version": "1",
        "canonicalization_version": "1",
        "dependency_graph_schema_version": "graph@1",
        "circuit_id": "circuit@v1",
        "verification_key_id": _VK,
    }


def _unit_payload(unit_id: str = "unit/a", **overrides: object) -> dict:
    payload = {
        "unit_id": unit_id,
        "proof_object_cid": _DIGEST_E,
        "category": "unit_test",
        "terminal_status": ProofTerminalStatus.INTEGRITY_VERIFIED.value,
        "proof_mode": ProofMode.INTEGRITY_ONLY.value,
        "required_for_seal": True,
        "freshly_verified": True,
        "cache_reused_without_fresh_verification": False,
        "circuit_id": "circuit@v1",
        "verification_key_id": _VK,
    }
    payload.update(overrides)
    return payload


def _plan_unit_payload(unit_id: str = "unit/a", **overrides: object) -> dict:
    payload = {
        "unit_id": unit_id,
        "preserved": True,
        "cache_key_complete": True,
        "admitted": True,
        "candidate_present": True,
    }
    payload.update(overrides)
    return payload


def _parent_payload() -> dict:
    return {
        "seal_cid": _PARENT,
        "repository_state_cid": _DIGEST_B,
        "source_root_cid": _DIGEST_A,
        "environment_cid": _DIGEST_C,
        "policy_cid": _DIGEST_D,
    }


def _write_json(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _run(argv: list[str]) -> tuple[int, dict]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(argv, stdout=stdout, stderr=stderr)
    text = stdout.getvalue().strip()
    assert text, f"expected JSON stdout, stderr={stderr.getvalue()!r}"
    payload = json.loads(text)
    assert payload["schema"].endswith("zk-seal-cli@1")
    assert payload["evidence_subset"] == CLI_SUBSET
    assert payload["proving_key_exported"] is False
    assert payload["witness_exported"] is False
    return code, payload


def test_cli_command_inventory() -> None:
    assert CLI_PROG == "zk-seal"
    assert CLI_SUBSET == "ips/cli@1"
    assert CLI_COMMANDS == (
        "full",
        "incremental",
        "verify",
        "plan",
        "explain-reuse",
        "explain-invalidation",
        "benchmark",
        "cache-status",
        "force-full",
    )
    parser = build_parser()
    # Subparsers are registered for every required command.
    actions = [
        action
        for action in parser._actions
        if getattr(action, "dest", None) == "command"
    ]
    assert actions
    choices = set(actions[0].choices or {})
    assert choices == set(CLI_COMMANDS)


def test_cli_full_and_force_full(tmp_path: Path) -> None:
    state = _write_json(tmp_path / "state.json", _state_payload())
    policy = _write_json(tmp_path / "policy.json", _policy_payload())
    units = _write_json(
        tmp_path / "units.json",
        [_unit_payload("unit/a"), _unit_payload("unit/b", proof_object_cid=_DIGEST_F)],
    )

    code, payload = _run(
        [
            "full",
            "--state",
            state,
            "--policy",
            policy,
            "--units",
            units,
            "--fallback-reason",
            "first_state",
            "--fallback-reason",
            "missing_parent",
        ]
    )
    assert code == 0
    assert payload["command"] == "full"
    assert payload["status"] == "sealed_full"
    assert payload["result"]["sealed"] is True
    assert payload["result"]["seal_status"] == SealStatus.SEALED_FULL.value

    code, forced = _run(
        [
            "force-full",
            "--state",
            state,
            "--policy",
            policy,
            "--units",
            units,
        ]
    )
    assert code == 0
    assert forced["command"] == "force-full"
    assert forced["status"] == "sealed_full"
    assert forced["result"]["forced_full"] is True
    assert forced["result"]["force_reason"] == "explicit_force"


def test_cli_full_rejects_simulated_evidence(tmp_path: Path) -> None:
    state = _write_json(tmp_path / "state.json", _state_payload())
    policy = _write_json(tmp_path / "policy.json", _policy_payload())
    units = _write_json(
        tmp_path / "units.json",
        [
            _unit_payload("unit/a"),
            _unit_payload(
                "unit/sim",
                proof_mode=ProofMode.SIMULATED.value,
                terminal_status=ProofTerminalStatus.SIMULATED.value,
            ),
        ],
    )
    code, payload = _run(
        ["full", "--state", state, "--policy", policy, "--units", units]
    )
    assert code == 4
    assert payload["status"] == "simulated_only"
    assert payload["result"]["sealed"] is False
    assert payload["result"]["seal_status"] == SealStatus.SIMULATED_ONLY.value
    assert "unit/sim" in payload["result"]["rejected_unit_ids"]


def test_cli_plan_and_incremental(tmp_path: Path) -> None:
    parent = _write_json(tmp_path / "parent.json", _parent_payload())
    units = _write_json(
        tmp_path / "units.json",
        [
            _plan_unit_payload("unit/reuse"),
            _plan_unit_payload(
                "unit/reprove",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
        ],
    )
    code, plan_payload = _run(
        [
            "plan",
            "--parent",
            parent,
            "--old-state",
            _DIGEST_B,
            "--new-state",
            _DIGEST_F,
            "--units",
            units,
        ]
    )
    assert code == 0
    assert plan_payload["command"] == "plan"
    assert plan_payload["status"] == "ok"
    assert plan_payload["result"]["mode"] == "incremental"
    assert "unit/reuse" in plan_payload["result"]["reusable_unit_ids"]

    code, inc = _run(
        [
            "incremental",
            "--parent",
            parent,
            "--old-state",
            _DIGEST_B,
            "--new-state",
            _DIGEST_F,
            "--units",
            units,
        ]
    )
    assert code == 0
    assert inc["command"] == "incremental"
    assert inc["result"]["plan"]["mode"] == "incremental"

    code, forced_plan = _run(
        [
            "plan",
            "--parent",
            parent,
            "--old-state",
            _DIGEST_B,
            "--new-state",
            _DIGEST_F,
            "--units",
            units,
            "--force-full",
        ]
    )
    assert code == 0
    assert forced_plan["result"]["mode"] == "full"
    assert forced_plan["status"] == "full_reproof_required"


def test_cli_verify(tmp_path: Path) -> None:
    seal = create_full_checkpoint(
        RepositoryStateView(**_state_payload()),  # type: ignore[arg-type]
        VerificationPolicyView(**_policy_payload()),  # type: ignore[arg-type]
        units=(
            RequiredUnitEvidence(**_unit_payload("unit/a")),  # type: ignore[arg-type]
        ),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
    )
    seal_path = _write_json(tmp_path / "seal.json", seal.to_canonical())
    policy = _write_json(tmp_path / "policy.json", _policy_payload())
    trusted = _write_json(tmp_path / "keys.json", [_VK, "n/a"])
    code, payload = _run(
        [
            "verify",
            "--seal",
            seal_path,
            "--policy",
            policy,
            "--trusted-keys",
            trusted,
        ]
    )
    assert code == 0
    assert payload["command"] == "verify"
    assert payload["status"] == "ok"
    assert payload["result"]["accepted"] is True


def test_cli_explain_reuse_and_invalidation(tmp_path: Path) -> None:
    seal = create_full_checkpoint(
        RepositoryStateView(**_state_payload()),  # type: ignore[arg-type]
        VerificationPolicyView(**_policy_payload()),  # type: ignore[arg-type]
        units=(
            RequiredUnitEvidence(**_unit_payload("unit/a")),  # type: ignore[arg-type]
        ),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
    )
    seal_path = _write_json(tmp_path / "seal.json", seal.to_canonical())
    code, reuse = _run(
        ["explain-reuse", "--seal", seal_path, "--unit-id", "unit/a"]
    )
    assert code == 0
    assert reuse["command"] == "explain-reuse"
    assert reuse["status"] == "ok"
    assert reuse["result"]["unit_id"] == "unit/a"
    assert reuse["result"]["substitutes_for_verification"] is False

    parent = _write_json(tmp_path / "parent.json", _parent_payload())
    units = _write_json(
        tmp_path / "units.json",
        [
            _plan_unit_payload(
                "unit/changed",
                preserved=False,
                invalidated=True,
                admitted=False,
            )
        ],
    )
    code, inv = _run(
        [
            "explain-invalidation",
            "--parent",
            parent,
            "--old-state",
            _DIGEST_B,
            "--new-state",
            _DIGEST_F,
            "--units",
            units,
            "--unit-id",
            "unit/changed",
        ]
    )
    assert code == 0
    assert inv["command"] == "explain-invalidation"
    assert inv["result"]["unit_id"] == "unit/changed"
    assert inv["result"]["invalidated"] is True
    assert inv["result"]["substitutes_for_verification"] is False


def test_cli_benchmark(tmp_path: Path) -> None:
    state = _write_json(tmp_path / "state.json", {**_state_payload(), "repository_state_cid": _DIGEST_F})
    parent = _write_json(tmp_path / "parent.json", _parent_payload())
    policy = _write_json(tmp_path / "policy.json", _policy_payload())
    units = _write_json(
        tmp_path / "units.json",
        [
            _plan_unit_payload("unit/reuse"),
            _plan_unit_payload(
                "unit/reprove",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
        ],
    )
    code, payload = _run(
        [
            "benchmark",
            "--state",
            state,
            "--parent",
            parent,
            "--policy",
            policy,
            "--units",
            units,
            "--old-state",
            _DIGEST_B,
        ]
    )
    assert code == 0
    assert payload["command"] == "benchmark"
    assert payload["status"] == "ok"
    assert payload["result"]["full_required_units"] >= 1
    assert payload["result"]["estimated"] is True


def test_cli_cache_status_types_missing_optional_capabilities() -> None:
    code, payload = _run(["cache-status", "--backend", "provekit", "--backend", "simulated"])
    assert code == 0
    assert payload["command"] == "cache-status"
    assert payload["status"] == "ok"
    cache = payload["result"]["cache"]
    assert cache["status"] == "unavailable"
    assert cache["networked"] is False
    assert cache["injected"] is False
    backends = payload["result"]["backends"]
    assert backends["simulated"]["production_seal_allowed"] is False
    assert backends["simulated"]["status"] == "simulated_only"
    assert backends["provekit"]["status"] in {"unavailable", "available", "unknown"}
    optional = payload["result"]["optional_capabilities"]
    assert optional["kit_store"]["status"] == "unavailable"


def test_cold_help_has_no_process_network_key_or_state_side_effect() -> None:
    script = r"""
import importlib
import socket
import subprocess as sp
import sys

connect_calls = []
popen_calls = []
real_connect = socket.socket.connect

def guarded_connect(self, address):
    connect_calls.append(address)
    raise AssertionError(f"network connect during cold help: {address!r}")

socket.socket.connect = guarded_connect
real_popen = sp.Popen

def guarded_popen(*args, **kwargs):
    popen_calls.append((args, kwargs))
    raise AssertionError(f"process spawn during cold help: {args!r}")

sp.Popen = guarded_popen

# Import only the CLI module path; package __init__ must stay cold-capable.
cli = importlib.import_module(
    "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.cli"
)
# Heavy modules must not load merely because the CLI module was imported.
heavy = [
    "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint",
    "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.executor",
    "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.trust",
    "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.backends",
    "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.sealer",
]
loaded = [name for name in heavy if name in sys.modules]
assert loaded == [], loaded

code = cli.main(["--help"])
assert code == 0
assert connect_calls == []
assert popen_calls == []

# Parser construction alone remains side-effect free.
parser = cli.build_parser()
assert parser.prog == "zk-seal"
command_action = next(
    action for action in parser._actions if getattr(action, "dest", None) == "command"
)
for command in cli.CLI_COMMANDS:
    assert command in (command_action.choices or {})

print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "ok" in completed.stdout
    # Help text mentions the required operations.
    help_run = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.cli "
                "import main; raise SystemExit(main(['--help']))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # argparse --help writes to stdout and exits 0.
    help_text = (help_run.stdout or "") + (help_run.stderr or "")
    for command in CLI_COMMANDS:
        assert command in help_text


def test_cli_error_is_machine_readable(tmp_path: Path) -> None:
    code, payload = _run(
        [
            "full",
            "--state",
            str(tmp_path / "missing-state.json"),
            "--policy",
            json.dumps(_policy_payload()),
        ]
    )
    assert code == 1
    assert payload["status"] == "error"
    assert payload["error"] is not None
    assert "code" in payload["error"]
    assert "message" in payload["error"]
