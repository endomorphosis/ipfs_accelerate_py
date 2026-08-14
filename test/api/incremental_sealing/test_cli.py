"""IPS-043: narrowly scoped zk-seal CLI."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.cli import (
    CLI_OPERATIONS,
    CLI_SUBSET,
    build_parser,
    discovery_manifest,
    main,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    create_full_checkpoint,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    ProofMode,
    ProofTerminalStatus,
)

_DIGEST_A = "sha256:" + ("aa" * 32)
_DIGEST_B = "sha256:" + ("bb" * 32)
_DIGEST_C = "sha256:" + ("cc" * 32)
_DIGEST_D = "sha256:" + ("dd" * 32)
_DIGEST_E = "sha256:" + ("ee" * 32)
_DIGEST_F = "sha256:" + ("ff" * 32)
_DIGEST_1 = "sha256:" + ("11" * 32)
_DIGEST_2 = "sha256:" + ("22" * 32)
_PARENT = "sha256:" + ("99" * 32)
_VK = "vk/prod-1"


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _state_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "repository_id": "repo/accelerate",
        "revision": "rev-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "source_root_cid": _DIGEST_A,
        "repository_state_cid": _DIGEST_B,
        "environment_cid": _DIGEST_C,
        "parent_revision_ids": [],
    }
    payload.update(overrides)
    return payload


def _policy_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "policy_cid": _DIGEST_D,
        "proof_schema_version": "1",
        "canonicalization_version": "1",
        "dependency_graph_schema_version": "graph@1",
        "circuit_id": "circuit@v1",
        "verification_key_id": _VK,
    }
    payload.update(overrides)
    return payload


def _unit_evidence(unit_id: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
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


def _plan_unit(unit_id: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "preserved": True,
        "cache_key_complete": True,
        "admitted": True,
        "candidate_present": True,
    }
    payload.update(overrides)
    return payload


def _parent_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "seal_cid": _PARENT,
        "repository_state_cid": _DIGEST_B,
        "source_root_cid": _DIGEST_A,
        "schema_version": "1",
        "canonicalization_version": "1",
        "environment_cid": _DIGEST_C,
        "policy_cid": _DIGEST_D,
    }
    payload.update(overrides)
    return payload


def _run(*argv: str) -> tuple[int, dict[str, object] | str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(list(argv), stdout=stdout, stderr=stderr)
    text = stdout.getvalue()
    try:
        payload: dict[str, object] | str = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        payload = text
    return code, payload, stderr.getvalue()


def test_cli_operations_freeze() -> None:
    from ipfs_accelerate_py.agent_supervisor.proof import incremental_sealing as package

    assert CLI_SUBSET == "ips/cli@1"
    assert CLI_OPERATIONS == (
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
    assert len(CLI_OPERATIONS) == 9
    # Package freeze and CLI module freeze must agree.
    assert package.CLI_OPERATIONS == CLI_OPERATIONS
    assert package.CLI_SUBSET == CLI_SUBSET
    manifest = discovery_manifest()
    assert manifest["operations"] == list(CLI_OPERATIONS)
    assert manifest["processes_started"] is False
    assert manifest["network_accessed"] is False
    assert manifest["keys_generated"] is False
    assert manifest["user_state_mutated"] is False
    assert manifest["auto_install"] is False
    assert manifest["proving_key_exported"] is False
    assert manifest["witness_exported"] is False

    help_text = build_parser().format_help()
    for operation in CLI_OPERATIONS:
        assert operation in help_text
    # Every operation is registered: unknown names fail closed at parse time.
    for operation in CLI_OPERATIONS:
        # Build help for the subcommand without executing sealing work.
        code, _, stderr = _run(operation, "--help")
        assert code == 0, f"{operation} --help failed: {stderr}"


def test_cold_help_has_no_side_effects() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, sys; "
                "mod = importlib.import_module("
                "'ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.cli'"
                "); "
                "assert mod.CLI_SUBSET == 'ips/cli@1'; "
                "assert len(mod.CLI_OPERATIONS) == 9; "
                "parser = mod.build_parser(); "
                "help_text = parser.format_help(); "
                "assert 'full' in help_text and 'force-full' in help_text; "
                "assert 'provekit' not in sys.modules; "
                "assert 'py_ecc' not in sys.modules; "
                "assert 'multiformats' not in sys.modules; "
                "assert 'ipfs_accelerate_py.agent_supervisor.proof."
                "incremental_sealing.full_checkpoint' not in sys.modules; "
                "assert 'ipfs_accelerate_py.agent_supervisor.proof."
                "incremental_sealing.provers' not in sys.modules; "
                "assert 'ipfs_accelerate_py.agent_supervisor.proof."
                "incremental_sealing.backends' not in sys.modules; "
                "code = mod.main(['--help']); "
                "assert code == 0"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_cli_full_and_simulated_rejection(tmp_path: Path) -> None:
    state = _write(tmp_path / "state.json", _state_payload())
    policy = _write(tmp_path / "policy.json", _policy_payload())
    units = _write(
        tmp_path / "units.json",
        [
            _unit_evidence("unit/a"),
            _unit_evidence("unit/b", proof_object_cid=_DIGEST_F),
        ],
    )
    code, payload, _ = _run(
        "full",
        "--state",
        str(state),
        "--policy",
        str(policy),
        "--units",
        str(units),
    )
    assert code == 0
    assert isinstance(payload, dict)
    assert payload["command"] == "full"
    assert payload["status"] == "ok"
    assert payload["proving_key_exported"] is False
    result = payload["result"]
    assert isinstance(result, dict)
    assert result["sealed"] is True
    assert result["seal_status"] == "sealed_full"

    sim_units = _write(
        tmp_path / "sim-units.json",
        [
            _unit_evidence("unit/a"),
            _unit_evidence(
                "unit/sim",
                proof_mode=ProofMode.SIMULATED.value,
                terminal_status=ProofTerminalStatus.SIMULATED.value,
            ),
        ],
    )
    code, payload, _ = _run(
        "full",
        "--state",
        str(state),
        "--policy",
        str(policy),
        "--units",
        str(sim_units),
    )
    assert code == 1
    assert isinstance(payload, dict)
    assert payload["command"] == "full"
    result = payload["result"]
    assert isinstance(result, dict)
    assert result["sealed"] is False
    assert result["seal_status"] == "simulated_only"


def test_cli_plan_incremental_verify_explain(tmp_path: Path) -> None:
    parent = _write(tmp_path / "parent.json", _parent_payload())
    old = _write(tmp_path / "old.json", {"cid": _DIGEST_B})
    new = _write(tmp_path / "new.json", {"cid": _DIGEST_1})
    units = _write(
        tmp_path / "units.json",
        [
            _plan_unit("unit/reuse"),
            _plan_unit(
                "unit/changed",
                preserved=False,
                invalidated=True,
                admitted=False,
                candidate_present=False,
            ),
            _plan_unit(
                "unit/new",
                preserved=False,
                added=True,
                admitted=False,
                candidate_present=False,
            ),
        ],
    )
    policy = _write(
        tmp_path / "policy.json",
        {**_policy_payload(), "changed_root_cids": [_DIGEST_2]},
    )

    code, payload, _ = _run(
        "plan",
        "--parent",
        str(parent),
        "--old",
        str(old),
        "--new",
        str(new),
        "--units",
        str(units),
        "--policy",
        str(policy),
    )
    assert code == 0
    assert isinstance(payload, dict)
    assert payload["command"] == "plan"
    result = payload["result"]
    assert isinstance(result, dict)
    assert result["mode"] == "incremental"
    assert "unit/reuse" in result["reusable_unit_ids"]

    code, payload, _ = _run(
        "incremental",
        "--parent",
        str(parent),
        "--old",
        str(old),
        "--new",
        str(new),
        "--units",
        str(units),
        "--policy",
        str(policy),
        "--no-backend-available",
    )
    # Without fetch/prove hooks the CLI executor is hermetic and may reject;
    # status remains typed JSON either way.
    assert isinstance(payload, dict)
    assert payload["command"] == "incremental"
    assert "result" in payload
    exec_result = payload["result"]
    assert isinstance(exec_result, dict)
    assert "outcome" in exec_result
    assert exec_result["plan_cid"]

    seal_obj = create_full_checkpoint(
        _state_payload(),
        _policy_payload(),
        units=[
            _unit_evidence("unit/a"),
            _unit_evidence("unit/b", proof_object_cid=_DIGEST_F),
        ],
        fallback_reasons=("first_state",),
    )
    seal_path = _write(tmp_path / "seal.json", seal_obj.to_canonical())
    keys = _write(tmp_path / "keys.json", [_VK, "n/a"])
    code, payload, _ = _run(
        "verify",
        "--seal",
        str(seal_path),
        "--trusted-keys",
        str(keys),
        "--policy",
        str(policy),
    )
    assert isinstance(payload, dict)
    assert payload["command"] == "verify"
    assert payload["proving_key_exported"] is False
    assert "witness_exported" in payload
    ver = payload["result"]
    assert isinstance(ver, dict)
    assert "accepted" in ver
    assert "reason" in ver

    code, payload, _ = _run(
        "explain-reuse",
        "--seal",
        str(seal_path),
        "--unit",
        "unit/a",
    )
    assert code == 0
    assert isinstance(payload, dict)
    assert payload["command"] == "explain-reuse"
    expl = payload["result"]
    assert isinstance(expl, dict)
    assert expl["unit_id"] == "unit/a"
    assert expl["substitutes_for_verification"] is False

    code, payload, _ = _run(
        "explain-invalidation",
        "--parent",
        str(parent),
        "--old",
        str(old),
        "--new",
        str(new),
        "--units",
        str(units),
        "--unit",
        "unit/changed",
        "--policy",
        str(policy),
    )
    assert code == 0
    assert isinstance(payload, dict)
    assert payload["command"] == "explain-invalidation"
    inv = payload["result"]
    assert isinstance(inv, dict)
    assert inv["unit_id"] == "unit/changed"
    assert inv["invalidated"] is True
    assert inv["substitutes_for_verification"] is False


def test_cli_benchmark_cache_status_force_full(tmp_path: Path) -> None:
    parent = _write(tmp_path / "parent.json", _parent_payload())
    state = _write(
        tmp_path / "state.json",
        _state_payload(repository_state_cid=_DIGEST_1, source_root_cid=_DIGEST_2),
    )
    old = _write(tmp_path / "old.json", {"cid": _DIGEST_B})
    new = _write(tmp_path / "new.json", {"cid": _DIGEST_1})
    units = _write(
        tmp_path / "units.json",
        [
            _plan_unit("unit/keep"),
            _plan_unit(
                "unit/changed",
                preserved=False,
                invalidated=True,
                admitted=False,
            ),
        ],
    )
    policy = _write(
        tmp_path / "policy.json",
        {**_policy_payload(), "changed_root_cids": [_DIGEST_2]},
    )

    code, payload, _ = _run(
        "benchmark",
        "--state",
        str(state),
        "--parent",
        str(parent),
        "--policy",
        str(policy),
        "--units",
        str(units),
        "--old",
        str(old),
    )
    assert code == 0
    assert isinstance(payload, dict)
    assert payload["command"] == "benchmark"
    result = payload["result"]
    assert isinstance(result, dict)
    assert result["estimated"] is True
    assert result["estimated_as_measured"] is False
    assert result["full_required_units"] >= 1

    code, payload, _ = _run(
        "cache-status",
        "--parent",
        str(parent),
        "--old",
        str(old),
        "--new",
        str(new),
        "--units",
        str(units),
        "--policy",
        str(policy),
    )
    assert code == 0
    assert isinstance(payload, dict)
    assert payload["command"] == "cache-status"
    cache = payload["result"]
    assert isinstance(cache, dict)
    assert cache["cache_authorizes_reuse"] is False
    assert cache["cache_is_hint_only"] is True
    assert cache["executed"] is False
    assert "unit/keep" in cache["reusable_unit_ids"]

    code, payload, _ = _run(
        "force-full",
        "--parent",
        str(parent),
        "--old",
        str(old),
        "--new",
        str(new),
        "--units",
        str(units),
        "--policy",
        str(policy),
    )
    assert code == 0
    assert isinstance(payload, dict)
    assert payload["command"] == "force-full"
    forced = payload["result"]
    assert isinstance(forced, dict)
    assert forced["mode"] == "full"
    assert forced["forced_full"] is True
    assert "full_fallback_required" in forced["fallback_reasons"]

    # force-full --seal constructs a production full checkpoint and rejects sim.
    evidence = _write(
        tmp_path / "evidence.json",
        [
            _unit_evidence("unit/a"),
            _unit_evidence(
                "unit/sim",
                proof_mode=ProofMode.SIMULATED.value,
                terminal_status=ProofTerminalStatus.SIMULATED.value,
            ),
        ],
    )
    code, payload, _ = _run(
        "force-full",
        "--parent",
        str(parent),
        "--old",
        str(old),
        "--new",
        str(new),
        "--units",
        str(evidence),
        "--policy",
        str(policy),
        "--state",
        str(state),
        "--seal",
    )
    assert code == 1
    assert isinstance(payload, dict)
    result = payload["result"]
    assert isinstance(result, dict)
    assert result["sealed"] is False
    assert result["seal_status"] == "simulated_only"


def test_cli_missing_command_is_usage_error() -> None:
    code, payload, stderr = _run()
    assert code == 2
    assert "usage" in stderr.lower() or payload == {} or isinstance(payload, str)

    code, payload, _ = _run("not-a-real-command")
    # argparse rejects unknown subcommands before handler dispatch.
    assert code == 2


def test_cli_version_discovery() -> None:
    code, payload, _ = _run("--version")
    assert code == 0
    assert isinstance(payload, dict)
    assert payload["evidence_subset"] == "ips/cli@1"
    assert payload["operations"] == list(CLI_OPERATIONS)
    assert payload["processes_started"] is False
