"""Fail-closed role-aware deployment attestation (FVT-053 / FVT-G200)."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CERTIFIER_PATH = (
    REPO_ROOT / "tools" / "logic" / "certify_formal_verification_toolchains.py"
)
BUILDER_PATH = (
    REPO_ROOT
    / "tools"
    / "logic"
    / "build_formal_verification_tactician_receipt.py"
)
CERTIFICATE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_toolchain_certificate.json"
)
COMPLETION_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_tactician_readiness_completion_receipt.json"
)
ROLE_RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_role_aware_deployment_receipt.json"
)

REQUIRED_ELEVATIONS = {
    "lean",
    "runtime-mtl",
    "datalog-authorization",
    "secpal-authorization",
    "coq",
    "isabelle",
}
REQUIRED_CHECK_KINDS = {"positive", "negative", "mutation", "replay"}
NON_AUTHORITATIVE_CLASSES = {
    "identity_plus_fixture_parser",
    "hermetic_adapter_shim",
    "hermetic_shadow_shim",
    "proposal_only_semantics",
}


def _load(path: Path, name: str):
    for candidate in (REPO_ROOT, REPO_ROOT / "ipfs_datasets_py"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def certifier():
    return _load(CERTIFIER_PATH, "fvt_role_aware_certifier_test")


@pytest.fixture(scope="module")
def builder():
    return _load(BUILDER_PATH, "fvt_role_aware_builder_test")


@pytest.fixture(scope="module")
def certificate(certifier) -> dict[str, Any]:
    return certifier.build_certificate(repo_root=REPO_ROOT, role_aware=True)


@pytest.fixture(scope="module")
def completion(builder) -> dict[str, Any]:
    return builder.build_receipt(repo_root=REPO_ROOT)


@pytest.fixture(scope="module")
def receipt(builder, certificate, completion) -> dict[str, Any]:
    return builder.build_role_aware_deployment_receipt(
        repo_root=REPO_ROOT,
        completion_receipt=completion,
        role_aware_certificate=certificate,
    )


def _tools(certificate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["tool_id"]: row for row in certificate["tools"]}


def test_outputs_exist_and_role_receipt_is_not_ignored() -> None:
    for path in (
        CERTIFIER_PATH,
        BUILDER_PATH,
        CERTIFICATE_PATH,
        COMPLETION_PATH,
        ROLE_RECEIPT_PATH,
    ):
        assert path.is_file(), path
    ignored = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", str(ROLE_RECEIPT_PATH)],
        check=False,
    )
    assert ignored.returncode != 0


def test_dirty_path_parser_preserves_first_filename_character(
    builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `_git_stdout` strips leading whitespace from the whole result, so an
    # unstaged-only first row can arrive as `M path` rather than ` M path`.
    monkeypatch.setattr(
        builder,
        "_git_stdout",
        lambda *_args, **_kwargs: "M .gitignore\n?? generated.json",
    )
    assert builder._dirty_paths(REPO_ROOT) == [".gitignore", "generated.json"]


def test_certificate_identity_binds_the_complete_body(
    certifier, certificate: dict[str, Any]
) -> None:
    body = {
        key: value
        for key, value in certificate.items()
        if key != "certificate_digest_sha256"
    }
    assert certificate["certificate_digest_sha256"] == certifier.content_digest(body)


def test_every_semantic_lane_retains_portable_receipt_and_full_check_sets(
    certifier, certificate: dict[str, Any]
) -> None:
    results = certificate["semantic_lane_results"]
    assert {
        "kernel",
        "kernel_rocq",
        "kernel_isabelle",
        "runtime_mtl",
        "datalog_secpal",
        "state_model",
        "protocol_tamarin",
        "protocol_proverif",
        "atp",
        "hyperltl",
        "runtime_mtl_external",
        "authorization_external",
        "attestation",
        "advisors",
    } <= {row["lane_id"] for row in results}

    for result in results:
        if result["status"] != "ran":
            assert result["block_reasons"]
            continue
        public_receipt = result["receipt"]
        assert result["digest_sha256"] == certifier.content_digest(public_receipt)
        assert result["receipt_integrity"]["valid"] is True, result["lane_id"]
        assert result["offline_observation"]["satisfied"] is True
        assert result["public_projection"]["portable_paths"] is True
        assert result["public_projection"]["raw_process_output_retained"] is False
        for per_tool in result["per_tool"].values():
            checks = per_tool["checks"]
            assert per_tool["check_set_digest_sha256"] == certifier.content_digest(
                checks
            )
            assert REQUIRED_CHECK_KINDS <= {
                check["kind"] for check in checks
            }
            assert per_tool["artifact_validation"]["valid"] is True


def test_generated_certificate_and_role_receipt_are_portable_and_redacted(
    certifier,
    builder,
    certificate: dict[str, Any],
    completion: dict[str, Any],
) -> None:
    certificate_text = json.dumps(certificate, sort_keys=True)
    assert "/home/" not in certificate_text
    assert "/tmp/" not in certificate_text
    assert "private-witness-FVT047-SECRET-AXIOM-NEVER-LEAK" not in certificate_text
    assert certificate["public_evidence_policy"]["satisfied"] is True

    role_receipt = builder.build_role_aware_deployment_receipt(
        repo_root=REPO_ROOT,
        observed_at="2026-07-31T00:00:00Z",
        completion_receipt=completion,
        role_aware_certificate=certificate,
        supervisor_evidence={
            "task_state_source": {"path": "/tmp/private/state.json"},
            "event_log_source": {"path": "/home/user/private/events.jsonl"},
            "probe": {
                "stdout": "unbounded-secret-output" * 10000,
                "secret": "raw-secret-value",
                "witness": "raw-witness-value",
            },
        },
    )
    role_text = json.dumps(role_receipt, sort_keys=True)
    assert "/home/" not in role_text
    assert "/tmp/" not in role_text
    assert "raw-secret-value" not in role_text
    assert "raw-witness-value" not in role_text
    assert "unbounded-secret-output" not in role_text
    assert role_receipt["public_evidence_policy"]["satisfied"] is True


def test_usable_pending_capabilities_keep_every_check_without_premature_promotion(
    certificate: dict[str, Any],
) -> None:
    tools = _tools(certificate)
    expected_minimums = {
        "lean": 14,
        "runtime-mtl": 12,
        "datalog-authorization": 24,
        "secpal-authorization": 24,
    }
    for tool_id, minimum in expected_minimums.items():
        tool = tools[tool_id]
        assert tool["usable"] is True, tool_id
        assert tool["production_certified"] is False, tool_id
        assert tool["promotion_blocked"] is True, tool_id
        assert "evidence_class_cannot_satisfy_production_authority" in tool[
            "block_reasons"
        ]
        assert len(tool["checks"]) >= minimum
        assert REQUIRED_CHECK_KINDS <= {
            check["kind"] for check in tool["checks"]
        }
        assert all(check["status"] == "passed" for check in tool["checks"])
        assert tool["semantic_receipt_digests"]
        assert any(
            artifact.get("sha256") for artifact in tool["artifact_identities"]
        )

    for tool_id in ("coq", "isabelle"):
        tool = tools[tool_id]
        assert tool["usable"] is False
        assert tool["unavailable"] is True
        assert tool["production_certified"] is False
        assert tool["promotion_blocked"] is True
        assert len(tool["checks"]) >= 4
        assert REQUIRED_CHECK_KINDS <= {
            check["kind"] for check in tool["checks"]
        }
        assert (
            "external_prover_installation_and_live_fanin_pending"
            in tool["block_reasons"]
        )
        assert tool["evidence_class"] == "external_prover_installation_pending"
        lane_id = "kernel_rocq" if tool_id == "coq" else "kernel_isabelle"
        lane = next(
            row
            for row in certificate["semantic_lane_results"]
            if row["lane_id"] == lane_id
        )
        assert lane["usable_elevation_allowed"] is False
        assert len(lane["per_tool"][tool_id]["checks"]) >= 12


def test_supported_missing_tools_are_blockers_not_platform_exceptions(
    certificate: dict[str, Any],
) -> None:
    managed = certificate["managed_deployment_readiness"]
    supported = {
        row["tool_id"]
        for row in managed["platform_rows"]
        if row["managed"] and row["supported"]
    }
    exceptions = {row["tool_id"] for row in managed["platform_exceptions"]}
    blockers = {row["tool_id"] for row in managed["all_blockers"]}
    assert supported.isdisjoint(exceptions)
    assert exceptions
    assert {"hyperltl", "autohyper", "mchyper"} <= supported
    assert {"hyperltl", "autohyper", "mchyper"} <= blockers
    assert not {"hyperltl", "autohyper", "mchyper"} & exceptions
    for exception in managed["platform_exceptions"]:
        assert exception["narrow_scope"] is True
        assert exception["complete"] is False
        assert exception["production_certified"] is False


def test_fixture_shim_parser_and_shadow_evidence_never_promotes(
    certificate: dict[str, Any],
) -> None:
    for tool in certificate["tools"]:
        if (
            tool["evidence_class"] in NON_AUTHORITATIVE_CLASSES
            or tool["executable_artifact_class"] == "generated_hermetic_shim"
        ):
            assert tool["production_certified"] is False, tool["tool_id"]


def test_non_certifying_authority_roles_never_promote(
    certificate: dict[str, Any],
) -> None:
    tools = _tools(certificate)
    roles = certificate["authority_roles"]["tools"]
    for tool_id, role in roles.items():
        if not role["can_satisfy_certified_authority"] and tool_id in tools:
            assert tools[tool_id]["production_certified"] is False, tool_id


def test_offline_policy_is_derived_from_lock_and_observations(
    certificate: dict[str, Any],
) -> None:
    policy = certificate["certification_policy"]
    lock_policy = policy["lock_offline_verification_policy"]
    assert policy["forbid_install"] == lock_policy["forbid_install"]
    assert policy["forbid_download"] == lock_policy["forbid_download"]
    assert policy["forbid_network"] == lock_policy["forbid_network"]
    assert policy["offline_observations"]
    assert policy["offline_policy_satisfied"] is True
    assert all(row["satisfied"] for row in policy["offline_observations"])


def test_role_receipt_is_blocked_and_explains_each_open_gate(
    receipt: dict[str, Any],
) -> None:
    assert receipt["interface"] == "RoleAwareFormalVerificationRelease@1"
    assert receipt["goal_id"] == "FVT-G200"
    assert receipt["task_id"] == "FVT-053"
    assert receipt["binding_mode"] == (
        "two_phase_source_then_attestation_publication"
    )
    assert receipt["status"] == "role_aware_deployment_blocked"
    assert receipt["deployment_blockers"]
    assert (
        receipt["acceptance"]["supported_managed_capabilities_ready"] is False
    )
    assert receipt["acceptance"]["supervisor_evidence_bound"] is False
    assert (
        receipt["acceptance"]["lean_runtime_mtl_authorization_elevated"]
        is False
    )
    assert set(receipt["acceptance"]["required_elevations_missing"]) == (
        REQUIRED_ELEVATIONS
    )
    assert receipt["source"]["attestation_excluded_from_source_tree"] is True
    assert receipt["source"]["publication_verification_required"] is True
    assert receipt["platform_exceptions"] == receipt["role_aware_certificate"][
        "managed_deployment_readiness"
    ]["platform_exceptions"]


def test_checked_in_receipt_is_content_addressed_and_not_false_ready(
    builder,
) -> None:
    checked = json.loads(ROLE_RECEIPT_PATH.read_text(encoding="utf-8"))
    stored = checked.pop("receipt_identity")
    assert stored == builder.content_digest(checked)
    assert checked["status"] != "role_aware_deployment_ready"
    assert checked["deployment_blockers"]


def test_omitted_semantic_check_changes_certificate_identity(
    certifier, certificate: dict[str, Any]
) -> None:
    mutated = copy.deepcopy(certificate)
    lean_result = next(
        row
        for row in mutated["semantic_lane_results"]
        if row["lane_id"] == "kernel"
    )
    lean_result["receipt"]["checks"].pop()
    original_body = {
        key: value
        for key, value in certificate.items()
        if key != "certificate_digest_sha256"
    }
    mutated_body = {
        key: value
        for key, value in mutated.items()
        if key != "certificate_digest_sha256"
    }
    assert certifier.content_digest(mutated_body) != certifier.content_digest(
        original_body
    )


def test_missing_artifact_identity_blocks_managed_readiness(
    certifier, certificate: dict[str, Any]
) -> None:
    tools = {
        row["tool_id"]: certifier.ToolCertification(
            **{
                key: value
                for key, value in row.items()
                if key
                in certifier.ToolCertification.__dataclass_fields__
                and key not in {"checks"}
            }
        )
        for row in certificate["tools"]
    }
    for row in certificate["tools"]:
        tools[row["tool_id"]].checks = [
            certifier.CheckResult(**check) for check in row["checks"]
        ]
    target = tools["cvc5"]
    target.artifact_identities = []
    target.executable_sha256 = None
    lock = certifier.load_lock(
        REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"
    )
    readiness = certifier.build_managed_deployment_readiness(
        lock=lock,
        tools_index=certifier.lock_tools_by_id(lock),
        tool_certs=tools,
        authority_roles=certificate["authority_roles"],
    )
    cvc5 = next(
        row for row in readiness["all_blockers"] if row["tool_id"] == "cvc5"
    )
    assert "supported_managed_installation_missing_or_shim_only" in cvc5[
        "reasons"
    ]


def test_launcher_script_cannot_stand_in_for_the_managed_prover_artifact(
    certificate: dict[str, Any],
) -> None:
    cvc5 = _tools(certificate)["cvc5"]
    assert cvc5["usable"] is True
    assert cvc5["executable_artifact_class"] == "launcher_script"
    assert cvc5["production_certified"] is False
    assert "launcher_target_artifact_unbound" in cvc5["block_reasons"]


def test_platform_mutation_moves_tool_between_exception_and_blocker(
    certifier, certificate: dict[str, Any]
) -> None:
    lock = certifier.load_lock(
        REPO_ROOT / "config" / "formal_verification_toolchains.lock.json"
    )
    tools_index = certifier.lock_tools_by_id(lock)
    row = certifier.tool_platform_support(
        tools_index["hyperltl"],
        host_platform=certifier.observed_platform_id(),
        global_supported_platforms=lock["platform_policy"][
            "supported_platforms"
        ],
    )
    assert row["supported"] is True
    mutated = copy.deepcopy(tools_index["hyperltl"])
    mutated["deployment_contract"]["supported_platforms"] = ["plan9-mips"]
    changed = certifier.tool_platform_support(
        mutated,
        host_platform=certifier.observed_platform_id(),
        global_supported_platforms=lock["platform_policy"][
            "supported_platforms"
        ],
    )
    assert changed["supported"] is False
    assert certifier.content_digest(row) != certifier.content_digest(changed)


def test_supervisor_binding_requires_canonical_cid_completion_validation_and_merge(
    builder,
    tmp_path: Path,
) -> None:
    cid = "baguqeera-test-cid"
    key = "task/v1/test-key"
    implementation_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "origin/main^"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    merge_commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "origin/main"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    implementation_tree = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "rev-parse",
            f"{implementation_commit}^{{tree}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    merge_tree = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "rev-parse",
            f"{merge_commit}^{{tree}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    completion_receipt = {
        "schema": builder.SUPERVISOR_COMPLETION_SCHEMA,
        "status": "succeeded",
        "task_id": "FVT-053",
        "canonical_task_cid": cid,
        "canonical_task_key": key,
        "implementation_commit": implementation_commit,
        "merge_commit": merge_commit,
    }
    state_path = tmp_path / "agent_release_task_state.json"
    event_path = tmp_path / "agent_release_events.jsonl"
    state = {
        "active_task_id": "",
        "active_task_cid": "",
        "implementation_in_progress": False,
        "last_implementation_task_id": "FVT-053",
        "last_implementation_task_cid": cid,
        "last_implementation_commit": implementation_commit,
        "last_merge_commit": merge_commit,
        "task_statuses": {"FVT-053": "completed"},
        "task_identities": {
            "FVT-053": {
                "canonical_task_cid": cid,
                "canonical_task_key": key,
            }
        },
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    event = {
        "type": "implementation_finished",
        "timestamp": "2026-07-31T00:00:00Z",
        "task_id": "FVT-053",
        "canonical_task_cid": cid,
        "canonical_task_key": key,
        "implementation_commit": implementation_commit,
        "validation_result": {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "target_commit": implementation_commit,
        },
        "merge_result": {
            "merged": True,
            "implementation_commit": implementation_commit,
            "merge_commit": merge_commit,
            "target_branch": "origin/main",
            "integration_commit_proof": {
                "passed": True,
                "implementation_tree": implementation_tree,
                "merge_tree": merge_tree,
            },
        },
        "completion_receipts": [completion_receipt],
        "stream_id": "event-log:sha256:" + "d" * 64,
        "snapshot_id": "event-log-snapshot:sha256:" + "e" * 64,
        "sequence": 1,
        "previous_event_id": "",
    }
    event["event_id"] = builder.content_digest(event)
    event_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    snapshot = builder.load_supervisor_evidence_snapshot(
        task_state_path=state_path,
        event_log_path=event_path,
    )
    assert builder.derive_supervisor_binding(
        snapshot,
        repo_root=REPO_ROOT,
    )["bound"] is True
    mutated = copy.deepcopy(snapshot)
    mutated["events"][0]["canonical_task_cid"] = "wrong"
    binding = builder.derive_supervisor_binding(
        mutated,
        repo_root=REPO_ROOT,
    )
    assert binding["bound"] is False
    assert "canonical_task_cid_not_bound" in binding["block_reasons"]


def test_receipt_identity_changes_with_supervisor_snapshot(
    builder, certificate, completion
) -> None:
    first = builder.build_role_aware_deployment_receipt(
        repo_root=REPO_ROOT,
        observed_at="2026-07-31T00:00:00Z",
        completion_receipt=completion,
        role_aware_certificate=certificate,
        supervisor_evidence={"task_id": "FVT-053", "task_state": {}, "events": []},
    )
    second = builder.build_role_aware_deployment_receipt(
        repo_root=REPO_ROOT,
        observed_at="2026-07-31T00:00:00Z",
        completion_receipt=completion,
        role_aware_certificate=certificate,
        supervisor_evidence={
            "task_id": "FVT-053",
            "task_state": {},
            "events": [{"event_id": "changed"}],
        },
    )
    assert first["receipt_identity"] != second["receipt_identity"]
