"""Fail-closed role-aware deployment attestation (FVT-083 / FVT-G200).

FVT-053 is retained only as legacy display context.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
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
def certificate_bundle(certifier) -> tuple[dict[str, Any], dict[str, Any]]:
    full_evidence: dict[str, Any] = {}
    certificate = certifier.build_certificate(
        repo_root=REPO_ROOT,
        role_aware=True,
        full_evidence_out=full_evidence,
    )
    return certificate, full_evidence


@pytest.fixture(scope="module")
def certificate(certificate_bundle) -> dict[str, Any]:
    return certificate_bundle[0]


@pytest.fixture(scope="module")
def source_specialized(certificate_bundle) -> dict[str, Any]:
    return certificate_bundle[1]["specialized_receipt_aggregation"]


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


def test_every_ran_semantic_lane_retains_one_receipt_and_bound_check_digests(
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
    } == {row["lane_id"] for row in results}
    ran = [row for row in results if row["status"] == "ran"]
    assert len(results) == 14
    assert sum(isinstance(row.get("receipt"), dict) for row in ran) == len(ran)

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
        assert result["projection_policy"][
            "canonical_full_receipt_retained_once"
        ] is True
        assert result["projection_policy"][
            "per_tool_checks_bound_by_digest"
        ] is True
        for per_tool in result["per_tool"].values():
            assert "checks" not in per_tool
            assert len(per_tool["check_set_digest_sha256"]) == 64
            assert REQUIRED_CHECK_KINDS <= set(
                per_tool["check_kinds_present"]
            )
            assert per_tool["artifact_validation"]["valid"] is True


def test_checked_certificate_is_compact_without_losing_handler_identities(
    certifier,
    certificate: dict[str, Any],
    source_specialized: dict[str, Any],
) -> None:
    assert CERTIFICATE_PATH.stat().st_size < 1024 * 1024
    assert all(
        "checks" not in per_tool
        for lane in certificate["semantic_lane_results"]
        for per_tool in lane["per_tool"].values()
    )
    assert all(
        "checks" not in elevation
        for elevation in certificate["role_aware"]["elevations"]
    )

    lane_by_id = {
        row["lane_id"]: row for row in certificate["semantic_lane_results"]
    }
    specialized = certificate["specialized_receipt_aggregation"]
    handlers = specialized["specialized_by_handler"]
    assert len(handlers) == 21
    assert specialized["enabled"] is True
    assert specialized["lossless"] is True
    assert specialized["source_aggregation_digest_sha256"]
    aggregation_body = {
        key: value
        for key, value in specialized.items()
        if key != "aggregation_digest_sha256"
    }
    assert specialized["aggregation_digest_sha256"] == certifier.content_digest(
        aggregation_body
    )
    source_handlers = source_specialized["specialized_by_handler"]
    for handler_key, handler in handlers.items():
        assert handler["source_tool_evidence_digest_sha256"]
        handler_body = {
            key: value
            for key, value in handler.items()
            if key != "tool_evidence_digest_sha256"
        }
        assert handler["tool_evidence_digest_sha256"] == certifier.content_digest(
            handler_body
        )
        assert handler["source_tool_evidence_digest_sha256"] == (
            source_handlers[handler_key]["tool_evidence_digest_sha256"]
        )
        assert handler["identity_digest_sha256"] == certifier.content_digest(
            source_handlers[handler_key]["identity"]
        )
        lane = lane_by_id[handler["semantic_lane_id"]]
        if lane["status"] == "ran":
            assert handler["receipt_digest_sha256"] == lane["digest_sha256"]
    composite_handlers = [
        handler_key
        for composite in specialized["composite_lanes"].values()
        for handler_key in composite["handler_keys"]
    ]
    assert len(specialized["composite_lanes"]) == 9
    assert len(composite_handlers) == 21
    assert set(composite_handlers) == set(handlers)


def test_mutated_supported_non_ran_lane_fails_semantic_binding(
    certifier,
    builder,
    certificate: dict[str, Any],
    completion: dict[str, Any],
) -> None:
    mutated = copy.deepcopy(certificate)
    lane = next(
        row
        for row in mutated["semantic_lane_results"]
        if row["lane_id"] == "kernel"
    )
    lane["status"] = "certifier_error"
    lane["block_reasons"] = ["deterministic_supported_lane_failure"]
    lane["elevated_tool_ids"] = []
    lane["semantically_usable_tool_ids"] = []
    lane.pop("receipt", None)
    mutated["certificate_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in mutated.items()
            if key != "certificate_digest_sha256"
        }
    )
    role_receipt = builder.build_role_aware_deployment_receipt(
        repo_root=REPO_ROOT,
        completion_receipt=completion,
        role_aware_certificate=mutated,
    )
    assert (
        role_receipt["acceptance"]["semantic_receipts_full_and_bound"]
        is False
    )
    assert "kernel:semantic_lane_not_run" in role_receipt[
        "deployment_blockers"
    ]


def test_generated_public_json_artifacts_are_portable_and_redacted(
    certifier,
    builder,
    certificate: dict[str, Any],
    completion: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_build_deployment_section = builder.build_deployment_section

    def _deployment_with_private_probe(**kwargs):
        deployment = original_build_deployment_section(**kwargs)
        deployment["private_probe"] = {
            "executable_path": "/home/user/.elan/bin/lake",
            "stdout": "unbounded-secret-output" * 10000,
            "stderr": "private-process-error",
            "secret": "raw-secret-value",
            "witness": "raw-witness-value",
            "artifact_digest_sha256": "sha256:" + ("a" * 64),
            "version_string": "Lake version 5.0.0",
        }
        return deployment

    monkeypatch.setattr(
        builder,
        "build_deployment_section",
        _deployment_with_private_probe,
    )
    private_completion = builder.build_receipt(
        repo_root=REPO_ROOT,
        observed_at="2026-07-31T00:00:00Z",
    )
    completion_probe = private_completion["deployment"]["private_probe"]
    assert completion_probe["executable_path"] == "<host-path-redacted>/lake"
    assert completion_probe["artifact_digest_sha256"] == "sha256:" + ("a" * 64)
    assert completion_probe["version_string"] == "Lake version 5.0.0"
    for key in ("stdout", "stderr", "secret", "witness"):
        redaction = completion_probe[key]
        assert redaction["redacted"] is True
        assert redaction["byte_length"] > 0
        assert re.fullmatch(r"[0-9a-f]{64}", redaction["sha256"])

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

    for public_artifact in (certificate, private_completion, role_receipt):
        artifact_text = json.dumps(public_artifact, sort_keys=True)
        assert "/home/" not in artifact_text
        assert "/tmp/" not in artifact_text
        assert "/private/tmp/" not in artifact_text
        assert "private-witness-FVT047-SECRET-AXIOM-NEVER-LEAK" not in artifact_text
        assert "raw-secret-value" not in artifact_text
        assert "raw-witness-value" not in artifact_text
        assert "unbounded-secret-output" not in artifact_text
        assert public_artifact["public_evidence_policy"]["satisfied"] is True

    # The normal standalone projection is covered too, independently of the
    # adversarial injection above.
    completion_text = json.dumps(completion, sort_keys=True)
    assert "/home/" not in completion_text
    assert "/tmp/" not in completion_text
    assert completion["public_evidence_policy"]["satisfied"] is True


def test_usable_pending_capabilities_keep_every_check_without_premature_promotion(
    certificate: dict[str, Any],
    certifier,
) -> None:
    tools = _tools(certificate)
    expected_minimums = {
        "runtime-mtl": 12,
        "datalog-authorization": 24,
        "secpal-authorization": 24,
    }
    expected_evidence_classes = {
        "runtime-mtl": "usable_pending_external_runtime_mtl",
        "datalog-authorization": "usable_pending_authorization_vendor_fanin",
        "secpal-authorization": "usable_pending_authorization_vendor_fanin",
    }
    # Role-aware reissue binds managed TypeScript prebuilt for offline parity
    # when the in-tree dist is absent; the runtime_mtl lane must record that
    # bind without ever building or installing.
    runtime_lane = next(
        row
        for row in certificate["semantic_lane_results"]
        if row["lane_id"] == "runtime_mtl"
    )
    prebuilt_bind = runtime_lane.get("managed_typescript_prebuilt_bind") or {}
    assert prebuilt_bind.get("certification_builds_or_installs") is False
    assert prebuilt_bind.get("reason") in {
        "in_tree_prebuilt_present",
        "managed_vendor_prebuilt_bound",
        "sealed_vendor_prebuilt_authenticated",
    }

    for tool_id, minimum in expected_minimums.items():
        tool = tools[tool_id]
        lane_id = (
            "runtime_mtl"
            if tool_id == "runtime-mtl"
            else "datalog_secpal"
        )
        lane = next(
            row
            for row in certificate["semantic_lane_results"]
            if row["lane_id"] == lane_id
        )
        fanin = lane.get("checked_vendor_fanin") or {}
        vendor_elevated = tool_id in set(
            fanin.get("eligible_tool_ids") or ()
        )
        assert tool["usable"] is True, tool_id
        if vendor_elevated:
            assert tool["production_certified"] is True, tool_id
            assert tool["promotion_blocked"] is False, tool_id
            assert tool["block_reasons"] == [], tool_id
            assert tool["evidence_class"] == (
                certifier.CHECKED_VENDOR_FANIN_SPECS[lane_id][
                    "evidence_class"
                ]
            )
            assert tool_id in certificate["role_aware"]["elevated_tool_ids"]
        else:
            assert tool["production_certified"] is False, tool_id
            assert tool["promotion_blocked"] is True, tool_id
            assert "evidence_class_cannot_satisfy_production_authority" in tool[
                "block_reasons"
            ], (tool_id, tool.get("block_reasons"), tool.get("evidence_class"))
            assert (
                tool["evidence_class"]
                == expected_evidence_classes[tool_id]
            ), tool_id
        assert len(tool["checks"]) >= minimum
        assert REQUIRED_CHECK_KINDS <= {
            check["kind"] for check in tool["checks"]
        }
        assert all(check["status"] == "passed" for check in tool["checks"]), (
            tool_id,
            [(c["check_id"], c["status"]) for c in tool["checks"]],
        )
        assert tool["semantic_receipt_digests"]
        assert any(
            artifact.get("sha256") for artifact in tool["artifact_identities"]
        )

    # FVT-083 objective validation repair is bound on the role-aware matrix.
    role_aware = certificate["role_aware"]
    assert role_aware["repair_task_id"] == "FVT-083"
    assert role_aware["objective_validation_evidence"] == (
        "objective validation repair"
    )
    assert role_aware["objective_validation_repair"] is True
    assert role_aware["acceptance"]["objective_validation_repair"] is True

    for tool_id in ("lean", "coq", "isabelle"):
        tool = tools[tool_id]
        lane_id = {
            "lean": "kernel",
            "coq": "kernel_rocq",
            "isabelle": "kernel_isabelle",
        }[tool_id]
        lane = next(
            row
            for row in certificate["semantic_lane_results"]
            if row["lane_id"] == lane_id
        )
        live = lane["live_specialized_receipt"]
        if tool_id in set(live["eligible_tool_ids"]):
            assert lane["production_elevation_allowed"] is True
            assert tool["usable"] is True
            assert tool["unavailable"] is False
            assert tool["production_certified"] is True
            assert tool["promotion_blocked"] is False
            assert tool["evidence_class"] == "live_specialized_semantic_receipt"
            assert tool_id in certificate["role_aware"]["elevated_tool_ids"]
        else:
            assert tool["production_certified"] is False
            assert tool["promotion_blocked"] is True
            assert live["per_tool_failures"].get(tool_id) or tool["block_reasons"]
        assert REQUIRED_CHECK_KINDS <= {
            check["kind"] for check in tool["checks"]
        }
        assert len(
            lane["per_tool"][tool_id]["check_set_digest_sha256"]
        ) == 64


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
    assert receipt["task_id"] == "FVT-083"
    assert receipt["legacy_display_task_id"] == "FVT-053"
    # FVT-083 is the trusted successor. FVT-053 remains display-only legacy
    # context and cannot supply supervisor release authority.
    assert receipt["repair_task_id"] == "FVT-083"
    assert receipt["objective_validation_evidence"] == "objective validation repair"
    assert receipt["objective_validation_repair"] is True
    assert "test_formal_verification_role_aware_completion.py" in (
        receipt["objective_validation_command"]
    )
    assert receipt["acceptance"]["objective_validation_repair"] is True
    assert receipt["acceptance"]["repair_task_id"] == "FVT-083"
    assert receipt["acceptance"]["role_aware_matrix_executed"] is True
    assert receipt["binding_mode"] == (
        "two_phase_source_then_attestation_publication"
    )
    assert receipt["status"] == "role_aware_deployment_blocked"
    assert receipt["deployment_blockers"]
    # Goal bindings and implementation completion are derived from the
    # current objective heap. A newly strengthened goal may be intentionally
    # unbound while its replacement task is pending; the receipt must describe
    # that state consistently instead of hard-coding a prior board count.
    completion = receipt["completion"]
    objective_child_count = completion["objective_child_count"]
    child_goals_bound = completion["child_goals_bound"]
    child_goals_unbound = list(completion["child_goals_unbound"])
    assert child_goals_bound + len(child_goals_unbound) == objective_child_count
    implementation_gate = (
        completion["implementation_status"] == "complete"
        and child_goals_bound == objective_child_count
        and not child_goals_unbound
    )
    assert (
        receipt["acceptance"]["implementation_complete_and_all_child_goals_bound"]
        is implementation_gate
    )
    if implementation_gate:
        assert "implementation_complete_and_all_child_goals_bound" not in (
            receipt["deployment_blockers"]
        )
    else:
        assert "implementation_complete_and_all_child_goals_bound" in (
            receipt["deployment_blockers"]
        )
    assert (
        receipt["acceptance"]["supported_managed_capabilities_ready"] is False
    )
    assert "supported_managed_capabilities_ready" in receipt["deployment_blockers"]
    assert receipt["acceptance"]["supervisor_evidence_bound"] is False
    assert "supervisor_evidence_bound" in receipt["deployment_blockers"]
    assert (
        receipt["acceptance"]["lean_runtime_mtl_authorization_elevated"]
        is False
    )
    assert "lean_runtime_mtl_authorization_elevated" in receipt[
        "deployment_blockers"
    ]
    present_elevations = set(
        receipt["acceptance"]["required_elevations_present"]
    )
    missing_elevations = set(
        receipt["acceptance"]["required_elevations_missing"]
    )
    required_present = present_elevations & REQUIRED_ELEVATIONS
    assert missing_elevations == REQUIRED_ELEVATIONS - required_present
    assert not required_present & missing_elevations
    assert receipt["acceptance"]["hard_zero_gates_clear"] is False
    assert "hard_zero_gates_clear" in receipt["deployment_blockers"]
    assert receipt["source"]["attestation_excluded_from_source_tree"] is True
    assert receipt["source"]["publication_verification_required"] is True
    assert receipt["platform_exceptions"] == receipt["role_aware_certificate"][
        "managed_deployment_readiness"
    ]["platform_exceptions"]
    # A supported non-ran semantic lane (the currently unavailable
    # hyperproperty vendor suite) has no canonical receipt and must block.
    assert receipt["acceptance"]["semantic_receipts_full_and_bound"] is False
    assert "semantic_receipts_full_and_bound" in receipt[
        "deployment_blockers"
    ]
    assert any(
        str(item).endswith(":semantic_lane_not_run")
        for item in receipt["deployment_blockers"]
    )


def test_duplicate_child_goal_population_cannot_fake_implementation_completion(
    builder,
    certificate: dict[str, Any],
    completion: dict[str, Any],
) -> None:
    forged = copy.deepcopy(completion)
    duplicate = copy.deepcopy(forged["child_goals"][0])
    duplicate["bound"] = True
    forged["child_goals"] = [copy.deepcopy(duplicate) for _ in range(67)]
    forged["implementation"]["status"] = "complete"
    forged["implementation"]["child_goal_count"] = 67
    forged["implementation"]["child_goals_bound"] = 67
    forged["implementation"]["child_goals_unbound"] = []
    forged["acceptance"]["implementation_complete"] = True
    forged.pop("receipt_identity", None)
    forged["receipt_identity"] = builder.content_digest(forged)

    role_receipt = builder.build_role_aware_deployment_receipt(
        repo_root=REPO_ROOT,
        completion_receipt=forged,
        role_aware_certificate=certificate,
    )
    assert (
        role_receipt["acceptance"][
            "implementation_complete_and_all_child_goals_bound"
        ]
        is False
    )
    assert (
        role_receipt["completion"]["exact_objective_child_population_bound"]
        is False
    )


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


def test_launcher_script_requires_a_bound_managed_prover_artifact(
    certificate: dict[str, Any],
) -> None:
    cvc5 = _tools(certificate)["cvc5"]
    if not cvc5["usable"]:
        # A hermetic supervisor PATH may contain no CVC5 at all.  Absence is
        # valid only when it remains unavailable and promotion-blocked.
        assert cvc5["unavailable"] is True
        assert cvc5["production_certified"] is False
        assert cvc5["promotion_blocked"] is True
        assert cvc5["block_reasons"]
        return
    assert cvc5["executable_artifact_class"] == "launcher_script"
    binding = cvc5["launcher_binding"]
    if binding["valid"] is True:
        assert binding["launcher_sha256"] == cvc5["executable_sha256"]
        assert binding["target_sha256"]
        assert cvc5["production_certified"] is True
        assert cvc5["promotion_blocked"] is False
        assert "launcher_target_artifact_unbound" not in cvc5["block_reasons"]
    else:
        assert cvc5["production_certified"] is False
        assert cvc5["promotion_blocked"] is True
        assert "launcher_target_artifact_unbound" in cvc5["block_reasons"]


def test_launcher_binding_is_structural_and_rejects_extra_shell_logic(
    certifier,
    tmp_path: Path,
) -> None:
    managed = tmp_path / "managed"
    launcher_dir = managed / "bin"
    target_dir = managed / "payload"
    launcher_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    target = target_dir / "prover"
    target.write_bytes(b"\x7fELF-current-prover-payload")
    target.chmod(0o755)
    launcher = launcher_dir / "prover"
    launcher.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        f'exec {target} "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    entry = {
        "tool_id": "prover",
        "runtime": "native",
        "pins": [],
    }
    identity = {"executable_path": str(launcher)}

    bound = certifier.bind_launcher_target_identity(entry, identity)
    assert bound["valid"] is True
    assert bound["target_sha256"] == certifier.file_digest(target)
    assert bound["target_artifact_class"] == "native_or_managed_binary"

    launcher.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "echo unreviewed-side-effect\n"
        f'exec {target} "$@"\n',
        encoding="utf-8",
    )
    rejected = certifier.bind_launcher_target_identity(entry, identity)
    assert rejected["valid"] is False
    assert "launcher_unreviewed_statement" in rejected["failures"]


def test_live_specialized_receipt_requires_self_digest_semantics_and_sources(
    certifier,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = (
        REPO_ROOT
        / "docs"
        / "architecture"
        / "formal_verification_kernel_live_certificate.json"
    )
    live_receipt = json.loads(source_path.read_text(encoding="utf-8"))
    receipt_path = tmp_path / "kernel-live.json"
    receipt_path.write_text(json.dumps(live_receipt), encoding="utf-8")
    configured = dict(certifier.LIVE_SPECIALIZED_RECEIPT_SPECS["kernel"])
    configured["path"] = Path("kernel-live.json")
    for source_relative in configured["source_modules"]:
        copied_source = tmp_path / source_relative
        copied_source.parent.mkdir(parents=True, exist_ok=True)
        copied_source.write_bytes((REPO_ROOT / source_relative).read_bytes())
    monkeypatch.setitem(
        certifier.LIVE_SPECIALIZED_RECEIPT_SPECS,
        "kernel",
        configured,
    )

    native = tmp_path / "lean"
    native.write_bytes(b"\x7fELF-lean-kernel")
    native.chmod(0o755)
    native_digest = certifier.file_digest(native)
    lean = certifier.ToolCertification(
        tool_id="lean",
        executable_path=str(native),
        executable_sha256=native_digest,
        executable_artifact_class="native_or_managed_binary",
        version_string=live_receipt["kernels"]["lean"]["version_string"],
        locked_version="v4.31.0",
        path_present=True,
        identity_probed=True,
        installed=True,
        usable=True,
        artifact_identities=[
            {
                "kind": "executable",
                "path": str(native),
                "sha256": native_digest,
                "artifact_class": "native_or_managed_binary",
            }
        ],
    )
    spec = next(
        row
        for row in certifier.SEMANTIC_CERTIFIER_SPECS
        if row["lane_id"] == "kernel"
    )
    module = certifier._load_module_from_path(
        REPO_ROOT / spec["module_relative"],
        "fvt_live_adapter_integrity_test",
    )
    valid = certifier._build_live_specialized_adapter(
        repo_root=tmp_path,
        spec=spec,
        module=module,
        tool_certs={"lean": lean},
    )
    assert valid["valid"] is True
    assert valid["eligible_tool_ids"] == ["lean"]
    assert certifier._validate_artifact_identities(
        valid["source_artifacts"],
        repo_root=tmp_path,
    )["valid"] is True

    tampered = copy.deepcopy(live_receipt)
    tampered["kernels"]["lean"]["checks"][0]["status"] = "failed"
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    bad_digest = certifier._build_live_specialized_adapter(
        repo_root=tmp_path,
        spec=spec,
        module=module,
        tool_certs={"lean": lean},
    )
    assert bad_digest["valid"] is False
    assert "live_receipt_self_digest_mismatch" in bad_digest["failures"]

    tampered["receipt_digest_sha256"] = certifier.content_digest(
        {
            key: value
            for key, value in tampered.items()
            if key != "receipt_digest_sha256"
        }
    )
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    bad_semantics = certifier._build_live_specialized_adapter(
        repo_root=tmp_path,
        spec=spec,
        module=module,
        tool_certs={"lean": lean},
    )
    assert bad_semantics["eligible_tool_ids"] == []
    assert "live_tool_checks_incomplete_or_failed" in bad_semantics[
        "per_tool_failures"
    ]["lean"]

    forged_sources = copy.deepcopy(valid["source_artifacts"])
    forged_sources[0]["sha256"] = "sha256:" + ("0" * 64)
    source_validation = certifier._validate_artifact_identities(
        forged_sources,
        repo_root=tmp_path,
    )
    assert source_validation["valid"] is False
    assert "artifact_0_sha256_mismatch" in source_validation["failures"]


def test_builder_independently_rejects_forged_live_specialized_policy(
    certifier,
    builder,
    certificate: dict[str, Any],
) -> None:
    semantic_results = copy.deepcopy(
        certificate["semantic_lane_results"]
    )
    kernel = next(
        lane for lane in semantic_results if lane["lane_id"] == "kernel"
    )
    assert kernel["live_specialized_receipt"]["valid"] is True
    kernel["live_specialized_receipt"]["file_sha256"] = (
        "sha256:" + ("0" * 64)
    )

    audit = builder._audit_semantic_lane_results(
        certifier=certifier,
        repo_root=REPO_ROOT,
        semantic_results=semantic_results,
    )
    assert audit["structurally_valid"] is False
    assert any(
        failure.endswith(
            "live_specialized:"
            "live_specialized_summary_file_sha256_mismatch"
        )
        for failure in audit["failures"]
    )


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


def test_raw_supervisor_files_cannot_replace_trusted_g212_release_evidence(
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
        task_id="FVT-053",
    )
    binding = builder.derive_supervisor_binding(
        snapshot,
        repo_root=REPO_ROOT,
    )
    assert binding["bound"] is False
    assert binding["trusted_release_evidence_bound"] is False
    assert "trusted_g212_release_evidence_not_bound" in binding["block_reasons"]
    assert (
        "raw_supervisor_state_is_not_release_evidence"
        in binding["trusted_release_evidence"]["failures"]
    )

    # Even a forged snapshot that names real, published commits and contains a
    # coherent event chain cannot acquire release authority from temporary raw
    # task-state/event files.
    mutated = copy.deepcopy(snapshot)
    mutated["events"][0]["canonical_task_cid"] = "wrong"
    mutated_binding = builder.derive_supervisor_binding(
        mutated,
        repo_root=REPO_ROOT,
    )
    assert mutated_binding["bound"] is False
    assert mutated_binding["trusted_release_evidence_bound"] is False


def test_untrusted_raw_supervisor_snapshot_cannot_affect_release_identity(
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
    assert first["receipt_identity"] == second["receipt_identity"]
    assert first["acceptance"]["supervisor_evidence_bound"] is False
    assert second["acceptance"]["supervisor_evidence_bound"] is False
