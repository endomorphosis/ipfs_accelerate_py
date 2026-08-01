"""Lossless specialized receipt aggregation with composite lane handlers.

FVT-065 / FVT-G203 — ``FormalVerificationSpecializedReceiptAggregation@1``.
FVT-079 re-proves acceptance when path evidence already exists
(``objective validation repair``).

Acceptance covered:

* handlers are keyed by ``(lane_id, tool_id)`` or a composite lane returns
  distinct per-tool receipts;
* kernel retains Lean, Rocq, and Isabelle evidence and protocol retains
  Tamarin and ProVerif evidence;
* state, protocol, kernel, ATP, hyperproperty, advisor, in-process and
  external authorization, in-process and external Runtime MTL, and ZKP
  certifiers are all represented;
* every check, case, binding, executable, artifact, dependency, source,
  authority ceiling, and raw receipt digest participates in the top-level
  digest;
* a second failed check of an already-present kind blocks promotion;
* mutating any retained check or identity changes the certificate digest;
* sibling tools never overwrite each other; checks are never collapsed by
  kind; installers are never run;
* objective validation repair evidence is bound on the aggregation surface.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ROLES_PATH = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "ipfs_datasets_py"
    / "logic"
    / "backends"
    / "toolchain_roles.py"
)
CERTIFIER_PATH = REPO_ROOT / "tools" / "logic" / "certify_formal_verification_toolchains.py"

INTERFACE = "FormalVerificationSpecializedReceiptAggregation@1"
GOAL_ID = "FVT-G203"
TASK_ID = "FVT-065"
REPAIR_TASK_ID = "FVT-079"
OBJECTIVE_VALIDATION_EVIDENCE = "objective validation repair"
OBJECTIVE_VALIDATION_COMMAND = (
    "PYTHONPATH=ipfs_datasets_py python -m pytest "
    "test/integration/toolchains/test_formal_verification_specialized_receipt_aggregation.py "
    "test/integration/test_formal_verification_real_tool_matrix.py -q"
)

REQUIRED_CERTIFIER_FAMILIES = {
    "state",
    "protocol",
    "kernel",
    "atp",
    "hyperproperty",
    "advisor",
    "authorization_in_process",
    "authorization_external",
    "runtime_mtl_in_process",
    "runtime_mtl_external",
    "zkp",
}

KERNEL_TOOLS = ("lean", "coq", "isabelle")
PROTOCOL_TOOLS = ("tamarin", "proverif")


def _load_module(path: Path, name: str):
    assert path.is_file(), f"missing expected output: {path}"
    datasets_root = REPO_ROOT / "ipfs_datasets_py"
    for candidate in (str(REPO_ROOT), str(datasets_root)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def roles_mod():
    return _load_module(ROLES_PATH, "toolchain_roles_specialized_agg")


@pytest.fixture(scope="module")
def certifier():
    return _load_module(CERTIFIER_PATH, "certify_toolchains_specialized_agg")


def _passed_check(tool_id: str, kind: str, suffix: str = "") -> dict[str, Any]:
    return {
        "check_id": f"{tool_id}.{kind}{suffix}",
        "kind": kind,
        "status": "passed",
        "expected": "pass",
        "observed": "pass",
        "detail": "synthetic",
    }


def _failed_check(tool_id: str, kind: str, suffix: str = "") -> dict[str, Any]:
    return {
        "check_id": f"{tool_id}.{kind}{suffix}",
        "kind": kind,
        "status": "failed",
        "expected": "pass",
        "observed": "fail",
        "detail": "synthetic failure",
    }


def _synthetic_tool_payload(
    tool_id: str,
    *,
    certified: bool = True,
    extra_checks: list[dict[str, Any]] | None = None,
    version: str = "1.0.0",
) -> dict[str, Any]:
    checks = [
        _passed_check(tool_id, "positive"),
        _passed_check(tool_id, "negative"),
        _passed_check(tool_id, "mutation"),
        _passed_check(tool_id, "replay"),
    ]
    if extra_checks:
        checks.extend(extra_checks)
    identity = {
        "executable_path": f"bin/{tool_id}",
        "version_string": version,
        "identity_probed": True,
        "artifacts": [
            {
                "kind": "semantic_executable",
                "path": f"bin/{tool_id}",
                "sha256": f"sha256:{tool_id.ljust(64, '0')[:64]}",
                "artifact_class": "native_binary",
            }
        ],
    }
    return {
        "certified": certified,
        "block_reasons": [] if certified else ["synthetic_block"],
        "checks": checks,
        "check_set_digest_sha256": "pending",
        "identity": identity,
        "artifact_validation": {"valid": True, "failures": []},
    }


def _synthetic_semantic_result(
    *,
    lane_id: str,
    property_lane_id: str,
    certifier_family: str,
    tool_ids: tuple[str, ...],
    interface: str,
    module: str,
    per_tool: dict[str, dict[str, Any]] | None = None,
    receipt_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tools = {
        tool_id: _synthetic_tool_payload(tool_id)
        for tool_id in tool_ids
    }
    if per_tool:
        tools.update(per_tool)
    receipt: dict[str, Any] = {
        "interface": interface,
        "schema_version": f"{lane_id}/v1",
        "goal_id": "FVT-SYN",
        "task_id": "FVT-SYN",
        "lock_path": "config/formal_verification_toolchains.lock.json",
        "lock_digest": "sha256:" + ("b" * 64),
        "dependencies": {tool_id: f"pin:{tool_id}" for tool_id in tool_ids},
        "certified": all(item.get("certified") for item in tools.values()),
    }
    if receipt_extra:
        receipt.update(receipt_extra)
    # Digest excludes itself; content_digest applied by caller when needed.
    return {
        "lane_id": lane_id,
        "property_lane_id": property_lane_id,
        "certifier_family": certifier_family,
        "interface": interface,
        "module": module,
        "tool_ids": list(tool_ids),
        "status": "ran",
        "certified": bool(receipt["certified"]),
        "digest_sha256": f"sha256:raw-{lane_id}",
        "receipt": receipt,
        "per_tool": tools,
        "certifier_module_sha256": f"sha256:mod-{lane_id}",
        "receipt_integrity": {"valid": True, "failures": []},
        "offline_observation": {"satisfied": True},
    }


def _full_synthetic_matrix(certifier) -> list[dict[str, Any]]:
    """One specialized result per SEMANTIC_CERTIFIER_SPECS entry."""

    results: list[dict[str, Any]] = []
    for spec in certifier.SEMANTIC_CERTIFIER_SPECS:
        results.append(
            _synthetic_semantic_result(
                lane_id=str(spec["lane_id"]),
                property_lane_id=str(spec["property_lane_id"]),
                certifier_family=str(spec["certifier_family"]),
                tool_ids=tuple(spec["tool_ids"]),
                interface=str(spec["interface"]),
                module=Path(spec["module_relative"]).as_posix(),
            )
        )
    # Fill check digests consistently with the certifier helper.
    for result in results:
        for tool_id, payload in result["per_tool"].items():
            payload["check_set_digest_sha256"] = certifier.content_digest(
                payload["checks"]
            )
    return results


# ---------------------------------------------------------------------------
# Artifact presence / surface
# ---------------------------------------------------------------------------


def test_expected_outputs_exist() -> None:
    assert ROLES_PATH.is_file()
    assert CERTIFIER_PATH.is_file()
    assert Path(__file__).is_file()


def test_specialized_aggregation_surface_constants(roles_mod, certifier) -> None:
    assert (
        roles_mod.FORMAL_VERIFICATION_SPECIALIZED_RECEIPT_AGGREGATION_INTERFACE
        == INTERFACE
    )
    assert certifier.SPECIALIZED_AGGREGATION_INTERFACE == INTERFACE
    assert certifier.SPECIALIZED_AGGREGATION_GOAL_ID == GOAL_ID
    assert certifier.SPECIALIZED_AGGREGATION_TASK_ID == TASK_ID
    assert certifier.SPECIALIZED_AGGREGATION_REPAIR_TASK_ID == REPAIR_TASK_ID
    assert (
        certifier.SPECIALIZED_AGGREGATION_OBJECTIVE_VALIDATION_EVIDENCE
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert (
        certifier.SPECIALIZED_AGGREGATION_OBJECTIVE_VALIDATION_COMMAND
        == OBJECTIVE_VALIDATION_COMMAND
    )
    assert certifier.SPECIALIZED_AGGREGATION_SCHEMA
    assert roles_mod.SPECIALIZED_RECEIPT_AGGREGATION_GOAL_ID == GOAL_ID
    assert roles_mod.SPECIALIZED_RECEIPT_AGGREGATION_TASK_ID == TASK_ID
    assert (
        roles_mod.SPECIALIZED_RECEIPT_AGGREGATION_REPAIR_TASK_ID == REPAIR_TASK_ID
    )
    assert (
        roles_mod.SPECIALIZED_RECEIPT_AGGREGATION_OBJECTIVE_VALIDATION_EVIDENCE
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert OBJECTIVE_VALIDATION_EVIDENCE == "objective validation repair"


# ---------------------------------------------------------------------------
# Handler registry: (lane_id, tool_id) without sibling overwrite
# ---------------------------------------------------------------------------


def test_handlers_keyed_by_lane_and_tool_without_sibling_overwrite(roles_mod) -> None:
    roles_mod.reset_default_policy()
    policy = roles_mod.default_promotion_policy()

    def lean_handler(**_kwargs: Any) -> dict[str, Any]:
        return {"tool_id": "lean", "certified": True, "evidence": "lean"}

    def rocq_handler(**_kwargs: Any) -> dict[str, Any]:
        return {"tool_id": "coq", "certified": True, "evidence": "rocq"}

    def isabelle_handler(**_kwargs: Any) -> dict[str, Any]:
        return {"tool_id": "isabelle", "certified": True, "evidence": "isabelle"}

    # Simulate module-level TOOL_ID inference used by real certifiers.
    lean_handler.TOOL_ID = "lean"  # type: ignore[attr-defined]
    rocq_handler.TOOL_ID = "coq"  # type: ignore[attr-defined]
    isabelle_handler.TOOL_ID = "isabelle"  # type: ignore[attr-defined]

    policy.register_lane_handler("kernel", lean_handler, replace=True)
    policy.register_lane_handler("kernel", rocq_handler, replace=True)
    policy.register_lane_handler("kernel", isabelle_handler, replace=True)

    keys = set(policy.registered_handler_keys())
    assert "kernel::lean" in keys
    assert "kernel::coq" in keys
    assert "kernel::isabelle" in keys

    assert policy.get_tool_handler("kernel", "lean") is lean_handler
    assert policy.get_tool_handler("kernel", "coq") is rocq_handler
    assert policy.get_tool_handler("kernel", "isabelle") is isabelle_handler

    composite = policy.get_lane_handler("kernel")
    assert isinstance(composite, roles_mod.CompositeLaneHandler)
    fanout = composite()
    assert fanout["composite"] is True
    assert fanout["lossless"] is True
    assert fanout["sibling_overwrite_forbidden"] is True
    assert set(fanout["per_tool_receipts"]) == {"lean", "coq", "isabelle"}
    assert fanout["per_tool_receipts"]["lean"]["evidence"] == "lean"
    assert fanout["per_tool_receipts"]["coq"]["evidence"] == "rocq"
    assert fanout["per_tool_receipts"]["isabelle"]["evidence"] == "isabelle"


def test_protocol_siblings_tamarin_and_proverif_coexist(roles_mod) -> None:
    roles_mod.reset_default_policy()
    policy = roles_mod.default_promotion_policy()

    def tamarin_handler(**_kwargs: Any) -> dict[str, Any]:
        return {"tool_id": "tamarin", "certified": True}

    def proverif_handler(**_kwargs: Any) -> dict[str, Any]:
        return {"tool_id": "proverif", "certified": True}

    tamarin_handler.TOOL_ID = "tamarin"  # type: ignore[attr-defined]
    proverif_handler.TOOL_ID = "proverif"  # type: ignore[attr-defined]

    policy.register_lane_handler("protocol", tamarin_handler, replace=True)
    policy.register_lane_handler("protocol", proverif_handler, replace=True)

    composite = policy.get_lane_handler("protocol")
    assert isinstance(composite, roles_mod.CompositeLaneHandler)
    receipts = composite()["per_tool_receipts"]
    assert set(receipts) == {"tamarin", "proverif"}


def test_explicit_tool_id_registration_and_handler_registry_key(roles_mod) -> None:
    roles_mod.reset_default_policy()
    policy = roles_mod.default_promotion_policy()

    def handler(**_kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    policy.register_lane_handler("atp", handler, tool_id="vampire", replace=True)
    assert roles_mod.handler_registry_key("atp", "vampire") == "atp::vampire"
    assert policy.get_lane_handler("atp", tool_id="vampire") is handler
    assert policy.get_tool_handler("atp", "vampire") is handler


# ---------------------------------------------------------------------------
# Aggregation: families, kernel/protocol retention, lossless digests
# ---------------------------------------------------------------------------


def test_aggregate_represents_all_required_certifier_families(certifier) -> None:
    results = _full_synthetic_matrix(certifier)
    aggregation = certifier.aggregate_specialized_receipts(results)

    assert aggregation["interface"] == INTERFACE
    assert aggregation["goal_id"] == GOAL_ID
    assert aggregation["task_id"] == TASK_ID
    assert aggregation["repair_task_id"] == REPAIR_TASK_ID
    assert (
        aggregation["objective_validation_evidence"]
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert aggregation["objective_validation_repair"] is True
    assert aggregation["all_required_certifiers_represented"] is True
    assert REQUIRED_CERTIFIER_FAMILIES <= set(
        aggregation["certifier_families_represented"]
    )
    assert aggregation["missing_certifier_families"] == []
    assert aggregation["policy"]["handlers_keyed_by_lane_and_tool"] is True
    assert aggregation["policy"]["collapse_by_check_kind"] is False
    assert aggregation["policy"]["sibling_overwrite_forbidden"] is True
    assert aggregation["policy"]["installers_never_run"] is True
    assert aggregation["acceptance"]["objective_validation_repair"] is True
    assert (
        aggregation["acceptance"]["objective_validation_evidence"]
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert aggregation["acceptance"]["repair_task_id"] == REPAIR_TASK_ID


def test_kernel_retains_lean_rocq_isabelle_and_protocol_retains_tamarin_proverif(
    certifier,
) -> None:
    results = _full_synthetic_matrix(certifier)
    aggregation = certifier.aggregate_specialized_receipts(results)

    kernel = aggregation["composite_lanes"]["kernel"]
    protocol = aggregation["composite_lanes"]["protocol"]

    assert set(KERNEL_TOOLS) <= set(kernel["tool_ids"])
    assert set(KERNEL_TOOLS) <= set(kernel["per_tool"])
    assert set(PROTOCOL_TOOLS) <= set(protocol["tool_ids"])
    assert set(PROTOCOL_TOOLS) <= set(protocol["per_tool"])

    assert set(KERNEL_TOOLS) <= set(aggregation["kernel_retained_tool_ids"])
    assert set(PROTOCOL_TOOLS) <= set(aggregation["protocol_retained_tool_ids"])

    for tool_id in KERNEL_TOOLS:
        record = kernel["per_tool"][tool_id]
        assert record["tool_id"] == tool_id
        assert record["raw_receipt_digest"]
        assert record["checks"]
        assert record["cases"]
        assert record["bindings"]
        assert record["executable"]
        assert record["artifacts"]
        assert record["sources"]
        assert record["handler_key"] == f"kernel::{tool_id}"

    for tool_id in PROTOCOL_TOOLS:
        record = protocol["per_tool"][tool_id]
        assert record["handler_key"] == f"protocol::{tool_id}"
        assert record["raw_receipt_digest"]


def test_every_evidence_field_participates_in_top_level_digest(certifier) -> None:
    results = _full_synthetic_matrix(certifier)
    aggregation = certifier.aggregate_specialized_receipts(results)
    baseline = aggregation["aggregation_digest_sha256"]
    assert baseline and len(baseline) == 64

    # Mutate a retained check → digest must change.
    mutated = copy.deepcopy(aggregation)
    lean_checks = mutated["composite_lanes"]["kernel"]["per_tool"]["lean"]["checks"]
    assert lean_checks
    lean_checks[0]["observed"] = "mutated-observation"
    mutated_digest = certifier.content_digest(
        {
            key: value
            for key, value in mutated.items()
            if key != "aggregation_digest_sha256"
        }
    )
    assert mutated_digest != baseline

    # Mutate identity / executable → digest must change.
    identity_mut = copy.deepcopy(results)
    lean_result = next(
        item for item in identity_mut if item["lane_id"] == "kernel"
    )
    lean_result["per_tool"]["lean"]["identity"]["version_string"] = "9.9.9-mutated"
    reaggregated = certifier.aggregate_specialized_receipts(identity_mut)
    assert reaggregated["aggregation_digest_sha256"] != baseline

    # Mutate raw receipt digest → digest must change.
    raw_mut = copy.deepcopy(results)
    raw_mut[0]["digest_sha256"] = "sha256:mutated-raw-receipt"
    reaggregated_raw = certifier.aggregate_specialized_receipts(raw_mut)
    assert reaggregated_raw["aggregation_digest_sha256"] != baseline


def test_digest_components_include_required_evidence_kinds(certifier) -> None:
    results = _full_synthetic_matrix(certifier)
    aggregation = certifier.aggregate_specialized_receipts(results)
    lean = aggregation["specialized_by_handler"]["kernel::lean"]
    for field in (
        "checks",
        "cases",
        "bindings",
        "executable",
        "artifacts",
        "dependencies",
        "sources",
        "authority_ceiling",
        "raw_receipt_digest",
    ):
        assert field in lean, field
    # Authority ceiling may be None without a role matrix; field still present.
    assert "authority_ceiling" in lean


# ---------------------------------------------------------------------------
# Second failed check of an already-present kind blocks promotion
# ---------------------------------------------------------------------------


def test_second_failed_check_of_already_present_kind_blocks_promotion(
    certifier,
) -> None:
    # First positive passes, second positive fails → block.
    checks = [
        certifier.CheckResult(
            check_id="lean.positive.1",
            kind="positive",
            status="passed",
            expected="pass",
            observed="pass",
        ),
        certifier.CheckResult(
            check_id="lean.positive.2",
            kind="positive",
            status="failed",
            expected="pass",
            observed="fail",
        ),
        certifier.CheckResult(
            check_id="lean.negative",
            kind="negative",
            status="passed",
            expected="pass",
            observed="pass",
        ),
        certifier.CheckResult(
            check_id="lean.mutation",
            kind="mutation",
            status="passed",
            expected="pass",
            observed="pass",
        ),
        certifier.CheckResult(
            check_id="lean.replay",
            kind="replay",
            status="passed",
            expected="pass",
            observed="pass",
        ),
    ]
    blocks, reasons = certifier.second_failed_check_blocks_promotion(checks)
    assert blocks is True
    assert any(
        reason.startswith("second_failed_check_of_already_present_kind:positive")
        for reason in reasons
    )

    results = _full_synthetic_matrix(certifier)
    lean_result = next(item for item in results if item["lane_id"] == "kernel")
    lean_result["per_tool"]["lean"] = _synthetic_tool_payload(
        "lean",
        certified=True,
        extra_checks=[_failed_check("lean", "positive", ".duplicate")],
    )
    lean_result["per_tool"]["lean"]["check_set_digest_sha256"] = (
        certifier.content_digest(lean_result["per_tool"]["lean"]["checks"])
    )
    aggregation = certifier.aggregate_specialized_receipts(results)
    lean_record = aggregation["composite_lanes"]["kernel"]["per_tool"]["lean"]
    assert lean_record["certified"] is False
    assert lean_record["promotion_blocked"] is True
    assert any(
        "second_failed_check_of_already_present_kind" in reason
        for reason in lean_record["block_reasons"]
    )


def test_normalize_semantic_checks_never_collapses_by_kind(certifier) -> None:
    raw = [
        _passed_check("tlc", "positive"),
        _failed_check("tlc", "positive", ".again"),
        _passed_check("tlc", "negative"),
        _passed_check("tlc", "mutation"),
        _passed_check("tlc", "replay"),
    ]
    normalized = certifier._normalize_semantic_checks("tlc", raw)
    positive = [check for check in normalized if check.kind == "positive"]
    assert len(positive) == 2
    assert {check.status for check in positive} == {"passed", "failed"}


# ---------------------------------------------------------------------------
# Sibling overwrite protection in aggregation
# ---------------------------------------------------------------------------


def test_aggregation_does_not_let_sibling_overwrite_handler(certifier) -> None:
    results = _full_synthetic_matrix(certifier)
    # Inject a conflicting second kernel lean result that would overwrite if
    # fan-in collapsed by tool id incorrectly using last-write-wins.
    duplicate = copy.deepcopy(
        next(item for item in results if item["lane_id"] == "kernel")
    )
    duplicate["digest_sha256"] = "sha256:should-not-overwrite"
    duplicate["per_tool"]["lean"]["identity"]["version_string"] = "OVERWRITTEN"
    results.append(duplicate)

    aggregation = certifier.aggregate_specialized_receipts(results)
    lean = aggregation["composite_lanes"]["kernel"]["per_tool"]["lean"]
    assert lean["identity"]["version_string"] != "OVERWRITTEN"
    assert lean["raw_receipt_digest"] == "sha256:raw-kernel"


# ---------------------------------------------------------------------------
# Certificate wiring (role-aware path embeds aggregation)
# ---------------------------------------------------------------------------


def test_build_certificate_embeds_aggregation_surface_when_role_aware_disabled(
    certifier,
) -> None:
    # Default path must remain cheap and still advertise the aggregation surface.
    certificate = certifier.build_certificate(repo_root=REPO_ROOT, role_aware=False)
    section = certificate["specialized_receipt_aggregation"]
    assert section["interface"] == INTERFACE
    assert section["goal_id"] == GOAL_ID
    assert section["repair_task_id"] == REPAIR_TASK_ID
    assert (
        section["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert section.get("enabled") is False
    # Mutating the aggregation section changes the certificate digest.
    body = {
        key: value
        for key, value in certificate.items()
        if key != "certificate_digest_sha256"
    }
    baseline = certificate["certificate_digest_sha256"]
    assert baseline == certifier.content_digest(body)

    mutated = copy.deepcopy(certificate)
    mutated["specialized_receipt_aggregation"] = {
        **section,
        "tamper": "yes",
    }
    mutated_body = {
        key: value
        for key, value in mutated.items()
        if key != "certificate_digest_sha256"
    }
    assert certifier.content_digest(mutated_body) != baseline


def test_semantic_specs_declare_property_lane_and_family(certifier) -> None:
    families = {str(spec["certifier_family"]) for spec in certifier.SEMANTIC_CERTIFIER_SPECS}
    assert REQUIRED_CERTIFIER_FAMILIES <= families

    kernel_specs = [
        spec
        for spec in certifier.SEMANTIC_CERTIFIER_SPECS
        if spec["property_lane_id"] == "kernel"
    ]
    protocol_specs = [
        spec
        for spec in certifier.SEMANTIC_CERTIFIER_SPECS
        if spec["property_lane_id"] == "protocol"
    ]
    assert {tool for spec in kernel_specs for tool in spec["tool_ids"]} >= set(
        KERNEL_TOOLS
    )
    assert {tool for spec in protocol_specs for tool in spec["tool_ids"]} >= set(
        PROTOCOL_TOOLS
    )


def test_authority_roles_binding_fills_ceiling_in_aggregation(certifier) -> None:
    results = _full_synthetic_matrix(certifier)
    authority_roles = {
        "tools": {
            "lean": {"authority_ceiling": "kernel"},
            "tamarin": {"authority_ceiling": "protocol"},
            "zkp-circuit": {"authority_ceiling": "attestation"},
        }
    }
    aggregation = certifier.aggregate_specialized_receipts(
        results,
        authority_roles=authority_roles,
    )
    assert (
        aggregation["composite_lanes"]["kernel"]["per_tool"]["lean"][
            "authority_ceiling"
        ]
        == "kernel"
    )
    assert (
        aggregation["composite_lanes"]["protocol"]["per_tool"]["tamarin"][
            "authority_ceiling"
        ]
        == "protocol"
    )
    assert (
        aggregation["composite_lanes"]["attestation"]["per_tool"]["zkp-circuit"][
            "authority_ceiling"
        ]
        == "attestation"
    )


# ---------------------------------------------------------------------------
# FVT-079 objective validation repair (re-prove FVT-G203)
# ---------------------------------------------------------------------------


def test_objective_validation_repair_proves_g203_acceptance(
    roles_mod, certifier
) -> None:
    """Bind and re-prove the synthetic evidence term for FVT-G203 / FVT-079."""

    assert OBJECTIVE_VALIDATION_EVIDENCE == "objective validation repair"
    assert REPAIR_TASK_ID == "FVT-079"
    assert (
        certifier.SPECIALIZED_AGGREGATION_OBJECTIVE_VALIDATION_EVIDENCE
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert certifier.SPECIALIZED_AGGREGATION_REPAIR_TASK_ID == REPAIR_TASK_ID
    assert (
        roles_mod.SPECIALIZED_RECEIPT_AGGREGATION_OBJECTIVE_VALIDATION_EVIDENCE
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert (
        roles_mod.SPECIALIZED_RECEIPT_AGGREGATION_REPAIR_TASK_ID == REPAIR_TASK_ID
    )

    # Exact-text discovery keys must appear in every declared output.
    roles_source = ROLES_PATH.read_text(encoding="utf-8")
    certifier_source = CERTIFIER_PATH.read_text(encoding="utf-8")
    test_source = Path(__file__).read_text(encoding="utf-8")
    for source in (roles_source, certifier_source, test_source):
        assert OBJECTIVE_VALIDATION_EVIDENCE in source
        assert REPAIR_TASK_ID in source
        assert GOAL_ID in source

    results = _full_synthetic_matrix(certifier)
    aggregation = certifier.aggregate_specialized_receipts(results)
    assert aggregation["objective_validation_repair"] is True
    assert (
        aggregation["objective_validation_evidence"]
        == OBJECTIVE_VALIDATION_EVIDENCE
    )
    assert aggregation["repair_task_id"] == REPAIR_TASK_ID
    assert aggregation["objective_validation_command"] == OBJECTIVE_VALIDATION_COMMAND
    assert aggregation["acceptance"]["objective_validation_repair"] is True
    assert aggregation["acceptance"]["kernel_retains_lean_rocq_isabelle"] is True
    assert aggregation["acceptance"]["protocol_retains_tamarin_proverif"] is True
    assert aggregation["acceptance"]["all_required_certifiers_represented"] is True
    assert aggregation["acceptance"]["handlers_keyed_by_lane_and_tool"] is True
    assert aggregation["acceptance"]["lossless"] is True
    assert aggregation["acceptance"]["collapse_by_check_kind"] is False
    assert aggregation["acceptance"]["sibling_overwrite_forbidden"] is True
    assert aggregation["acceptance"]["installers_never_run"] is True

    # Composite lane fan-out also advertises the repair surface (needs ≥2
    # sibling tools so get_lane_handler returns CompositeLaneHandler).
    roles_mod.reset_default_policy()
    policy = roles_mod.default_promotion_policy()

    def lean_handler(**_kwargs: Any) -> dict[str, Any]:
        return {"tool_id": "lean", "certified": True}

    def rocq_handler(**_kwargs: Any) -> dict[str, Any]:
        return {"tool_id": "coq", "certified": True}

    lean_handler.TOOL_ID = "lean"  # type: ignore[attr-defined]
    rocq_handler.TOOL_ID = "coq"  # type: ignore[attr-defined]
    policy.register_lane_handler("kernel", lean_handler, replace=True)
    policy.register_lane_handler("kernel", rocq_handler, replace=True)
    composite = policy.get_lane_handler("kernel")
    assert isinstance(composite, roles_mod.CompositeLaneHandler)
    fanout = composite()
    assert fanout["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert fanout["repair_task_id"] == REPAIR_TASK_ID
    assert fanout["objective_validation_repair"] is True
    assert fanout["goal_id"] == GOAL_ID
    assert set(fanout["per_tool_receipts"]) == {"lean", "coq"}

    policy_dict = policy.to_dict()
    surface = policy_dict["specialized_receipt_aggregation"]
    assert surface["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert surface["repair_task_id"] == REPAIR_TASK_ID
    assert surface["objective_validation_repair"] is True
    assert surface["interface"] == INTERFACE

    # Compact projection preserves the evidence term for durable certificates.
    compact = certifier._compact_specialized_receipt_aggregation(aggregation)
    assert compact["objective_validation_evidence"] == OBJECTIVE_VALIDATION_EVIDENCE
    assert compact["repair_task_id"] == REPAIR_TASK_ID
    assert compact["objective_validation_repair"] is True
    assert compact["acceptance"]["objective_validation_repair"] is True
