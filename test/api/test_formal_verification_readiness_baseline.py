"""Executable contract for FormalVerificationReadinessBaseline@1 (FVT-001 / FVT-G005).

Validates that
``docs/architecture/formal_verification_readiness_baseline.json`` is a
current-tree, machine-specific readiness ledger that:

* defines the full status ladder (implemented … unavailable);
* records parent / gitlink / origin alignment as separate dimensions;
* reports exact executable and package identities for tools;
* detects Lean shim/toolchain mismatches;
* labels synthetic and offline evidence;
* never treats source or PATH presence as usability.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_readiness_baseline.json"
)
OBJECTIVES_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_tactician_readiness.objectives.md"
)

INTERFACE = "FormalVerificationReadinessBaseline@1"
SCHEMA_VERSION = "formal-verification-readiness-baseline/v1"
GOAL_ID = "FVT-G005"
TASK_ID = "FVT-001"

STATUS_LADDER = (
    "implemented",
    "fixture_tested",
    "live_tested",
    "installed",
    "usable",
    "production_certified",
    "unsupported",
    "unavailable",
)

ALIGNMENT_DIMENSIONS = (
    "parent_commit",
    "datasets_gitlink",
    "datasets_embedded_head",
    "datasets_origin_main",
    "datasets_cleanliness",
)

GIT_TIMEOUT_SECONDS = 10.0
PROBE_TIMEOUT_SECONDS = 5.0
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _load_baseline() -> dict[str, Any]:
    assert BASELINE_PATH.is_file(), f"missing readiness baseline: {BASELINE_PATH}"
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
        env=environment,
    )


def _git_stdout(
    repository: Path,
    *arguments: str,
    allow_empty: bool = False,
) -> str | None:
    try:
        completed = _git(repository, *arguments)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    if allow_empty:
        return value
    return value or None


def _bounded_run(
    argv: list[str],
    *,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def detect_lean_shim_toolchain_mismatch(
    selected_toolchain: str | None,
    installed_toolchains: list[str] | tuple[str, ...],
) -> bool:
    """Return True when the selected Lean toolchain is not offline-installed."""

    if not selected_toolchain or not str(selected_toolchain).strip():
        return False
    installed = {item.strip() for item in installed_toolchains if item and item.strip()}
    return selected_toolchain.strip() not in installed


def _status_block(entry: dict[str, Any]) -> dict[str, bool]:
    statuses = entry["statuses"]
    assert isinstance(statuses, dict)
    for name, value in statuses.items():
        assert isinstance(value, bool), name
    return statuses  # type: ignore[return-value]


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    return _load_baseline()


def test_baseline_schema_interface_and_goal_binding(baseline: dict[str, Any]) -> None:
    assert baseline["schema_version"] == SCHEMA_VERSION
    assert baseline["interface"] == INTERFACE
    assert baseline["goal_id"] == GOAL_ID
    assert baseline["task_id"] == TASK_ID
    assert baseline["binding_mode"] == "current_tree_bounded_probes"
    assert OBJECTIVES_PATH.is_file()
    objectives = OBJECTIVES_PATH.read_text(encoding="utf-8")
    assert f"## {GOAL_ID} " in objectives
    assert "production-readiness baseline" in objectives


def test_status_ladder_is_complete_and_defined(baseline: dict[str, Any]) -> None:
    ladder = baseline["status_ladder"]
    assert isinstance(ladder, dict)
    assert set(ladder) == set(STATUS_LADDER)
    for name in STATUS_LADDER:
        text = ladder[name]
        assert isinstance(text, str) and text.strip(), name

    policy = baseline["inference_policy"]
    assert policy["path_presence_is_not_usability"] is True
    assert policy["source_presence_is_not_usability"] is True
    assert policy["path_presence_is_not_installed"] is True
    assert policy["synthetic_evidence_cannot_certify_production"] is True
    assert policy["offline_fixture_evidence_is_not_live"] is True
    assert policy["lfv_completion_is_not_deployment_certificate"] is True
    assert policy["network_download_during_verification_forbidden"] is True
    assert "exact_executable_or_package_identity" in policy["usability_requires"]
    assert "no_toolchain_shim_mismatch" in policy["usability_requires"]
    assert "non_synthetic_outcome_evidence" in policy["production_certified_requires"]


def test_tree_alignment_records_dimensions_separately(
    baseline: dict[str, Any],
) -> None:
    alignment = baseline["tree_alignment"]
    assert alignment["dimensions"] == list(ALIGNMENT_DIMENSIONS)

    parent = alignment["parent"]
    gitlink = alignment["gitlink"]
    embedded = alignment["embedded_checkout"]
    origin = alignment["origin_main"]

    # Independent keys — not a single collapsed "aligned_commit".
    assert set(parent) >= {"path", "commit", "tree"}
    assert set(gitlink) >= {"path", "mode", "commit"}
    assert set(embedded) >= {"path", "head", "clean", "matches_gitlink"}
    assert set(origin) >= {
        "ref",
        "commit",
        "matches_gitlink",
        "matches_embedded_head",
    }

    for value in (
        parent["commit"],
        parent["tree"],
        gitlink["commit"],
        embedded["head"],
        origin["commit"],
    ):
        assert COMMIT_RE.fullmatch(value), value

    assert gitlink["path"] == "ipfs_datasets_py"
    assert gitlink["mode"] == "160000"
    assert origin["ref"] == "origin/main"

    source = baseline["source"]
    assert source["parent_commit"] == parent["commit"]
    assert source["datasets_gitlink"] == gitlink["commit"]
    assert source["datasets_embedded_head"] == embedded["head"]
    assert source["datasets_origin_main"] == origin["commit"]

    # Live re-probe of current tree; values may drift from the snapshot but
    # must remain well-formed and independently observable.
    live_parent = _git_stdout(REPO_ROOT, "rev-parse", "HEAD")
    live_tree = _git_stdout(REPO_ROOT, "rev-parse", "HEAD^{tree}")
    assert live_parent and COMMIT_RE.fullmatch(live_parent)
    assert live_tree and COMMIT_RE.fullmatch(live_tree)

    ls_tree = _git_stdout(REPO_ROOT, "ls-tree", "HEAD", "ipfs_datasets_py")
    assert ls_tree is not None
    assert ls_tree.startswith("160000 commit ")
    live_gitlink = ls_tree.split()[2]
    assert COMMIT_RE.fullmatch(live_gitlink)

    datasets = REPO_ROOT / "ipfs_datasets_py"
    live_embedded = _git_stdout(datasets, "rev-parse", "HEAD")
    live_origin = _git_stdout(datasets, "rev-parse", "origin/main")
    assert live_embedded and COMMIT_RE.fullmatch(live_embedded)
    # origin/main may be absent in shallow/worktree clones; record only when present.
    if live_origin is not None:
        assert COMMIT_RE.fullmatch(live_origin)

    porcelain = _git_stdout(
        datasets,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
        allow_empty=True,
    )
    assert porcelain is not None
    live_clean = porcelain == ""

    # Snapshot honesty: if the ledger claims alignment, its internal dimensions
    # must agree with each other (not with a future machine state).
    if baseline["tree_alignment"]["aligned"] is True:
        assert embedded["matches_gitlink"] is True
        assert gitlink["commit"] == embedded["head"]
        assert origin["matches_gitlink"] is True
        assert origin["matches_embedded_head"] is True
        assert embedded["clean"] is True

    # Live cleanliness is an independent dimension from parent commit identity.
    assert isinstance(live_clean, bool)
    assert "datasets_cleanliness" in alignment["dimensions"]
    assert "parent_commit" in alignment["dimensions"]
    assert "datasets_gitlink" in alignment["dimensions"]
    assert "datasets_origin_main" in alignment["dimensions"]


def test_tools_report_exact_identities_and_status_vocabulary(
    baseline: dict[str, Any],
) -> None:
    tools = baseline["tools"]
    assert isinstance(tools, list) and len(tools) >= 8
    tool_ids = [tool["tool_id"] for tool in tools]
    assert len(tool_ids) == len(set(tool_ids))
    assert {"z3", "cvc5", "lean", "lake", "tlc", "coqc", "isabelle"} <= set(tool_ids)

    for tool in tools:
        statuses = _status_block(tool)
        assert set(statuses) == set(STATUS_LADDER)
        for name, value in statuses.items():
            assert isinstance(value, bool), f"{tool['tool_id']}.{name}"

        identity = tool["identity"]
        assert set(identity) >= {
            "executable_path",
            "version_string",
            "package_or_toolchain",
        }
        probe = tool["probe"]
        assert probe["network"] is False
        assert float(probe["timeout_seconds"]) <= 10.0

        if statuses["installed"] or statuses["usable"] or statuses["live_tested"]:
            # Exact identity required — never PATH-only.
            assert identity["executable_path"], tool["tool_id"]
            assert identity["version_string"], tool["tool_id"]
            assert identity["package_or_toolchain"], tool["tool_id"]
            assert Path(identity["executable_path"]).is_absolute()

        if statuses["unavailable"]:
            assert statuses["usable"] is False
            assert statuses["production_certified"] is False
            assert identity["executable_path"] is None
            assert identity["version_string"] is None

        if statuses["usable"]:
            assert statuses["installed"] is True
            assert statuses["unsupported"] is False
            assert statuses["unavailable"] is False

        if statuses["production_certified"]:
            assert statuses["usable"] is True
            assert statuses["live_tested"] is True
            assert tool["evidence_class"] != "synthetic"

        assert tool["evidence_class"] in {"live", "offline", "synthetic"}


def test_lean_shim_toolchain_mismatch_detection(baseline: dict[str, Any]) -> None:
    rules = baseline["detection_rules"]
    rule = rules["lean_shim_toolchain_mismatch"]
    assert rule["id"] == "lean_shim_toolchain_mismatch"
    assert rule["severity"] == "high"
    assert "selected_toolchain" in rule["inputs"]
    assert "installed_toolchains" in rule["inputs"]
    assert rule["effect_on_status"]["usable"] is False
    assert rule["effect_on_status"]["production_certified"] is False
    assert "4.32.2" in rule["audit_observation"]
    assert "4.32.1" in rule["audit_observation"]

    # Synthetic reproduction of the observed audit mismatch.
    assert (
        detect_lean_shim_toolchain_mismatch(
            "leanprover/lean4:v4.32.2",
            [
                "leanprover/lean4:v4.32.0",
                "leanprover/lean4:v4.32.1",
            ],
        )
        is True
    )
    assert (
        detect_lean_shim_toolchain_mismatch(
            "leanprover/lean4:v4.32.2",
            [
                "leanprover/lean4:v4.32.1",
                "leanprover/lean4:v4.32.2",
            ],
        )
        is False
    )
    assert detect_lean_shim_toolchain_mismatch(None, ["leanprover/lean4:v4.32.2"]) is False
    assert detect_lean_shim_toolchain_mismatch("", ["leanprover/lean4:v4.32.2"]) is False

    lean = next(tool for tool in baseline["tools"] if tool["tool_id"] == "lean")
    identity = lean["identity"]
    selected = identity["selected_toolchain"]
    installed = identity["installed_toolchains"]
    assert isinstance(selected, str) and selected
    assert isinstance(installed, list) and installed
    expected_mismatch = detect_lean_shim_toolchain_mismatch(selected, installed)
    assert identity["shim_toolchain_mismatch"] is expected_mismatch

    if expected_mismatch:
        assert lean["statuses"]["usable"] is False
        assert lean["statuses"]["production_certified"] is False

    findings = {item["id"]: item for item in baseline["known_findings"]}
    assert "lean_shim_toolchain_mismatch" in findings
    assert (
        findings["lean_shim_toolchain_mismatch"]["current_host_mismatch"]
        is expected_mismatch
    )


def test_never_infer_usability_from_source_or_path_presence(
    baseline: dict[str, Any],
) -> None:
    # Source presence without a tool must not yield usable/production_certified.
    for tool_id in ("tlc", "coqc", "isabelle", "vampire", "eprover"):
        tool = next(item for item in baseline["tools"] if item["tool_id"] == tool_id)
        for repo_path in tool.get("repository_paths") or []:
            assert (REPO_ROOT / repo_path).exists(), repo_path
        assert tool["statuses"]["usable"] is False
        assert tool["statuses"]["production_certified"] is False
        assert tool["statuses"]["installed"] is False
        assert tool["statuses"]["unavailable"] is True

    # PATH-only executable without version identity cannot be marked installed.
    for tool in baseline["tools"]:
        path = tool["identity"]["executable_path"]
        if path and shutil.which(Path(path).name) and not tool["identity"]["version_string"]:
            assert tool["statuses"]["installed"] is False
            assert tool["statuses"]["usable"] is False

    # Capability rows with only repository paths stay non-usable when labeled
    # as trust-boundary defects or incomplete pipelines.
    blocked = {
        "capability.receipt.verify_and_attest",
        "capability.counterexample.public_boundary",
        "capability.replanner.verifier_backed_closure",
        "capability.pipeline.source_vc_smt",
        "capability.examples.software_verification",
    }
    for capability in baseline["capabilities"]:
        if capability["id"] in blocked:
            assert capability["statuses"]["usable"] is False
            assert capability["statuses"]["production_certified"] is False
            for repo_path in capability.get("repository_paths") or []:
                assert (REPO_ROOT / repo_path).exists(), repo_path


def test_synthetic_and_offline_evidence_are_labeled(baseline: dict[str, Any]) -> None:
    labels = baseline["evidence_labels"]
    assert set(labels) >= {"synthetic", "offline", "live"}
    assert labels["synthetic"]
    assert labels["offline"]
    assert labels["live"]

    assert "capability.examples.software_verification" in labels["synthetic"]
    assert any(item.startswith("tool:") for item in labels["live"])

    for capability in baseline["capabilities"]:
        evidence = capability["evidence"]
        assert "synthetic" in evidence
        assert isinstance(evidence["synthetic"], bool)
        if capability["evidence_class"] == "synthetic":
            assert evidence["synthetic"] is True
            assert capability["statuses"]["production_certified"] is False
            assert capability["id"] in labels["synthetic"]
        if evidence["synthetic"] is True:
            assert capability["statuses"]["production_certified"] is False
        if capability["evidence_class"] == "offline":
            # Offline-labeled capabilities must not claim production certification.
            assert capability["statuses"]["production_certified"] is False


def test_capabilities_use_full_status_ladder_and_reference_tools(
    baseline: dict[str, Any],
) -> None:
    capabilities = baseline["capabilities"]
    assert isinstance(capabilities, list) and len(capabilities) >= 8
    ids = [item["id"] for item in capabilities]
    assert len(ids) == len(set(ids))
    assert "capability.smt.z3" in ids
    assert "capability.kernel.lean" in ids
    assert "capability.receipt.verify_and_attest" in ids

    tool_ids = {tool["tool_id"] for tool in baseline["tools"]}
    for capability in capabilities:
        statuses = _status_block(capability)
        assert set(statuses) == set(STATUS_LADDER)
        for name, value in statuses.items():
            assert isinstance(value, bool), f"{capability['id']}.{name}"

        for tool_id in capability.get("tool_ids") or []:
            assert tool_id in tool_ids, tool_id

        if statuses["unsupported"]:
            assert statuses["usable"] is False
            assert statuses["production_certified"] is False
            assert statuses["implemented"] is False

        if statuses["production_certified"]:
            assert capability["evidence_class"] != "synthetic"
            assert capability["evidence"]["synthetic"] is False


def test_known_findings_cover_audit_p0s(baseline: dict[str, Any]) -> None:
    findings = {item["id"]: item for item in baseline["known_findings"]}
    required = {
        "lean_shim_toolchain_mismatch",
        "receipt_verification_fail_open",
        "public_counterexample_raw_leak",
        "structural_repair_as_closure",
        "synthetic_examples_and_metrics",
    }
    assert required <= set(findings)
    for finding_id in required:
        item = findings[finding_id]
        assert item["summary"].strip()
        assert item["status"] in {"open", "detection_required", "resolved"}


def test_summary_matches_ledger_contents(baseline: dict[str, Any]) -> None:
    summary = baseline["summary"]
    tools = baseline["tools"]
    capabilities = baseline["capabilities"]

    assert summary["tool_count"] == len(tools)
    assert summary["capability_count"] == len(capabilities)
    assert summary["never_infer_usability_from_path_or_source"] is True

    # Readiness wave 0: no production-certified tools or capabilities yet.
    assert summary["production_certified_count"] == 0
    for tool in tools:
        assert tool["statuses"]["production_certified"] is False
    for capability in capabilities:
        assert capability["statuses"]["production_certified"] is False

    usable = {tool["tool_id"] for tool in tools if tool["statuses"]["usable"]}
    unavailable = {
        tool["tool_id"] for tool in tools if tool["statuses"]["unavailable"]
    }
    assert set(summary["usable_tools"]) == usable
    assert set(summary["unavailable_tools"]) == unavailable

    synthetic_ids = [
        capability["id"]
        for capability in capabilities
        if capability["evidence_class"] == "synthetic"
        or capability["evidence"]["synthetic"]
    ]
    assert set(summary["synthetic_capability_ids"]) == set(synthetic_ids)
    assert summary["alignment_aligned"] is baseline["tree_alignment"]["aligned"]

    lean = next(tool for tool in tools if tool["tool_id"] == "lean")
    assert (
        summary["lean_shim_mismatch_on_host"]
        is lean["identity"]["shim_toolchain_mismatch"]
    )


def test_live_bounded_probes_do_not_claim_usability_from_path_alone(
    baseline: dict[str, Any],
) -> None:
    """Re-probe common tools with tight timeouts; PATH alone is insufficient."""

    for tool_id in ("z3", "cvc5"):
        tool = next(item for item in baseline["tools"] if item["tool_id"] == tool_id)
        path = tool["identity"]["executable_path"]
        if not path or not Path(path).is_file():
            # Machine may differ from snapshot; absence must not imply usable.
            if tool["statuses"]["usable"]:
                pytest.skip(f"{tool_id} marked usable but missing on this host")
            continue
        completed = _bounded_run([path, "--version"], timeout=2.0)
        assert completed is not None, f"{tool_id} version probe failed"
        version_line = _first_line(completed.stdout) or _first_line(completed.stderr)
        assert version_line, f"{tool_id} produced empty version output"
        # Identity fields must be non-empty when usable.
        if tool["statuses"]["usable"]:
            assert tool["identity"]["version_string"]
            assert tool["identity"]["package_or_toolchain"]

    # Explicit negative: a name on PATH without a successful identity probe is
    # not installed. Simulate with a nonsense binary name that happens to resolve
    # only via which() semantics for missing tools.
    assert shutil.which("this-formal-verification-tool-does-not-exist-fvt001") is None

    missing_tool = next(item for item in baseline["tools"] if item["tool_id"] == "tlc")
    assert shutil.which("tlc") is None or missing_tool["statuses"]["installed"]
    if shutil.which("tlc") is None:
        assert missing_tool["statuses"]["installed"] is False
        assert missing_tool["statuses"]["usable"] is False
        assert missing_tool["statuses"]["unavailable"] is True


def test_baseline_is_json_object_without_placeholders(baseline: dict[str, Any]) -> None:
    raw = BASELINE_PATH.read_text(encoding="utf-8")
    for token in ("TODO", "TBD", "placeholder", "FIXME", "...", "xxx"):
        assert token not in raw, f"found placeholder token {token!r}"
    assert baseline["summary"]["tool_count"] > 0
    assert baseline["detection_rules"]["lean_shim_toolchain_mismatch"]
    assert baseline["inference_policy"]["path_presence_is_not_usability"] is True
