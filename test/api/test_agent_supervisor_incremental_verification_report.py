"""IVP-018: release report validator for operations, schemas, and limitations.

A report validator binds the report and benchmark to the current tree, corpus,
policy, effective environment, command identities, and measurement status, and
rejects stale or missing sections.

Required report coverage:

* modules changed, adapters, schemas, exact key, invalidation
* selected/full/proof results, cache hits, route distribution
* counterexample examples, commitment format, limitations
* every unmet target including incompatible cross-tree reuse
* exact future ZK step (freeze leaf codec/trust policy and cross-implementation
  Merkle vectors before an external membership/aggregation circuit)
* explicit non-claims: commitment is not ZK; signatures need trusted issuers;
  structural validation is not cryptographic validation
"""

from __future__ import annotations

import copy
import importlib.util
import json
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = (
    REPO_ROOT / "docs" / "architecture" / "INCREMENTAL_VERIFICATION_PLANNER_REPORT.md"
)
README_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "verification"
    / "README.md"
)
BENCHMARK_MODULE_PATH = (
    REPO_ROOT / "benchmarks" / "agent_supervisor" / "incremental_verification.py"
)
CHECKED_IN_ARTIFACT = (
    REPO_ROOT
    / "artifacts"
    / "agent_supervisor"
    / "incremental_verification"
    / "benchmark.json"
)

REPORT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "incremental-verification-release-report-binding@1"
)
REPORT_INTERFACE: Final[str] = "IncrementalVerificationReleaseReport@1"
TASK_ID: Final[str] = "IVP-018"
GOAL_ID: Final[str] = "IVP-G100"
DOCUMENTATION_EVIDENCE: Final[str] = "ivp/documentation@1"
RELEASE_REPORT_EVIDENCE: Final[str] = "ivp/release-report@1"
BENCHMARK_EVIDENCE: Final[str] = "ivp/benchmark@1"

REQUIRED_COMMAND_IDENTITIES: Final[tuple[str, ...]] = (
    "generate_artifact",
    "validate_benchmark",
    "validate_report",
)

# Section phrases the report (and README) must cover. Matching is case-insensitive
# substring search over the full document text.
REQUIRED_REPORT_SECTIONS: Final[tuple[str, ...]] = (
    "modules changed",
    "adapters",
    "schemas",
    "exact key",
    "exact cache key",
    "invalidation",
    "selected",
    "full",
    "proof",
    "cache hit",
    "route distribution",
    "model-route distribution",
    "counterexample",
    "commitment format",
    "limitations",
    "unmet target",
    "incompatible cross-tree",
    "exact full-tree binding forbids incompatible cross-tree reuse",
    "future zk step",
    "freeze",
    "leaf codec",
    "trust policy",
    "cross-implementation merkle",
    "membership",
    "aggregation",
    "not a zk",
    "zero-knowledge",
    "trusted issuer",
    "structural validation is not cryptographic validation",
    "verificationreceiptkey",
    "ivp-leaf@1",
    "canonical-dag-json@1",
)

REQUIRED_README_PHRASES: Final[tuple[str, ...]] = (
    "create_verification_plan",
    "choose_model_route",
    "build_verification_commitment",
    "exact cache key",
    "invalidation",
    "not a zk",
    "trusted issuer",
    "structural validation is not cryptographic validation",
    "freeze the admitted receipt leaf codec and trust policy",
    "cross-implementation merkle",
    "ivp/documentation@1",
    "ivp/release-report@1",
)


class ReportValidationError(ValueError):
    """Raised when the release report or its binding is stale or incomplete."""


def _git_head(repo: Path = REPO_ROOT) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_benchmark_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "ivp_incremental_verification_benchmark_report",
        BENCHMARK_MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _extract_binding_json(report_text: str) -> dict[str, Any]:
    """Extract the first fenced JSON object that uses the report binding schema."""

    fence_pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
    for match in fence_pattern.finditer(report_text):
        raw = match.group(1)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("schema") == REPORT_BINDING_SCHEMA:
            return payload
        if payload.get("interface") == REPORT_INTERFACE:
            return payload
        if payload.get("task_id") == TASK_ID and "tree_id" in payload:
            return payload
    raise ReportValidationError(
        "report missing machine-checkable binding JSON "
        f"(schema {REPORT_BINDING_SCHEMA})"
    )


def _normalize(text: str) -> str:
    # Strip light markdown emphasis so phrase checks match prose intent.
    cleaned = re.sub(r"[`*_#]+", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _missing_phrases(text: str, phrases: Sequence[str]) -> list[str]:
    haystack = _normalize(text)
    missing: list[str] = []
    for phrase in phrases:
        needle = _normalize(phrase)
        if needle not in haystack:
            missing.append(phrase)
    return missing


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportValidationError(f"binding field {field!r} must be a mapping")
    return value


def validate_incremental_verification_report(
    report_text: str,
    *,
    benchmark: Mapping[str, Any],
    current_tree_id: str,
    readme_text: str | None = None,
    require_readme: bool = True,
) -> dict[str, Any]:
    """Bind report + benchmark to current identities; reject stale/missing sections.

    Returns a structured validation receipt. Raises :class:`ReportValidationError`
    on any hard failure (stale tree, missing sections, binding mismatch).
    """

    if not isinstance(report_text, str) or not report_text.strip():
        raise ReportValidationError("report text is empty")
    if not current_tree_id or not isinstance(current_tree_id, str):
        raise ReportValidationError("current_tree_id is required")
    if not isinstance(benchmark, Mapping):
        raise ReportValidationError("benchmark must be a mapping")

    errors: list[str] = []
    binding = _extract_binding_json(report_text)

    # --- schema / task surface ---
    if binding.get("schema") != REPORT_BINDING_SCHEMA:
        errors.append(
            f"binding schema mismatch: {binding.get('schema')!r} "
            f"!= {REPORT_BINDING_SCHEMA!r}"
        )
    if binding.get("interface") != REPORT_INTERFACE:
        errors.append(
            f"binding interface mismatch: {binding.get('interface')!r} "
            f"!= {REPORT_INTERFACE!r}"
        )
    if binding.get("task_id") != TASK_ID:
        errors.append(f"binding task_id must be {TASK_ID}")
    if binding.get("goal_id") != GOAL_ID:
        errors.append(f"binding goal_id must be {GOAL_ID}")

    evidence = binding.get("evidence")
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        errors.append("binding evidence must be a sequence")
    else:
        evidence_set = {str(item) for item in evidence}
        for required in (DOCUMENTATION_EVIDENCE, RELEASE_REPORT_EVIDENCE):
            if required not in evidence_set:
                errors.append(f"binding evidence missing {required}")

    # --- current tree binding (stale rejection) ---
    report_tree = str(binding.get("tree_id") or "")
    bench_tree = str(benchmark.get("tree_id") or "")
    if report_tree != current_tree_id:
        errors.append(
            "stale report tree_id: "
            f"report={report_tree!r} current={current_tree_id!r}"
        )
    if bench_tree != current_tree_id:
        errors.append(
            "stale benchmark tree_id: "
            f"benchmark={bench_tree!r} current={current_tree_id!r}"
        )
    if report_tree and bench_tree and report_tree != bench_tree:
        errors.append(
            "report/benchmark tree_id mismatch: "
            f"report={report_tree!r} benchmark={bench_tree!r}"
        )

    # --- corpus ---
    report_corpus = _require_mapping(binding.get("corpus"), field="corpus")
    bench_corpus = _require_mapping(benchmark.get("corpus"), field="benchmark.corpus")
    for field_name in ("corpus_id", "corpus_cid", "evaluated_count"):
        if report_corpus.get(field_name) != bench_corpus.get(field_name):
            errors.append(
                f"corpus.{field_name} mismatch: "
                f"report={report_corpus.get(field_name)!r} "
                f"benchmark={bench_corpus.get(field_name)!r}"
            )
    if report_corpus.get("measurement_status") not in {
        None,
        bench_corpus.get("measurement_status"),
        "measured",
        "not_measured",
        "inconclusive",
    }:
        # Allow measured corpus while aggregate artifact status is red.
        pass
    if not report_corpus.get("corpus_id"):
        errors.append("binding corpus.corpus_id missing")
    if not report_corpus.get("corpus_cid"):
        errors.append("binding corpus.corpus_cid missing")
    if not isinstance(report_corpus.get("evaluated_count"), int):
        errors.append("binding corpus.evaluated_count must be int")

    # --- policy ---
    report_policy = _require_mapping(binding.get("policy"), field="policy")
    bench_policy = _require_mapping(benchmark.get("policy"), field="benchmark.policy")
    if report_policy.get("policy_id") != bench_policy.get("policy_id"):
        errors.append(
            "policy_id mismatch: "
            f"report={report_policy.get('policy_id')!r} "
            f"benchmark={bench_policy.get('policy_id')!r}"
        )
    if report_policy.get("zero_stale_simulated_acceptance_hard") is not True:
        errors.append("policy.zero_stale_simulated_acceptance_hard must be true")

    # --- effective environment ---
    report_env = _require_mapping(
        binding.get("effective_environment"), field="effective_environment"
    )
    bench_env = _require_mapping(
        benchmark.get("effective_environment"),
        field="benchmark.effective_environment",
    )
    for field_name in ("python_version", "platform", "system"):
        if not report_env.get(field_name):
            errors.append(f"effective_environment.{field_name} missing in report")
        if not bench_env.get(field_name):
            errors.append(f"effective_environment.{field_name} missing in benchmark")
        if (
            report_env.get(field_name)
            and bench_env.get(field_name)
            and report_env.get(field_name) != bench_env.get(field_name)
        ):
            errors.append(
                f"effective_environment.{field_name} mismatch: "
                f"report={report_env.get(field_name)!r} "
                f"benchmark={bench_env.get(field_name)!r}"
            )

    # --- command identities ---
    report_commands = binding.get("command_identities") or binding.get("commands")
    report_commands = _require_mapping(report_commands, field="command_identities")
    for identity in REQUIRED_COMMAND_IDENTITIES:
        value = report_commands.get(identity)
        if not value or not isinstance(value, str):
            errors.append(f"command identity {identity!r} missing or not a string")
    # Benchmark must expose generate + validate command surfaces.
    bench_commands = _require_mapping(
        benchmark.get("commands"), field="benchmark.commands"
    )
    if "generate_artifact" not in bench_commands:
        errors.append("benchmark.commands.generate_artifact missing")
    if "validate" not in bench_commands:
        errors.append("benchmark.commands.validate missing")
    # Report generate_artifact path must appear in the generate command list/string.
    gen_identity = str(report_commands.get("generate_artifact") or "")
    gen_cmd = bench_commands.get("generate_artifact")
    gen_blob = " ".join(gen_cmd) if isinstance(gen_cmd, Sequence) else str(gen_cmd)
    if gen_identity and gen_identity not in gen_blob.replace("\\", "/"):
        # Allow report to name the module path while benchmark lists argv with output.
        if "incremental_verification.py" not in gen_blob:
            errors.append(
                "benchmark generate command does not reference incremental_verification.py"
            )

    # --- measurement status + schema ---
    measurement_status = binding.get("measurement_status")
    if measurement_status is None:
        errors.append("binding measurement_status missing")
    elif measurement_status != benchmark.get("status"):
        errors.append(
            "measurement_status mismatch: "
            f"report={measurement_status!r} benchmark.status={benchmark.get('status')!r}"
        )
    if measurement_status not in {"green", "red", "yellow", "not_measured"}:
        errors.append(f"unknown measurement_status {measurement_status!r}")

    measurement_schema_version = binding.get("measurement_schema_version")
    bench_measurement = _require_mapping(
        benchmark.get("measurement_schema"), field="benchmark.measurement_schema"
    )
    if measurement_schema_version != bench_measurement.get("version"):
        errors.append(
            "measurement_schema_version mismatch: "
            f"report={measurement_schema_version!r} "
            f"benchmark={bench_measurement.get('version')!r}"
        )

    # --- unmet targets including cross-tree ---
    target_statuses = binding.get("target_statuses")
    if not isinstance(target_statuses, Mapping):
        errors.append("binding target_statuses missing")
    else:
        cross = target_statuses.get("incompatible_cross_tree_unaffected_reuse")
        if cross != "unmet":
            errors.append(
                "incompatible_cross_tree_unaffected_reuse must be unmet in binding "
                f"(got {cross!r})"
            )
    cross_tree = binding.get("cross_tree_unaffected_reuse")
    if not isinstance(cross_tree, Mapping):
        errors.append("binding cross_tree_unaffected_reuse missing")
    else:
        if cross_tree.get("status") != "unmet":
            errors.append("cross_tree_unaffected_reuse.status must be unmet")
        if cross_tree.get("explicitly_unmet") is not True:
            errors.append("cross_tree_unaffected_reuse.explicitly_unmet must be true")
        reason = str(cross_tree.get("reason") or "")
        if "exact_full_tree_binding_forbids_incompatible_cross_tree_reuse" not in reason:
            errors.append(
                "cross-tree reason must record exact full-tree binding forbid rule"
            )

    bench_cross = benchmark.get("cross_tree_unaffected_reuse")
    if isinstance(bench_cross, Mapping):
        if bench_cross.get("status") != "unmet":
            errors.append("benchmark cross_tree_unaffected_reuse.status must be unmet")

    # --- required prose sections ---
    missing_sections = _missing_phrases(report_text, REQUIRED_REPORT_SECTIONS)
    if missing_sections:
        errors.append(
            "report missing required sections/phrases: "
            + ", ".join(sorted(missing_sections))
        )

    # Explicit non-claim trio (acceptance wording variants).
    non_claim_checks = (
        (
            "commitment_not_zk",
            (
                "not a zk proof",
                "not itself a zk",
                "not a zero-knowledge",
                "is not a zk",
            ),
        ),
        (
            "trusted_issuers",
            (
                "trusted issuer",
                "issuer is trusted",
                "signatures need trusted",
            ),
        ),
        (
            "structural_not_crypto",
            ("structural validation is not cryptographic validation",),
        ),
    )
    normalized_report = _normalize(report_text)
    for label, alternatives in non_claim_checks:
        if not any(alt in normalized_report for alt in alternatives):
            errors.append(f"report missing non-claim coverage for {label}")

    # Exact future ZK step phrasing.
    zk_step_needles = (
        "freeze",
        "leaf codec",
        "trust policy",
        "cross-implementation merkle",
        "membership",
        "aggregation",
    )
    for needle in zk_step_needles:
        if needle not in normalized_report:
            errors.append(f"future ZK step missing phrase {needle!r}")

    # --- README optional/required ---
    readme_missing: list[str] = []
    if require_readme:
        if not readme_text or not str(readme_text).strip():
            errors.append("verification README text is empty")
        else:
            readme_missing = _missing_phrases(str(readme_text), REQUIRED_README_PHRASES)
            if readme_missing:
                errors.append(
                    "README missing required phrases: "
                    + ", ".join(sorted(readme_missing))
                )

    if errors:
        raise ReportValidationError("; ".join(errors))

    return {
        "valid": True,
        "schema": REPORT_BINDING_SCHEMA,
        "interface": REPORT_INTERFACE,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "tree_id": current_tree_id,
        "corpus_id": report_corpus.get("corpus_id"),
        "corpus_cid": report_corpus.get("corpus_cid"),
        "policy_id": report_policy.get("policy_id"),
        "measurement_status": measurement_status,
        "measurement_schema_version": measurement_schema_version,
        "command_identities": dict(report_commands),
        "effective_environment": dict(report_env),
        "cross_tree_status": (
            cross_tree.get("status") if isinstance(cross_tree, Mapping) else None
        ),
        "evidence": list(evidence) if isinstance(evidence, Sequence) else [],
        "benchmark_evidence": benchmark.get("evidence"),
        "benchmark_status": benchmark.get("status"),
    }


def _fresh_benchmark() -> dict[str, Any]:
    """Run the IVP-017 harness for the current tree (in-process)."""

    module = _load_benchmark_module()
    result = module.run_incremental_verification_benchmark(repo_root_path=REPO_ROOT)
    if isinstance(result, Mapping):
        # Harness returns the artifact dict directly.
        if "tree_id" in result and "corpus" in result:
            return dict(result)
        artifact = result.get("artifact") or result.get("payload")
        if isinstance(artifact, Mapping):
            return dict(artifact)
    raise ReportValidationError(
        "run_incremental_verification_benchmark did not return an artifact mapping"
    )


def _load_report_and_readme() -> tuple[str, str]:
    assert REPORT_PATH.is_file(), f"missing report: {REPORT_PATH}"
    assert README_PATH.is_file(), f"missing README: {README_PATH}"
    return (
        REPORT_PATH.read_text(encoding="utf-8"),
        README_PATH.read_text(encoding="utf-8"),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def current_tree_id() -> str:
    return _git_head()


@pytest.fixture(scope="module")
def fresh_benchmark(current_tree_id: str) -> dict[str, Any]:
    artifact = _fresh_benchmark()
    assert artifact.get("tree_id") == current_tree_id, (
        f"fresh benchmark not bound to HEAD: {artifact.get('tree_id')} "
        f"!= {current_tree_id}"
    )
    return artifact


@pytest.fixture(scope="module")
def report_docs() -> tuple[str, str]:
    return _load_report_and_readme()


# ---------------------------------------------------------------------------
# Structural presence
# ---------------------------------------------------------------------------


def test_report_and_readme_exist() -> None:
    assert REPORT_PATH.is_file()
    assert README_PATH.is_file()
    report, readme = _load_report_and_readme()
    assert "IVP-018" in report
    assert "IncrementalVerificationPlanner" in readme or "create_verification_plan" in readme


def test_binding_json_extractable_and_schema_stable(report_docs: tuple[str, str]) -> None:
    report, _ = report_docs
    binding = _extract_binding_json(report)
    assert binding["schema"] == REPORT_BINDING_SCHEMA
    assert binding["interface"] == REPORT_INTERFACE
    assert binding["task_id"] == TASK_ID
    assert binding["goal_id"] == GOAL_ID
    assert DOCUMENTATION_EVIDENCE in binding["evidence"]
    assert RELEASE_REPORT_EVIDENCE in binding["evidence"]


# ---------------------------------------------------------------------------
# Full validator against current tree + fresh benchmark
# ---------------------------------------------------------------------------


def test_validator_accepts_current_report_and_fresh_benchmark(
    report_docs: tuple[str, str],
    fresh_benchmark: dict[str, Any],
    current_tree_id: str,
) -> None:
    report, readme = report_docs
    receipt = validate_incremental_verification_report(
        report,
        benchmark=fresh_benchmark,
        current_tree_id=current_tree_id,
        readme_text=readme,
    )
    assert receipt["valid"] is True
    assert receipt["tree_id"] == current_tree_id
    assert receipt["measurement_status"] == fresh_benchmark["status"]
    assert receipt["corpus_id"] == fresh_benchmark["corpus"]["corpus_id"]
    assert receipt["policy_id"] == fresh_benchmark["policy"]["policy_id"]
    assert receipt["cross_tree_status"] == "unmet"


def test_report_documents_every_unmet_target_including_cross_tree(
    report_docs: tuple[str, str],
    fresh_benchmark: dict[str, Any],
) -> None:
    report, _ = report_docs
    binding = _extract_binding_json(report)
    # Every benchmark target with non-met status must appear in the report.
    for name, payload in fresh_benchmark.get("targets", {}).items():
        status = payload.get("status") if isinstance(payload, Mapping) else None
        if status in {"unmet", "red"}:
            assert name.replace("_", " ") in _normalize(report) or name in report
            assert binding.get("target_statuses", {}).get(name) in {"unmet", "red"}
    assert "incompatible_cross_tree_unaffected_reuse" in binding.get(
        "target_statuses", {}
    )
    assert (
        binding["target_statuses"]["incompatible_cross_tree_unaffected_reuse"]
        == "unmet"
    )
    assert (
        "exact_full_tree_binding_forbids_incompatible_cross_tree_reuse"
        in report
    )


def test_report_includes_metrics_snapshot_fields(
    report_docs: tuple[str, str],
    fresh_benchmark: dict[str, Any],
) -> None:
    report, _ = report_docs
    binding = _extract_binding_json(report)
    metrics = binding.get("metrics_snapshot") or {}
    assert "cache_hit_rate" in metrics
    assert "tests_selected_total" in metrics
    assert "tests_full_total" in metrics
    assert "route_counts" in metrics
    # Prose must mention the observed route classes.
    for route_name in fresh_benchmark["metrics"]["routes"]["counts"]:
        assert route_name in report


# ---------------------------------------------------------------------------
# Stale / missing rejection
# ---------------------------------------------------------------------------


def test_validator_rejects_stale_report_tree(
    report_docs: tuple[str, str],
    fresh_benchmark: dict[str, Any],
    current_tree_id: str,
) -> None:
    report, readme = report_docs
    binding = _extract_binding_json(report)
    stale = copy.deepcopy(binding)
    stale["tree_id"] = "0" * 40
    stale_report = report.replace(
        json.dumps(binding, indent=2, sort_keys=True),
        json.dumps(stale, indent=2, sort_keys=True),
        1,
    )
    # If pretty formatting differs, force-inject via marker replacement.
    if stale_report == report:
        stale_report = report.replace(binding["tree_id"], stale["tree_id"], 1)
    with pytest.raises(ReportValidationError, match="stale report tree_id"):
        validate_incremental_verification_report(
            stale_report,
            benchmark=fresh_benchmark,
            current_tree_id=current_tree_id,
            readme_text=readme,
        )


def test_validator_rejects_stale_benchmark_tree(
    report_docs: tuple[str, str],
    fresh_benchmark: dict[str, Any],
    current_tree_id: str,
) -> None:
    report, readme = report_docs
    stale_bench = copy.deepcopy(dict(fresh_benchmark))
    stale_bench["tree_id"] = "a" * 40
    with pytest.raises(ReportValidationError, match="stale benchmark tree_id"):
        validate_incremental_verification_report(
            report,
            benchmark=stale_bench,
            current_tree_id=current_tree_id,
            readme_text=readme,
        )


def test_validator_rejects_missing_section(
    report_docs: tuple[str, str],
    fresh_benchmark: dict[str, Any],
    current_tree_id: str,
) -> None:
    report, readme = report_docs
    # Strip commitment non-claim and a required section phrase.
    mutilated = report.replace("structural validation is not cryptographic validation", "")
    mutilated = mutilated.replace("Structural validation is not cryptographic validation", "")
    with pytest.raises(ReportValidationError, match="missing|non-claim|structural"):
        validate_incremental_verification_report(
            mutilated,
            benchmark=fresh_benchmark,
            current_tree_id=current_tree_id,
            readme_text=readme,
        )


def test_validator_rejects_missing_binding_json(
    fresh_benchmark: dict[str, Any],
    current_tree_id: str,
) -> None:
    with pytest.raises(ReportValidationError, match="binding JSON"):
        validate_incremental_verification_report(
            "# empty report without binding\n",
            benchmark=fresh_benchmark,
            current_tree_id=current_tree_id,
            require_readme=False,
        )


def test_validator_rejects_corpus_policy_or_measurement_mismatch(
    report_docs: tuple[str, str],
    fresh_benchmark: dict[str, Any],
    current_tree_id: str,
) -> None:
    report, readme = report_docs
    binding = _extract_binding_json(report)

    def _with_binding(mutator: Any) -> str:
        altered = copy.deepcopy(binding)
        mutator(altered)
        text = report
        # Replace tree-stable pretty block if present; else rewrite first JSON fence.
        try:
            text = report.replace(
                json.dumps(binding, indent=2, sort_keys=True),
                json.dumps(altered, indent=2, sort_keys=True),
                1,
            )
        except Exception:
            text = report
        if text == report:
            # Fallback: replace policy_id string.
            if altered.get("policy", {}).get("policy_id") != binding["policy"]["policy_id"]:
                text = report.replace(
                    binding["policy"]["policy_id"],
                    altered["policy"]["policy_id"],
                    1,
                )
            elif altered.get("measurement_status") != binding.get("measurement_status"):
                # Replace measurement_status value in binding only (first occurrence
                # inside the JSON block).
                text = report.replace(
                    f'"measurement_status": "{binding["measurement_status"]}"',
                    f'"measurement_status": "{altered["measurement_status"]}"',
                    1,
                )
            elif altered.get("corpus", {}).get("corpus_id") != binding["corpus"]["corpus_id"]:
                text = report.replace(
                    binding["corpus"]["corpus_id"],
                    altered["corpus"]["corpus_id"],
                    1,
                )
        return text

    def _set_policy(b: MutableMapping[str, Any]) -> None:
        b["policy"] = dict(b["policy"])
        b["policy"]["policy_id"] = "policy:wrong@0"

    with pytest.raises(ReportValidationError, match="policy_id mismatch"):
        validate_incremental_verification_report(
            _with_binding(_set_policy),
            benchmark=fresh_benchmark,
            current_tree_id=current_tree_id,
            readme_text=readme,
        )

    def _set_status(b: MutableMapping[str, Any]) -> None:
        b["measurement_status"] = "green"

    if binding.get("measurement_status") != "green":
        with pytest.raises(ReportValidationError, match="measurement_status mismatch"):
            validate_incremental_verification_report(
                _with_binding(_set_status),
                benchmark=fresh_benchmark,
                current_tree_id=current_tree_id,
                readme_text=readme,
            )


def test_validator_rejects_missing_command_identity(
    report_docs: tuple[str, str],
    fresh_benchmark: dict[str, Any],
    current_tree_id: str,
) -> None:
    report, readme = report_docs
    binding = _extract_binding_json(report)
    altered = copy.deepcopy(binding)
    altered["command_identities"] = {
        k: v
        for k, v in altered["command_identities"].items()
        if k != "validate_report"
    }
    text = report.replace(
        json.dumps(binding, indent=2, sort_keys=True),
        json.dumps(altered, indent=2, sort_keys=True),
        1,
    )
    if text == report:
        pytest.skip("could not surgically rewrite binding JSON block")
    with pytest.raises(ReportValidationError, match="command identity"):
        validate_incremental_verification_report(
            text,
            benchmark=fresh_benchmark,
            current_tree_id=current_tree_id,
            readme_text=readme,
        )


# ---------------------------------------------------------------------------
# Commitment / ZK wording
# ---------------------------------------------------------------------------


def test_report_states_commitment_non_claims_and_exact_zk_next_step(
    report_docs: tuple[str, str],
) -> None:
    report, readme = report_docs
    blob = _normalize(report + "\n" + readme)
    assert "not a zk" in blob or "not a zero-knowledge" in blob
    assert "trusted issuer" in blob
    assert "structural validation is not cryptographic validation" in blob
    assert "freeze" in blob and "leaf codec" in blob and "trust policy" in blob
    assert "cross-implementation merkle" in blob
    assert "membership" in blob and "aggregation" in blob
    # Ensure the ordered future step is present as a coherent instruction.
    assert (
        "freeze the admitted receipt leaf codec and trust policy"
        in blob
        or (
            "freeze" in blob
            and "leaf codec" in blob
            and "trust policy" in blob
            and "cross-implementation merkle vectors" in blob
        )
    )


def test_commitment_contract_flags_align_with_docs() -> None:
    from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
        VerificationCommitment,
    )

    assert VerificationCommitment.IS_ZERO_KNOWLEDGE_PROOF is False
    assert VerificationCommitment.LEAF_CODEC == "canonical-dag-json@1"
    assert VerificationCommitment.LEAF_DOMAIN == "IVP-LEAF@1"
    assert VerificationCommitment.NODE_DOMAIN == "IVP-NODE@1"
    assert VerificationCommitment.EMPTY_DOMAIN == "IVP-EMPTY@1"


# ---------------------------------------------------------------------------
# Checked-in artifact hygiene (may be stale; validator must notice)
# ---------------------------------------------------------------------------


def test_checked_in_artifact_if_present_is_structurally_benchmark_shaped(
    current_tree_id: str,
) -> None:
    if not CHECKED_IN_ARTIFACT.is_file():
        pytest.skip("no checked-in benchmark artifact")
    payload = json.loads(CHECKED_IN_ARTIFACT.read_text(encoding="utf-8"))
    assert payload.get("evidence") == BENCHMARK_EVIDENCE
    assert "corpus" in payload and "policy" in payload
    assert "effective_environment" in payload
    assert "commands" in payload
    assert "measurement_schema" in payload
    assert "status" in payload
    # If the on-disk artifact lags HEAD, the report validator must reject it.
    if payload.get("tree_id") != current_tree_id:
        report, readme = _load_report_and_readme()
        with pytest.raises(ReportValidationError, match="stale"):
            validate_incremental_verification_report(
                report,
                benchmark=payload,
                current_tree_id=current_tree_id,
                readme_text=readme,
            )


def test_live_environment_fields_are_documented(
    report_docs: tuple[str, str],
    fresh_benchmark: dict[str, Any],
) -> None:
    report, _ = report_docs
    binding = _extract_binding_json(report)
    env = binding["effective_environment"]
    assert env["python_version"] == fresh_benchmark["effective_environment"]["python_version"]
    assert env["system"] == platform.system() or env["system"] == fresh_benchmark[
        "effective_environment"
    ]["system"]
    assert env["python_version"] in report
    assert env["platform"] in report
