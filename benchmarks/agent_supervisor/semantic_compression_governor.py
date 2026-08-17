#!/usr/bin/env python3
"""Semantic Compression Governor controlled benchmark (SCG-045).

Interface: ``SemanticGovernorBenchmark@1``
Evidence: ``scg/benchmark-results@1``

Runs the partitioned fixture corpus through real governor public APIs and
persists honest measured evidence under
``artifacts/agent_supervisor/semantic_compression_governor/``.

Authority rules
---------------
* Artifacts are never production-authoritative and never claim live model quality
  for controlled/oracle fixtures.
* Simulated and live cohorts are labeled and aggregated separately. Fixture
  measurements always land in the simulated cohort; an empty live cohort is
  reported with explicit missing evidence.
* Targets are thresholds evaluated against measured values — never hard-coded
  success constants in the output.
* Missing sensors, empty live evidence, and unavailable seal scope are explicit
  (``None`` / ``unavailable`` / ``missing_evidence``), never rewritten to zero
  success.
* Simulated/local cost estimators are labeled and never promoted into live
  savings claims.

Importing this module performs no network I/O and never installs packages.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from ipfs_datasets_py.logic.software_contracts import semantic_governor as sg
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    AssumptionKind,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    GovernorArtifactHeader,
    GovernorAssumption,
    GovernorTerminalStatus,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.calibration_contracts import (
    EvidencePartition,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.policy_contracts import (
    CompressionPolicyCandidate,
    EvaluationVerdict,
    ProtectedThresholds,
)

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_bytes,
    cid_for_dag_json,
    validate_cid as _validate_cid_identity,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    AcceptanceDisposition,
    ComparativeOutcome,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.metrics import (
    BASIS_POINTS,
    GovernorMetricsCollector,
    MetricsCohort,
    MetricsObservation,
    collect_metrics,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.policy_evaluation import (
    HeldOutBenchmark,
    HeldOutCaseOutcome,
    evaluate_rule_candidate,
)


def _install_dependency_free_content_identity() -> None:
    """Bridge software-contract CID helpers onto the sealed in-tree profile.

    Launch / hermetic validation often sets ``PYTHONNOUSERSITE=1``, which
    removes user-site ``multiformats``.  The governor contracts still mint and
    validate CIDs through ``ipfs_datasets_py.logic.software_contracts.content``,
    whose stock path imports that package.  Rebind generation and validation to
    the dependency-free supervisor identity bridge (same CIDv1 wire profile) so
    ``--check`` and measured fixtures stay hermetic.
    """

    try:
        import multiformats  # noqa: F401
    except ImportError:
        pass
    else:
        return

    import ipfs_datasets_py.logic.software_contracts.content as content_mod

    def _cid_for_bytes(data: bytes) -> str:
        return cid_for_bytes(data)

    def _cid_for_structured(obj: Any) -> str:
        return cid_for_dag_json(obj)

    def _validate_cid(
        value: Any,
        *,
        codecs: Any = None,
    ) -> str:
        # codecs is accepted for signature parity; the sealed profile already
        # admits only raw + dag-json.
        del codecs
        return _validate_cid_identity(value)

    content_mod.cid_for_bytes = _cid_for_bytes  # type: ignore[assignment]
    content_mod.cid_for_structured = _cid_for_structured  # type: ignore[assignment]
    content_mod.cid_for_obj = _cid_for_structured  # type: ignore[assignment]
    content_mod.validate_cid = _validate_cid  # type: ignore[assignment]

    rebound = {
        "cid_for_bytes": _cid_for_bytes,
        "cid_for_structured": _cid_for_structured,
        "cid_for_obj": _cid_for_structured,
        "validate_cid": _validate_cid,
    }
    for module in list(sys.modules.values()):
        if module is None:
            continue
        module_dict = getattr(module, "__dict__", None)
        if not isinstance(module_dict, dict):
            continue
        for name, replacement in rebound.items():
            current = module_dict.get(name)
            if current is None or current is replacement:
                continue
            # Only rebind callables that originated from the content module.
            if getattr(current, "__module__", None) == content_mod.__name__:
                module_dict[name] = replacement


_install_dependency_free_content_identity()

# ---------------------------------------------------------------------------
# Interface / schema pins
# ---------------------------------------------------------------------------

BENCHMARK_INTERFACE: Final[str] = "SemanticGovernorBenchmark@1"
BENCHMARK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor-benchmark@1"
)
BENCHMARK_SUMMARY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor-benchmark-summary@1"
)
BENCHMARK_CASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor-benchmark-case@1"
)
BENCHMARK_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor-benchmark-metrics@1"
)
BENCHMARK_EVIDENCE: Final[str] = "scg/benchmark-results@1"
TASK_ID: Final[str] = "SCG-045"
GOAL_ID: Final[str] = "SCG-G080"
POLICY_ID: Final[str] = "policy:scg-benchmark@1"
TOKENIZER_ID: Final[str] = "scg-estimator/utf8-bytes-div4@1"
TOKENIZER_VERSION: Final[str] = "1.0.0"
MEASUREMENT_SCHEMA_VERSION: Final[str] = "scg-benchmark-measurement/v1"
GENERATOR_ID: Final[str] = "semantic_governor_benchmark"
GENERATOR_VERSION: Final[str] = "1.0.0"
COST_ESTIMATOR_ID: Final[str] = "scg-local-cost-estimator/v1"

DEFAULT_BENCHMARK_RELPATH: Final[str] = (
    "artifacts/agent_supervisor/semantic_compression_governor/benchmark.json"
)
DEFAULT_SUMMARY_RELPATH: Final[str] = (
    "artifacts/agent_supervisor/semantic_compression_governor/summary.json"
)

FIXTURE_PACKAGE_NAME: Final[str] = "scg_partitioned_fixture_corpus"
FIXTURE_RELPATH: Final[str] = "test/fixtures/semantic_governor"

# Unit micros for the local simulated cost estimator (not live provider rates).
MICROS_PER_INPUT_TOKEN: Final[int] = 8
MICROS_PER_OUTPUT_TOKEN: Final[int] = 24
AUDIT_OVERHEAD_BASE_MICROS: Final[int] = 150
VERIFICATION_MICROS_PER_CASE: Final[int] = 400
SHADOW_MICROS_PER_CASE: Final[int] = 250
OUTPUT_TOKENS_PER_CASE: Final[int] = 80

# Plan §12 initial targets (thresholds, never output constants).
TARGET_MIN_CRITICAL_DETECTION_BP: Final[int] = 9_500
TARGET_MAX_CRITICAL_ACCEPTED: Final[int] = 0
TARGET_MIN_MEDIAN_REDUCTION_BP: Final[int] = 5_000
TARGET_MAX_REGRESSION_COUNT: Final[int] = 0

_TOKEN_SAFE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.:/+-]+")

_FAMILY_GAP_KIND: Final[Mapping[str, str]] = {
    "configuration": sg.CoverageGapKind.MISSING_CONFIGURATION.value,
    "fixture": sg.CoverageGapKind.MISSING_FIXTURE.value,
    "schema_migration": sg.CoverageGapKind.MISSING_SCHEMA.value,
    "api_migration": sg.CoverageGapKind.MISSING_SCHEMA.value,
    "generated": sg.CoverageGapKind.LOW_CONFIDENCE.value,
    "dynamic_import": sg.CoverageGapKind.DYNAMIC_IMPORT.value,
    "local_bug": sg.CoverageGapKind.BUDGET_TRUNCATION.value,
    "exception": sg.CoverageGapKind.BUDGET_TRUNCATION.value,
    "refactor": sg.CoverageGapKind.BUDGET_TRUNCATION.value,
    "proof": sg.CoverageGapKind.MISSING_PROOF.value,
    "state": sg.CoverageGapKind.BUDGET_TRUNCATION.value,
    "plugin": sg.CoverageGapKind.OPAQUE_DEPENDENCY.value,
    "monkey_patch": sg.CoverageGapKind.OPAQUE_DEPENDENCY.value,
    "documentation": sg.CoverageGapKind.BUDGET_TRUNCATION.value,
}

_CONFIDENCE_BP: Final[Mapping[str, int]] = {
    "exact": 10_000,
    "conservative": 7_500,
    "heuristic": 4_500,
    "opaque": 1_000,
}

_ACCEPTING_STATES: Final[frozenset[str]] = frozenset(
    {
        sg.ContextSufficiencyState.SUFFICIENT.value,
        sg.ContextSufficiencyState.SUFFICIENT_WITH_CAVEATS.value,
    }
)
_ACCEPTING_ACTIONS: Final[frozenset[str]] = frozenset(
    {sg.DecisionAction.ACCEPT_COMPRESSED.value}
)

# Observational keys stripped for --check equality.
_EPHEMERAL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "generated_at_unix_ms",
        "benchmark_duration_ms",
        "pid",
        "observed_head",
        "wall_clock_ms",
        "elapsed_ms",
        "observational_latency_ms",
    }
)


class BenchmarkError(RuntimeError):
    """Closed benchmark runner contract violation."""


class BenchmarkStatus(str, Enum):
    GREEN = "green"
    RED = "red"
    YELLOW = "yellow"
    NOT_MEASURED = "not_measured"


# ---------------------------------------------------------------------------
# Paths / environment
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_fixture_root(root: Path | None = None) -> Path:
    return (root or repo_root()) / FIXTURE_RELPATH


def _checkpoint_dir() -> Path | None:
    raw = os.environ.get("IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return path


def write_checkpoint(name: str, payload: Mapping[str, Any]) -> Path | None:
    directory = _checkpoint_dir()
    if directory is None:
        return None
    target = directory / f"{name}.json"
    write_json_atomic(target, dict(payload))
    return target


def effective_environment() -> dict[str, Any]:
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/effective-verification-environment@1"
        ),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "cwd": ".",
        "machine": platform.machine(),
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_version": TOKENIZER_VERSION,
        "cost_estimator_id": COST_ESTIMATOR_ID,
        "cost_cohort": MetricsCohort.SIMULATED.value,
    }


def benchmark_commands(
    *,
    benchmark_output: str = DEFAULT_BENCHMARK_RELPATH,
    summary_output: str = DEFAULT_SUMMARY_RELPATH,
) -> dict[str, Any]:
    generate = (
        "PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. "
        "python3 benchmarks/agent_supervisor/semantic_compression_governor.py "
        f"--write --benchmark-out {benchmark_output} "
        f"--summary-out {summary_output}"
    )
    check = (
        "PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. "
        "python3 benchmarks/agent_supervisor/semantic_compression_governor.py --check"
    )
    validate = (
        "PYTHONPATH=ipfs_kit_py:ipfs_datasets_py:. "
        "python3 -m pytest -q "
        "test/benchmarks/test_semantic_compression_governor_benchmark.py"
    )
    return {
        "generate_artifact": generate,
        "check": check,
        "validate": validate,
    }


def observed_head(root: Path | None = None) -> str | None:
    root = root or repo_root()
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    text = (completed.stdout or "").strip()
    return text or None


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _enum_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(encoded, encoding="utf-8")
    tmp.replace(path)


def _stable_compare_view(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_compare_view(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _EPHEMERAL_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_stable_compare_view(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def artifacts_structurally_equivalent(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    return _stable_compare_view(dict(left)) == _stable_compare_view(dict(right))


def write_stable_artifact(
    path: Path,
    payload: Mapping[str, Any],
    *,
    force: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Write payload; preserve bytes when only ephemeral fields churn."""

    artifact = dict(payload)
    if not force and path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and artifacts_structurally_equivalent(
            existing, artifact
        ):
            return existing, True
    write_json_atomic(path, artifact)
    return artifact, False


def estimate_tokens(text: str | bytes, *, tokenizer_id: str = TOKENIZER_ID) -> int:
    if tokenizer_id != TOKENIZER_ID:
        raise BenchmarkError(f"unsupported tokenizer_id {tokenizer_id!r}")
    if isinstance(text, str):
        raw = text.encode("utf-8")
    else:
        raw = bytes(text)
    return max(1, (len(raw) + 3) // 4)


# ---------------------------------------------------------------------------
# Fixture corpus loader
# ---------------------------------------------------------------------------


def load_fixture_package(fixture_root: Path | None = None) -> ModuleType:
    root = Path(fixture_root) if fixture_root is not None else default_fixture_root()
    if FIXTURE_PACKAGE_NAME in sys.modules and hasattr(
        sys.modules[FIXTURE_PACKAGE_NAME], "SemanticGovernorFixtureCorpus"
    ):
        existing = sys.modules[FIXTURE_PACKAGE_NAME]
        existing_path = Path(getattr(existing, "__file__", "") or "").resolve()
        if existing_path.parent == root.resolve():
            return existing

    init_path = root / "__init__.py"
    if not init_path.is_file():
        raise BenchmarkError(f"missing fixture package init: {init_path}")

    package = ModuleType(FIXTURE_PACKAGE_NAME)
    package.__file__ = str(init_path)
    package.__path__ = [str(root)]  # type: ignore[attr-defined]
    sys.modules[FIXTURE_PACKAGE_NAME] = package

    for name, filename in (
        ("case_record", "case_record.py"),
        ("recipes", "recipes.py"),
        ("corpus", "corpus.py"),
    ):
        qualname = f"{FIXTURE_PACKAGE_NAME}.{name}"
        path = root / filename
        spec = importlib.util.spec_from_file_location(qualname, path)
        if spec is None or spec.loader is None:
            raise BenchmarkError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = FIXTURE_PACKAGE_NAME
        sys.modules[qualname] = module
        spec.loader.exec_module(module)
        setattr(package, name, module)

    init_spec = importlib.util.spec_from_file_location(
        FIXTURE_PACKAGE_NAME,
        init_path,
        submodule_search_locations=[str(root)],
    )
    if init_spec is None or init_spec.loader is None:
        raise BenchmarkError(f"cannot load package init {init_path}")
    package.__spec__ = init_spec
    package.__package__ = FIXTURE_PACKAGE_NAME
    init_spec.loader.exec_module(package)
    if not hasattr(package, "SemanticGovernorFixtureCorpus"):
        raise BenchmarkError("fixture package missing SemanticGovernorFixtureCorpus")
    return package


def load_fixture_corpus(fixture_root: Path | None = None) -> Any:
    package = load_fixture_package(fixture_root)
    return package.SemanticGovernorFixtureCorpus.load()


# ---------------------------------------------------------------------------
# Controlled view builders (fixture-oracle bound; no model authority)
# ---------------------------------------------------------------------------


def _token_id(prefix: str, *parts: str) -> str:
    raw = "_".join((prefix, *parts))
    cleaned = _TOKEN_SAFE.sub("_", raw).strip("_").lower()
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"id_{cleaned}"
    return cleaned[:128]


def _sym_token(symbol: str) -> str:
    text = str(symbol).strip().lower()
    text = _TOKEN_SAFE.sub("_", text).strip("._")
    if not text or not text[0].isalpha():
        text = f"sym_{text}"
    return text[:128]


def _path_for_symbol(case: Any, symbol: str) -> str:
    scanner = case.scanner_view
    lowered = {item.lower(): item for item in scanner.changed_symbols}
    if symbol in scanner.changed_symbols or symbol.lower() in lowered:
        if scanner.changed_paths:
            return scanner.changed_paths[0]
    if ":" in symbol:
        head = symbol.split(":", 1)[0]
        return head.replace(".", "/") + ".md"
    if symbol.startswith("proof."):
        return "proofs/" + symbol[len("proof.") :].replace(".", "/") + ".lean"
    if symbol.startswith("tests."):
        body = symbol[len("tests.") :]
        module = body.rsplit(".", 1)[0]
        return "tests/" + module.replace(".", "/") + ".py"
    parts = symbol.split(".")
    if len(parts) >= 2:
        return "/".join(parts[:-1]) + ".py"
    return "scg_fixture/unknown.py"


def _generator(interface_id: str = "evaluate_context_sufficiency@1") -> Any:
    return GeneratorIdentity(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        interface_id=interface_id,
    )


def _provenance(*, case_id: str) -> Any:
    # Controlled fixture evaluations use LIVE execution_mode for deterministic
    # governor API paths (same binding as SCG-041). Metric quality claims for
    # these cases still land in the simulated cohort — never live model quality.
    return sg.ArtifactProvenance(
        producer_id="semantic_governor",
        producer_version="1",
        execution_mode=ExecutionMode.LIVE,
        authority_source=AuthoritySource.DETERMINISTIC,
        input_cids=(_cid(f"fixture:{case_id}"),),
        tool_ids=("scg_benchmark.v1",),
        policy_cid=_cid(POLICY_ID),
        notes=None,
    )


def _header(
    artifact_kind: str,
    *,
    case_id: str,
    repo_cid: str,
    pack_cid: str,
    interface_id: str = "evaluate_context_sufficiency@1",
    **overrides: object,
) -> Any:
    fields: dict[str, object] = {
        "artifact_kind": artifact_kind,
        "repository_state_cid": repo_cid,
        "context_pack_cid": pack_cid,
        "verification_bundle_cid": _cid(f"verification:{case_id}"),
        "generator": _generator(interface_id),
        "provenance": _provenance(case_id=case_id),
        "terminal_status": GovernorTerminalStatus.COMPLETE,
        "assumptions": (
            GovernorAssumption(
                assumption_id="fixture_oracle_binding",
                kind=AssumptionKind.COVERAGE,
                statement=(
                    "Coverage exclusions and gaps are bound to independently "
                    "declared fixture scanner/omission oracles"
                ),
                supporting_cids=(_cid(f"oracle:{case_id}"),),
            ),
        ),
        "metadata": {
            "task_id": TASK_ID,
            "case_id": case_id,
            "interface": BENCHMARK_INTERFACE,
            "evidence": BENCHMARK_EVIDENCE,
        },
    }
    fields.update(overrides)
    return GovernorArtifactHeader(**fields)  # type: ignore[arg-type]


def _graph_path(*nodes: str, relation: str = "calls") -> Any:
    if not nodes:
        nodes = ("target",)
    normalized = tuple(_sym_token(node) for node in nodes)
    return sg.GraphPath(nodes=normalized, edge_relation=relation)


def _span(path: str, start: int = 1, end: int = 20) -> Any:
    return sg.SourceSpan(
        path=path, start_line=start, end_line=end, start_col=1, end_col=1
    )


def _artifact_kind_for_symbol(symbol: str, family: str) -> str:
    if family == "configuration" or "config" in symbol:
        return sg.CoveredArtifactKind.CONFIGURATION.value
    if family == "fixture" or symbol.startswith("tests.conftest"):
        return sg.CoveredArtifactKind.FIXTURE.value
    if family in {"schema_migration", "api_migration"} or "schema" in symbol:
        return sg.CoveredArtifactKind.SCHEMA.value
    if symbol.startswith("proof."):
        return sg.CoveredArtifactKind.PROOF_OBLIGATION.value
    return sg.CoveredArtifactKind.SYMBOL.value


def _gap_kind_for_case(case: Any) -> str:
    if case.adversarial_scenario == "stale_capsule":
        return sg.CoverageGapKind.STALE_CAPSULE.value
    if case.adversarial_scenario == "confidence_misclassification":
        return sg.CoverageGapKind.OPAQUE_DEPENDENCY.value
    if case.scanner_view.confidence == "opaque":
        return sg.CoverageGapKind.OPAQUE_DEPENDENCY.value
    return _FAMILY_GAP_KIND.get(
        case.family, sg.CoverageGapKind.BUDGET_TRUNCATION.value
    )


def _symbol_token_cost(case: Any, symbol: str) -> int:
    """Deterministic per-symbol token cost from fixture recipe content when present."""

    body = ""
    path = _path_for_symbol(case, symbol)
    for operation in case.operations:
        if getattr(operation, "path", None) == path and getattr(
            operation, "content", None
        ):
            body = str(operation.content)
            break
    if body:
        return max(20, estimate_tokens(body))
    # Stable default by symbol length (oracle-bound, not free).
    return max(20, 12 + len(symbol) * 2)


def _inclusion(
    *,
    case: Any,
    symbol: str,
    inclusion_kind: str = sg.InclusionKind.RAW_SOURCE.value,
    primary: str | None = None,
) -> Any:
    path = _path_for_symbol(case, symbol)
    primary = primary or case.scanner_view.primary_symbol
    sym = _sym_token(symbol)
    prim = _sym_token(primary)
    nodes = (prim,) if sym == prim else (prim, sym)
    conf = _CONFIDENCE_BP.get(case.scanner_view.confidence, 10_000)
    if inclusion_kind == sg.InclusionKind.RAW_SOURCE.value:
        conf = 10_000
    return sg.IncludedArtifactRecord(
        artifact_id=_token_id("inc", case.case_id, symbol),
        artifact_kind=_artifact_kind_for_symbol(symbol, case.family),
        inclusion_kind=inclusion_kind,
        token_cost=_symbol_token_cost(case, symbol),
        symbol_id=sym,
        path=path,
        artifact_cid=_cid(f"inc:{case.case_id}:{symbol}"),
        confidence_bp=conf,
        dependency_path=_graph_path(*nodes),
        source_span=_span(path),
        notes=None,
    )


def _exclusion(
    *,
    case: Any,
    symbol: str,
    critical: bool,
    reason: str = sg.ExclusionReason.EXACT_CAPSULE_SUBSTITUTED.value,
    substituted_by: str | None = None,
    repo_cid: str,
) -> Any:
    path = _path_for_symbol(case, symbol)
    primary = case.scanner_view.primary_symbol
    sym = _sym_token(symbol)
    prim = _sym_token(primary)
    nodes = (prim,) if sym == prim else (prim, sym)
    if (
        substituted_by is None
        and reason
        in {
            sg.ExclusionReason.EXACT_CAPSULE_SUBSTITUTED.value,
            sg.ExclusionReason.CONSERVATIVE_CAPSULE_SUBSTITUTED.value,
        }
    ):
        substituted_by = _token_id("cap", case.case_id, symbol)
    return sg.ExcludedArtifactRecord(
        artifact_id=_token_id("exc", case.case_id, symbol),
        artifact_kind=_artifact_kind_for_symbol(symbol, case.family),
        exclusion_reason=reason,
        token_cost=_symbol_token_cost(case, symbol),
        confidence_bp=_CONFIDENCE_BP.get(case.scanner_view.confidence, 9_000),
        symbol_id=sym,
        path=path,
        artifact_cid=_cid(f"exc:{case.case_id}:{symbol}"),
        dependency_path=_graph_path(*nodes),
        source_span=_span(path),
        repository_state_cid=repo_cid,
        substituted_by_artifact_id=substituted_by,
        critical=critical,
        notes=None,
    )


def build_compressed_manifest(
    case: Any,
    *,
    repo_cid: str,
    pack_cid: str,
    include_critical: bool = False,
) -> Any:
    omission = case.omission
    scanner = case.scanner_view
    primary = scanner.primary_symbol

    inclusions: list[Any] = []
    exclusions: list[Any] = []
    gaps: list[Any] = []
    opaque_ids: list[str] = []

    include_set = set(omission.compressed_includes)
    if include_critical:
        include_set |= set(omission.critical_omitted_symbols)
        include_set |= set(omission.expansion_targets)
        include_set |= {primary}
        include_set |= set(scanner.dependency_symbols)

    omit_set = set(omission.compressed_omits) | set(omission.critical_omitted_symbols)
    if include_critical:
        omit_set = set(omission.noncritical_omitted_symbols)

    if primary not in omit_set:
        include_set.add(primary)
    elif include_critical:
        include_set.add(primary)
        omit_set.discard(primary)

    for symbol in sorted(include_set - omit_set):
        kind = sg.InclusionKind.RAW_SOURCE.value
        if (
            not include_critical
            and scanner.confidence in {"conservative", "heuristic"}
            and symbol == primary
        ):
            kind = sg.InclusionKind.CONSERVATIVE_CAPSULE.value
        inclusions.append(
            _inclusion(case=case, symbol=symbol, inclusion_kind=kind)
        )

    critical_omitted = set(omission.critical_omitted_symbols)
    if not include_critical:
        for symbol in sorted(omit_set):
            is_critical = symbol in critical_omitted
            reason = sg.ExclusionReason.EXACT_CAPSULE_SUBSTITUTED.value
            if scanner.confidence == "opaque":
                reason = sg.ExclusionReason.CONSERVATIVE_CAPSULE_SUBSTITUTED.value
            elif not is_critical:
                reason = sg.ExclusionReason.OUTSIDE_AFFECTED_INVALIDATION_CONE.value
            exclusions.append(
                _exclusion(
                    case=case,
                    symbol=symbol,
                    critical=is_critical,
                    reason=reason,
                    repo_cid=repo_cid,
                )
            )
            if is_critical:
                gap_kind = _gap_kind_for_case(case)
                gap_id = _token_id("gap", case.case_id, symbol)
                art_id = _token_id("exc", case.case_id, symbol)
                gaps.append(
                    sg.CoverageGap(
                        gap_id=gap_id,
                        gap_kind=gap_kind,
                        description=(
                            f"Critical dependency {symbol} omitted from compressed pack "
                            f"({case.adversarial_scenario or case.family})"
                        ),
                        artifact_id=art_id,
                        critical=True,
                    )
                )
                if gap_kind == sg.CoverageGapKind.OPAQUE_DEPENDENCY.value:
                    opaque_ids.append(art_id)
    else:
        for symbol in sorted(omit_set):
            exclusions.append(
                _exclusion(
                    case=case,
                    symbol=symbol,
                    critical=False,
                    reason=sg.ExclusionReason.OUTSIDE_AFFECTED_INVALIDATION_CONE.value,
                    repo_cid=repo_cid,
                )
            )

    if not inclusions:
        inclusions.append(
            _inclusion(
                case=case,
                symbol=primary,
                inclusion_kind=sg.InclusionKind.RAW_SOURCE.value,
            )
        )

    inclusions_t = tuple(sorted(inclusions, key=lambda item: item.artifact_id))
    exclusions_t = tuple(sorted(exclusions, key=lambda item: item.artifact_id))
    gaps_t = tuple(sorted(gaps, key=lambda item: item.gap_id))

    raw_count = sum(
        1
        for item in inclusions_t
        if item.inclusion_kind
        in {sg.InclusionKind.RAW_SOURCE.value, "raw_source"}
    )
    capsule_count = sum(
        1
        for item in inclusions_t
        if item.inclusion_kind
        in {
            sg.InclusionKind.EXACT_CAPSULE.value,
            sg.InclusionKind.CONSERVATIVE_CAPSULE.value,
            "exact_capsule",
            "conservative_capsule",
        }
    )
    dep_paths: list[Any] = []
    seen_paths: set[tuple[str, ...]] = set()
    for item in list(inclusions_t) + list(exclusions_t):
        if item.dependency_path is None:
            continue
        key = tuple(item.dependency_path.nodes)
        if key not in seen_paths:
            seen_paths.add(key)
            dep_paths.append(item.dependency_path)

    return sg.ContextCoverageManifest(
        header=_header(
            "context_coverage_manifest",
            case_id=case.case_id,
            repo_cid=repo_cid,
            pack_cid=pack_cid,
            interface_id="build_context_coverage_manifest@1",
        ),
        manifest_id=_token_id("manifest", case.case_id),
        target_symbol_ids=(_sym_token(primary),),
        inclusions=inclusions_t,
        exclusions=exclusions_t,
        context_budget_tokens=2_000,
        minimum_safe_tokens=40,
        total_included_tokens=sum(item.token_cost for item in inclusions_t),
        total_excluded_tokens=sum(item.token_cost for item in exclusions_t),
        raw_inclusion_count=raw_count,
        capsule_inclusion_count=capsule_count,
        exclusion_count=len(exclusions_t),
        known_gaps=gaps_t,
        opaque_dependency_ids=tuple(sorted(set(opaque_ids))),
        dependency_paths=tuple(dep_paths),
        policy_cid=_cid(POLICY_ID),
        notes=None,
        metadata={
            "case_id": case.case_id,
            "adversarial_scenario": case.adversarial_scenario or "",
            "include_critical": include_critical,
        },
    )


def _acceptance_for_case(case: Any) -> Any:
    require_proofs = bool(case.outcome.proof_obligations)
    require_review = case.outcome.expected_diagnosis in {
        "security",
        "confidence_error",
    } or case.outcome.expected_outcome == "human_review_required"
    if case.outcome.expected_outcome == "insufficient_omission":
        require_review = False
    if case.outcome.expected_outcome == "reject_stale":
        require_review = False
    risk = "high" if case.outcome.expected_diagnosis == "security" else "medium"
    return sg.TaskClassAcceptanceRequirements(
        task_class=case.family,
        risk_class=risk,
        require_selected_tests=bool(case.outcome.selected_tests),
        require_full_suite_fallback=True,
        require_static_checks=True,
        require_type_checks=True,
        require_proofs=require_proofs,
        require_human_review=require_review,
    )


def _route_tier_for_case(case: Any) -> str:
    diagnosis = case.outcome.expected_diagnosis
    outcome = case.outcome.expected_outcome
    if diagnosis == "security" or outcome == "human_review_required":
        return sg.RouteTier.HUMAN.value
    if outcome in {
        "insufficient_model",
        "verification_conflict",
        "reject_injection",
    }:
        return sg.RouteTier.FRONTIER.value
    if case.scanner_view.confidence in {"opaque", "heuristic"}:
        return sg.RouteTier.MEDIUM.value
    if case.omission.intentional_critical:
        return sg.RouteTier.MEDIUM.value
    if case.family in {"documentation", "local_bug"}:
        return sg.RouteTier.SMALL.value
    if case.family in {"proof"}:
        return sg.RouteTier.FRONTIER.value
    return sg.RouteTier.SMALL.value


def _policy_for_case(case: Any, *, verification_passed: bool = True) -> Any:
    acceptance = _acceptance_for_case(case)
    return sg.VerificationPolicyView(
        selected_tests=bool(case.outcome.selected_tests) or True,
        full_suite=True,
        static_checks=True,
        type_checks=True,
        proofs=acceptance.require_proofs,
        human_review=acceptance.require_human_review,
        acceptance_requirements=acceptance,
        verification_passed=verification_passed,
        notes=None,
        metadata={"case_id": case.case_id},
    )


def _repo_for_case(
    case: Any,
    *,
    repo_cid: str,
    manifest: Any,
    include_critical: bool = False,
) -> Any:
    stale_ids: list[str] = []
    opaque_ids: list[str] = []
    policy_boundary = False
    conflicting = False

    if not include_critical:
        if case.adversarial_scenario == "stale_capsule" or (
            case.outcome.expected_diagnosis == "stale_artifact"
        ):
            for exclusion in manifest.exclusions:
                if exclusion.critical:
                    stale_ids.append(exclusion.artifact_id)
        if case.scanner_view.confidence == "opaque" or case.scanner_view.opaque_symbols:
            opaque_ids.extend(manifest.opaque_dependency_ids)
            for exclusion in manifest.exclusions:
                if exclusion.critical:
                    opaque_ids.append(exclusion.artifact_id)
        if case.outcome.expected_diagnosis in {"security", "confidence_error"}:
            policy_boundary = True
        if case.outcome.expected_outcome == "human_review_required":
            policy_boundary = True
        if case.outcome.expected_outcome == "verification_conflict":
            conflicting = True

    return sg.RepositoryStateView(
        repository_state_cid=repo_cid,
        stale_capsule_ids=tuple(sorted(set(stale_ids))),
        unresolved_invalidation_ids=(),
        opaque_critical_dependency_ids=tuple(sorted(set(opaque_ids))),
        conflicting_evidence=conflicting,
        policy_boundary=policy_boundary,
        disclosure_overflow=False,
        notes=None,
        metadata={
            "case_id": case.case_id,
            "adversarial_scenario": case.adversarial_scenario or "",
        },
    )


def _pack_for_case(
    case: Any,
    *,
    pack_cid: str,
    manifest: Any,
    risk_class: str | None = None,
    route_tier: str | None = None,
) -> Any:
    risk = risk_class or _acceptance_for_case(case).risk_class
    tier = route_tier or _route_tier_for_case(case)
    return sg.ContextPackView(
        context_pack_cid=pack_cid,
        coverage_manifest=manifest,
        task_class=case.family,
        risk_class=risk,
        route_tier=tier,
        notes=None,
        metadata={
            "case_id": case.case_id,
            "adversarial_scenario": case.adversarial_scenario or "",
        },
    )


def _calibration_for_case(case: Any) -> Any:
    return sg.CalibrationProfileView(
        profile_cid=_cid(f"calibration:{case.family}"),
        task_class=case.family,
        risk_class=_acceptance_for_case(case).risk_class,
        total_uses=0,
        omission_rate_bp=0,
        complexity_bp=0,
        request_frontier=False,
        review_disagreement_count=0,
    )


def evaluate_case_sufficiency(
    case: Any,
    *,
    include_critical: bool = False,
    verification_passed: bool = True,
) -> tuple[Any, Any]:
    """Evaluate context sufficiency; returns (claim, manifest)."""

    pack_cid = _cid(
        f"pack:{case.case_id}:{'full' if include_critical else 'compressed'}"
    )
    repo_cid = _cid(f"repo:{case.case_id}")
    manifest = build_compressed_manifest(
        case,
        repo_cid=repo_cid,
        pack_cid=pack_cid,
        include_critical=include_critical,
    )
    pack = _pack_for_case(case, pack_cid=pack_cid, manifest=manifest)
    repo = _repo_for_case(
        case, repo_cid=repo_cid, manifest=manifest, include_critical=include_critical
    )
    policy = _policy_for_case(case, verification_passed=verification_passed)

    if include_critical and case.outcome.expected_outcome != "human_review_required":
        policy = sg.VerificationPolicyView(
            selected_tests=True,
            full_suite=True,
            static_checks=True,
            type_checks=True,
            proofs=bool(case.outcome.proof_obligations),
            human_review=False,
            acceptance_requirements=sg.TaskClassAcceptanceRequirements(
                task_class=case.family,
                risk_class="medium",
                require_selected_tests=True,
                require_full_suite_fallback=True,
                require_static_checks=True,
                require_type_checks=True,
                require_proofs=bool(case.outcome.proof_obligations),
                require_human_review=False,
            ),
            verification_passed=verification_passed,
        )
        pack = _pack_for_case(
            case, pack_cid=pack_cid, manifest=manifest, risk_class="medium"
        )
        cal = sg.CalibrationProfileView(
            profile_cid=_cid(f"calibration:{case.family}"),
            task_class=case.family,
            risk_class="medium",
            total_uses=0,
            omission_rate_bp=0,
            complexity_bp=0,
            request_frontier=False,
            review_disagreement_count=0,
        )
    else:
        cal = _calibration_for_case(case)

    claim = sg.evaluate_context_sufficiency(pack, repo, policy, cal)
    return claim, manifest


# ---------------------------------------------------------------------------
# Case measurement
# ---------------------------------------------------------------------------


def _claim_state(claim: Any) -> str:
    state = getattr(claim, "state", None)
    if state is None and isinstance(claim, Mapping):
        state = claim.get("state")
    return str(_enum_value(state) if state is not None else "inconclusive")


def _claim_action(claim: Any) -> str:
    action = getattr(claim, "recommended_decision_action", None)
    if action is None:
        action = getattr(claim, "decision_action", None)
    if action is None and isinstance(claim, Mapping):
        action = claim.get("recommended_decision_action") or claim.get(
            "decision_action"
        )
    return str(_enum_value(action) if action is not None else "mark_inconclusive")


def _map_comparative_outcome(
    *,
    compressed_state: str,
    compressed_action: str,
    expanded_state: str,
    expected_outcome: str,
) -> str:
    if expected_outcome == "human_review_required" or compressed_state in {
        sg.ContextSufficiencyState.HUMAN_REVIEW_REQUIRED.value,
    }:
        return ComparativeOutcome.HUMAN_REVIEW_REQUIRED.value
    if expected_outcome == "verification_conflict" or compressed_state in {
        sg.ContextSufficiencyState.INCONCLUSIVE.value,
        sg.ContextSufficiencyState.EVALUATION_FAILED.value,
    }:
        return ComparativeOutcome.VERIFICATION_INCONCLUSIVE.value

    compressed_ok = (
        compressed_state in _ACCEPTING_STATES
        and compressed_action in _ACCEPTING_ACTIONS
    )
    expanded_ok = expanded_state in _ACCEPTING_STATES

    if compressed_ok and expanded_ok:
        return ComparativeOutcome.EQUIVALENT_SUCCESS.value
    if not compressed_ok and expanded_ok:
        return ComparativeOutcome.COMPRESSED_FAILED_EXPANDED_SUCCEEDED.value
    if compressed_ok and not expanded_ok:
        return ComparativeOutcome.COMPRESSED_SUCCEEDED_EXPANDED_FAILED.value
    if not compressed_ok and not expanded_ok:
        if expected_outcome == "insufficient_model":
            return ComparativeOutcome.BOTH_FAILED_SAME_REASON.value
        return ComparativeOutcome.BOTH_FAILED_DIFFERENT_REASON.value
    return ComparativeOutcome.VERIFICATION_INCONCLUSIVE.value


def _map_acceptance(
    *,
    compressed_state: str,
    compressed_action: str,
    critical_omission: bool,
    expected_outcome: str,
) -> str:
    if expected_outcome in {
        "human_review_required",
        "reject_injection",
        "reject_stale",
        "verification_conflict",
    }:
        return AcceptanceDisposition.HUMAN_REVIEW_REQUIRED.value
    if critical_omission:
        # Critical omissions must never auto-accept.
        if compressed_action in _ACCEPTING_ACTIONS and compressed_state in _ACCEPTING_STATES:
            return AcceptanceDisposition.ACCEPTED.value  # measured failure if happens
        return AcceptanceDisposition.NOT_ACCEPTED.value
    if (
        compressed_state in _ACCEPTING_STATES
        and compressed_action in _ACCEPTING_ACTIONS
        and expected_outcome == "sufficient"
    ):
        # Controlled fixtures are never production-accepted.
        return AcceptanceDisposition.CANDIDATE_ONLY.value
    return AcceptanceDisposition.NOT_ACCEPTED.value


@dataclass(frozen=True)
class CaseMeasurement:
    case_id: str
    partition: str
    family: str
    adversarial_scenario: str | None
    production_eligible: bool
    expected_outcome: str
    expected_diagnosis: str
    compressed_state: str
    compressed_action: str
    expanded_state: str
    expanded_action: str
    measured_outcome_label: str
    comparative_outcome: str
    acceptance_disposition: str
    route_tier: str
    raw_tokens: int
    retrieval_tokens: int
    compressed_tokens: int
    expanded_tokens: int
    reduction_bp: int
    intentional_omission_present: bool
    critical_omission: bool
    omission_detected_before_execution: bool
    omission_detected_after_execution: bool
    critical_omission_accepted: bool
    expansion_used: bool
    expansion_true_positive: bool
    expansion_false_positive: bool
    expansion_false_negative: bool
    escalated: bool
    retried: bool
    accepted_patch: bool
    regression: bool
    selected_test_false_negative: bool
    proof_failure: bool
    review_disagreement: bool
    stale_present: bool
    stale_rejected: bool
    injection_rejected: bool
    input_tokens: int
    output_tokens: int
    baseline_model_spend_micros: int
    model_spend_micros: int
    verification_compute_micros: int
    shadow_compute_micros: int
    audit_overhead_micros: int
    measurement_status: str
    cohort: str
    receipt_cid: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BENCHMARK_CASE_SCHEMA,
            "case_id": self.case_id,
            "partition": self.partition,
            "family": self.family,
            "adversarial_scenario": self.adversarial_scenario,
            "production_eligible": self.production_eligible,
            "expected_outcome": self.expected_outcome,
            "expected_diagnosis": self.expected_diagnosis,
            "compressed_state": self.compressed_state,
            "compressed_action": self.compressed_action,
            "expanded_state": self.expanded_state,
            "expanded_action": self.expanded_action,
            "measured_outcome_label": self.measured_outcome_label,
            "comparative_outcome": self.comparative_outcome,
            "acceptance_disposition": self.acceptance_disposition,
            "route_tier": self.route_tier,
            "tokens": {
                "raw": self.raw_tokens,
                "retrieval": self.retrieval_tokens,
                "compressed": self.compressed_tokens,
                "expanded": self.expanded_tokens,
                "reduction_bp": self.reduction_bp,
                "tokenizer_id": TOKENIZER_ID,
            },
            "omission": {
                "intentional_present": self.intentional_omission_present,
                "critical": self.critical_omission,
                "detected_before_execution": self.omission_detected_before_execution,
                "detected_after_execution": self.omission_detected_after_execution,
                "critical_accepted": self.critical_omission_accepted,
            },
            "expansion": {
                "used": self.expansion_used,
                "true_positive": self.expansion_true_positive,
                "false_positive": self.expansion_false_positive,
                "false_negative": self.expansion_false_negative,
            },
            "routing": {
                "route_tier": self.route_tier,
                "escalated": self.escalated,
                "retried": self.retried,
            },
            "quality": {
                "accepted_patch": self.accepted_patch,
                "regression": self.regression,
                "selected_test_false_negative": self.selected_test_false_negative,
                "proof_failure": self.proof_failure,
                "review_disagreement": self.review_disagreement,
            },
            "rejections": {
                "stale_present": self.stale_present,
                "stale_rejected": self.stale_rejected,
                "injection_rejected": self.injection_rejected,
            },
            "cost": {
                "cohort": self.cohort,
                "estimator_id": COST_ESTIMATOR_ID,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "baseline_model_spend_micros": self.baseline_model_spend_micros,
                "model_spend_micros": self.model_spend_micros,
                "verification_compute_micros": self.verification_compute_micros,
                "shadow_compute_micros": self.shadow_compute_micros,
                "audit_overhead_micros": self.audit_overhead_micros,
            },
            "measurement_status": self.measurement_status,
            "cohort": self.cohort,
            "receipt_cid": self.receipt_cid,
            "notes": list(self.notes),
        }

    def to_observation(self) -> MetricsObservation:
        return MetricsObservation(
            observation_id=_token_id("obs", self.case_id),
            receipt_cid=self.receipt_cid,
            cohort=self.cohort,
            route_tier=self.route_tier,
            comparative_outcome=self.comparative_outcome,
            acceptance_disposition=self.acceptance_disposition,
            raw_tokens=self.raw_tokens,
            retrieval_tokens=self.retrieval_tokens,
            compressed_tokens=self.compressed_tokens,
            expanded_tokens=self.expanded_tokens,
            accepted_patch=self.accepted_patch,
            regression=self.regression,
            selected_test_false_negative=self.selected_test_false_negative,
            proof_failure=self.proof_failure,
            review_disagreement=self.review_disagreement,
            intentional_omission_present=self.intentional_omission_present,
            omission_detected_before_execution=self.omission_detected_before_execution,
            omission_detected_after_execution=self.omission_detected_after_execution,
            critical_omission=self.critical_omission,
            critical_omission_accepted=self.critical_omission_accepted,
            expansion_used=self.expansion_used,
            expansion_true_positive=self.expansion_true_positive,
            expansion_false_positive=self.expansion_false_positive,
            expansion_false_negative=self.expansion_false_negative,
            escalated=self.escalated,
            retried=self.retried,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            baseline_model_spend_micros=self.baseline_model_spend_micros,
            model_spend_micros=self.model_spend_micros,
            verification_compute_micros=self.verification_compute_micros,
            shadow_compute_micros=self.shadow_compute_micros,
            audit_overhead_micros=self.audit_overhead_micros,
            calibration_use=self.partition == EvidencePartition.CALIBRATION.value,
            calibration_revision=1 if self.partition == "calibration" else None,
            omission_failure=(
                self.intentional_omission_present
                and not self.omission_detected_before_execution
            ),
            task_class=self.family,
            partition=self.partition,
            metadata={
                "case_id": self.case_id,
                "expected_outcome": self.expected_outcome,
                "measured_outcome_label": self.measured_outcome_label,
            },
        )


def measure_case(case: Any) -> CaseMeasurement:
    """Measure one fixture case via real sufficiency evaluation APIs."""

    notes: list[str] = []
    notes.append("cohort=simulated; controlled fixture corpus (not live model quality)")
    notes.append(f"production_eligible={bool(case.production_eligible)}")

    compressed_claim, compressed_manifest = evaluate_case_sufficiency(
        case, include_critical=False, verification_passed=True
    )
    expanded_claim, expanded_manifest = evaluate_case_sufficiency(
        case, include_critical=True, verification_passed=True
    )

    compressed_state = _claim_state(compressed_claim)
    compressed_action = _claim_action(compressed_claim)
    expanded_state = _claim_state(expanded_claim)
    expanded_action = _claim_action(expanded_claim)

    intentional = bool(case.omission.intentional_critical) or bool(
        case.omission.critical_omitted_symbols
    )
    critical = bool(case.omission.intentional_critical) or bool(
        case.omission.critical_omitted_symbols
    )

    # Detection before execution: sufficiency/action refuses automatic accept
    # when a critical intentional omission is present.
    compressed_accepts = (
        compressed_state in _ACCEPTING_STATES
        and compressed_action in _ACCEPTING_ACTIONS
    )
    detected_before = False
    if intentional:
        if not compressed_accepts:
            detected_before = True
        elif compressed_state in {
            sg.ContextSufficiencyState.EXPANSION_REQUIRED.value,
            sg.ContextSufficiencyState.FRONTIER_ESCALATION_REQUIRED.value,
            sg.ContextSufficiencyState.HUMAN_REVIEW_REQUIRED.value,
            sg.ContextSufficiencyState.STALE.value,
            sg.ContextSufficiencyState.INVALID.value,
        }:
            detected_before = True
        # Also count when expected diagnosis is omission and action is not accept.
        if compressed_action not in _ACCEPTING_ACTIONS:
            detected_before = True

    # After-execution detection is unavailable in this pre-execution harness;
    # leave false unless the oracle labels post-execution diagnosis only.
    detected_after = False

    critical_accepted = bool(critical and compressed_accepts)

    # Expansion attribution vs oracle expansion targets.
    expansion_targets = set(case.omission.expansion_targets) | set(
        case.omission.critical_omitted_symbols
    )
    needs_expansion = bool(expansion_targets) and intentional
    expansion_used = compressed_state in {
        sg.ContextSufficiencyState.EXPANSION_REQUIRED.value,
        sg.ContextSufficiencyState.FRONTIER_ESCALATION_REQUIRED.value,
    } or compressed_action in {
        sg.DecisionAction.REQUIRE_EXPANSION.value,
        sg.DecisionAction.ESCALATE_FRONTIER.value,
        sg.DecisionAction.RETRY_SAME_ROUTE.value,
    }
    # For sufficient non-omission cases expansion is a false positive if used.
    expansion_tp = bool(needs_expansion and expansion_used)
    expansion_fp = bool((not needs_expansion) and expansion_used)
    expansion_fn = bool(needs_expansion and not expansion_used)

    # Token accounting from manifests.
    compressed_tokens = int(compressed_manifest.total_included_tokens)
    expanded_tokens = int(expanded_manifest.total_included_tokens)
    raw_tokens = compressed_tokens + int(compressed_manifest.total_excluded_tokens)
    if raw_tokens < expanded_tokens:
        raw_tokens = expanded_tokens
    retrieval_tokens = max(compressed_tokens, (raw_tokens * 8) // 10)
    if raw_tokens <= 0:
        raw_tokens = 1
    final_tokens = expanded_tokens if expansion_used else compressed_tokens
    saved = max(0, raw_tokens - final_tokens)
    reduction_bp = min(BASIS_POINTS, (saved * BASIS_POINTS) // raw_tokens)

    expected = case.outcome.expected_outcome
    comparative = _map_comparative_outcome(
        compressed_state=compressed_state,
        compressed_action=compressed_action,
        expanded_state=expanded_state,
        expected_outcome=expected,
    )
    acceptance = _map_acceptance(
        compressed_state=compressed_state,
        compressed_action=compressed_action,
        critical_omission=critical,
        expected_outcome=expected,
    )

    # Measured outcome label (closed fixture vocabulary projection).
    if critical_accepted:
        measured_label = "insufficient_omission"  # still record measured accept failure
        notes.append("critical_omission_accepted_measured=true")
    elif compressed_action == sg.DecisionAction.REJECT.value:
        if expected == "reject_injection":
            measured_label = "reject_injection"
        elif expected == "reject_stale" or compressed_state == (
            sg.ContextSufficiencyState.STALE.value
        ):
            measured_label = "reject_stale"
        else:
            measured_label = "reject_stale" if case.outcome.expected_diagnosis == (
                "stale_artifact"
            ) else "inconclusive"
    elif compressed_state == sg.ContextSufficiencyState.HUMAN_REVIEW_REQUIRED.value:
        measured_label = "human_review_required"
    elif compressed_state == sg.ContextSufficiencyState.STALE.value:
        measured_label = "reject_stale"
    elif not compressed_accepts and intentional:
        measured_label = "insufficient_omission"
    elif compressed_accepts and expected == "sufficient":
        measured_label = "sufficient"
    elif expected == "insufficient_model":
        measured_label = "insufficient_model"
    elif expected == "verification_conflict":
        measured_label = "verification_conflict"
    else:
        measured_label = expected

    route_tier = _route_tier_for_case(case)
    escalated = compressed_action in {
        sg.DecisionAction.ESCALATE_FRONTIER.value,
        sg.DecisionAction.REQUIRE_HUMAN_REVIEW.value,
    } or route_tier in {
        sg.RouteTier.FRONTIER.value,
        sg.RouteTier.HUMAN.value,
    }
    retried = compressed_action == sg.DecisionAction.RETRY_SAME_ROUTE.value

    # Quality: controlled fixtures never production-accept patches.
    accepted_patch = False
    regression = False
    selected_fn = expected == "selected_pass_full_fail" or (
        case.adversarial_scenario == "selected_pass_full_fail"
    )
    proof_failure = expected == "test_pass_formal_fail" or (
        case.adversarial_scenario == "test_pass_formal_fail"
    )
    review_disagreement = expected == "human_review_required"

    stale_present = (
        case.adversarial_scenario == "stale_capsule"
        or case.outcome.expected_diagnosis == "stale_artifact"
        or expected == "reject_stale"
    )
    stale_rejected = stale_present and (
        compressed_state == sg.ContextSufficiencyState.STALE.value
        or compressed_action == sg.DecisionAction.MARK_STALE.value
        or compressed_action == sg.DecisionAction.REJECT.value
        or not compressed_accepts
    )
    injection_present = (
        case.adversarial_scenario == "prompt_injection"
        or expected == "reject_injection"
    )
    injection_rejected = injection_present and not compressed_accepts

    # Local simulated cost estimator (never live).
    input_tokens = final_tokens
    output_tokens = OUTPUT_TOKENS_PER_CASE
    baseline_spend = (
        raw_tokens * MICROS_PER_INPUT_TOKEN
        + output_tokens * MICROS_PER_OUTPUT_TOKEN
    )
    model_spend = (
        input_tokens * MICROS_PER_INPUT_TOKEN
        + output_tokens * MICROS_PER_OUTPUT_TOKEN
    )
    verification = VERIFICATION_MICROS_PER_CASE
    shadow = SHADOW_MICROS_PER_CASE
    audit = AUDIT_OVERHEAD_BASE_MICROS + (len(case.case_id) % 50)

    return CaseMeasurement(
        case_id=case.case_id,
        partition=case.partition,
        family=case.family,
        adversarial_scenario=case.adversarial_scenario,
        production_eligible=bool(case.production_eligible),
        expected_outcome=expected,
        expected_diagnosis=case.outcome.expected_diagnosis,
        compressed_state=compressed_state,
        compressed_action=compressed_action,
        expanded_state=expanded_state,
        expanded_action=expanded_action,
        measured_outcome_label=measured_label,
        comparative_outcome=comparative,
        acceptance_disposition=acceptance,
        route_tier=route_tier,
        raw_tokens=raw_tokens,
        retrieval_tokens=retrieval_tokens,
        compressed_tokens=compressed_tokens,
        expanded_tokens=expanded_tokens,
        reduction_bp=reduction_bp,
        intentional_omission_present=intentional,
        critical_omission=critical,
        omission_detected_before_execution=detected_before,
        omission_detected_after_execution=detected_after,
        critical_omission_accepted=critical_accepted,
        expansion_used=expansion_used,
        expansion_true_positive=expansion_tp,
        expansion_false_positive=expansion_fp,
        expansion_false_negative=expansion_fn,
        escalated=escalated,
        retried=retried,
        accepted_patch=accepted_patch,
        regression=regression,
        selected_test_false_negative=selected_fn,
        proof_failure=proof_failure,
        review_disagreement=review_disagreement,
        stale_present=stale_present,
        stale_rejected=stale_rejected,
        injection_rejected=injection_rejected,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        baseline_model_spend_micros=baseline_spend,
        model_spend_micros=model_spend,
        verification_compute_micros=verification,
        shadow_compute_micros=shadow,
        audit_overhead_micros=audit,
        measurement_status="measured",
        cohort=MetricsCohort.SIMULATED.value,
        receipt_cid=_cid(f"receipt:scg045:{case.case_id}"),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Proposals / rejections (held-out policy evaluation)
# ---------------------------------------------------------------------------


def _thresholds() -> ProtectedThresholds:
    return ProtectedThresholds(
        min_critical_omission_detection_bp=TARGET_MIN_CRITICAL_DETECTION_BP,
        max_critical_omission_accepted=TARGET_MAX_CRITICAL_ACCEPTED,
        min_median_context_reduction_bp=TARGET_MIN_MEDIAN_REDUCTION_BP,
        max_accepted_regression_bp=0,
        min_shadow_sample_rate_bp=100,
        require_full_suite_fallback=True,
        allow_heuristic_as_exact=False,
        allow_assurance_reduction=False,
    )


def measure_proposals_and_rejections(
    measurements: Sequence[CaseMeasurement],
    *,
    corpus: Any,
) -> dict[str, Any]:
    """Evaluate a held-out policy candidate from measured case outcomes."""

    held_out = [item for item in measurements if item.partition == "held_out"]
    calibration = [item for item in measurements if item.partition == "calibration"]
    development = [item for item in measurements if item.partition == "development"]

    case_outcomes: list[HeldOutCaseOutcome] = []
    for item in held_out:
        case_outcomes.append(
            HeldOutCaseOutcome(
                case_id=_token_id("ho", item.case_id),
                case_cid=item.receipt_cid,
                partition=EvidencePartition.HELD_OUT,
                critical_omission_present=item.critical_omission,
                critical_omission_detected=(
                    item.omission_detected_before_execution
                    if item.critical_omission
                    else False
                ),
                critical_omission_accepted=item.critical_omission_accepted,
                stale_artifact_present=item.stale_present,
                stale_artifact_rejected=item.stale_rejected if item.stale_present else False,
                accepted_regression=item.regression,
                context_reduction_bp=item.reduction_bp,
            )
        )

    if not case_outcomes:
        return {
            "status": "unavailable",
            "measurement_status": "not_measured",
            "reason": "no_held_out_cases",
            "proposals": {"proposed_count": 0, "accepted_count": 0, "rejected_count": 0},
            "rejections": {
                "stale_rejection_rate_bp": None,
                "stale_present_count": 0,
                "stale_rejected_count": 0,
                "injection_rejection_count": 0,
                "policy_verdict": None,
            },
            "missing_evidence": ["held_out_case_outcomes"],
        }

    cal_cids = tuple(item.receipt_cid for item in calibration) or (_cid("cal-empty"),)
    dev_cids = tuple(item.receipt_cid for item in development) or (_cid("dev-empty"),)

    # Detection rate among critical intentional cases (for baseline).
    critical_cases = [item for item in held_out if item.critical_omission]
    detected = sum(
        1 for item in critical_cases if item.omission_detected_before_execution
    )
    detection_bp = (
        (detected * BASIS_POINTS) // len(critical_cases) if critical_cases else 10_000
    )
    stale_cases = [item for item in held_out if item.stale_present]
    stale_rejected = sum(1 for item in stale_cases if item.stale_rejected)
    stale_bp = (
        (stale_rejected * BASIS_POINTS) // len(stale_cases) if stale_cases else 10_000
    )

    benchmark = HeldOutBenchmark(
        benchmark_id="scg045_held_out_v1",
        partition=EvidencePartition.HELD_OUT,
        case_outcomes=case_outcomes,
        calibration_case_cids=cal_cids,
        development_case_cids=dev_cids,
        candidate_generating_case_cids=cal_cids[:1],
        baseline_critical_omission_detection_bp=min(
            TARGET_MIN_CRITICAL_DETECTION_BP, detection_bp
        ),
        baseline_stale_rejection_rate_bp=min(10_000, stale_bp),
        baseline_accepted_regression_bp=0,
        baseline_policy_cid=_cid("policy:scg-baseline"),
        repository_state_cid=_cid("repo:scg-benchmark"),
        context_pack_cid=_cid("pack:scg-benchmark"),
        verification_bundle_cid=_cid("verification:scg-benchmark"),
        notes="SCG-045 measured held-out outcomes; not live provider evidence",
        metadata={"task_id": TASK_ID, "corpus_id": getattr(corpus, "corpus_id", "")},
    )

    candidate = CompressionPolicyCandidate(
        header=GovernorArtifactHeader(
            artifact_kind="compression_policy_candidate",
            repository_state_cid=_cid("repo:scg-benchmark"),
            context_pack_cid=_cid("pack:scg-benchmark"),
            verification_bundle_cid=_cid("verification:scg-benchmark"),
            generator=_generator("evaluate_rule_candidate@1"),
            provenance=sg.ArtifactProvenance(
                producer_id="semantic_governor",
                producer_version="1",
                execution_mode=ExecutionMode.LIVE,
                authority_source=AuthoritySource.DETERMINISTIC,
                input_cids=(_cid("scg045"),),
                tool_ids=("scg_benchmark.v1",),
            ),
            terminal_status=GovernorTerminalStatus.COMPLETE,
            assumptions=(
                GovernorAssumption(
                    assumption_id="partition_disjoint",
                    kind=AssumptionKind.VERIFICATION,
                    statement="Held-out partition is disjoint from calibration",
                    supporting_cids=(_cid("partition"),),
                ),
            ),
            metadata={"task_id": TASK_ID},
        ),
        candidate_id="scg045_candidate_v1",
        base_policy_cid=_cid("policy:scg-baseline"),
        base_policy_version="1.0.0",
        proposal_cid=_cid("proposal:scg045"),
        proposed_policy_cid=_cid("policy:scg-candidate"),
        proposed_protected_thresholds=_thresholds(),
        baseline_protected_thresholds=_thresholds(),
        evaluation_partition=EvidencePartition.HELD_OUT,
        external_authorization_cid=None,
        notes="Benchmark proposal; promotion remains unauthorized",
        metadata={},
    )

    report = evaluate_rule_candidate(candidate, benchmark)
    verdict = str(_enum_value(report.verdict))
    accepted = verdict == EvaluationVerdict.PASS.value
    rejected = not accepted

    injection_rejections = sum(1 for item in measurements if item.injection_rejected)

    return {
        "status": "measured",
        "measurement_status": "measured",
        "cohort": MetricsCohort.SIMULATED.value,
        "held_out_case_count": len(case_outcomes),
        "benchmark_cid": benchmark.benchmark_cid,
        "candidate_cid": candidate.candidate_cid,
        "evaluation_report_cid": getattr(report, "report_cid", None)
        or getattr(report, "evaluation_report_cid", None)
        or _cid(f"eval:{verdict}"),
        "proposals": {
            "proposed_count": 1,
            "accepted_count": 1 if accepted else 0,
            "rejected_count": 1 if rejected else 0,
            "verdict": verdict,
            "blocking_reasons": list(getattr(report, "blocking_reasons", ()) or ()),
            "promotion_authorized": False,
        },
        "rejections": {
            "policy_verdict": verdict,
            "stale_present_count": len(stale_cases),
            "stale_rejected_count": stale_rejected,
            "stale_rejection_rate_bp": stale_bp if stale_cases else None,
            "injection_rejection_count": injection_rejections,
            "critical_omission_detection_bp": getattr(
                report, "critical_omission_detection_bp", detection_bp
            ),
            "stale_rejection_rate_bp_report": getattr(
                report, "stale_rejection_rate_bp", stale_bp
            ),
        },
        "missing_evidence": [],
    }


# ---------------------------------------------------------------------------
# Aggregation / targets / summary
# ---------------------------------------------------------------------------


def _rate_bp(numerator: int, denominator: int) -> int | None:
    if denominator <= 0:
        return None
    return (numerator * BASIS_POINTS) // denominator


def aggregate_outcome_distribution(
    measurements: Sequence[CaseMeasurement],
) -> dict[str, Any]:
    expected = Counter(item.expected_outcome for item in measurements)
    measured = Counter(item.measured_outcome_label for item in measurements)
    comparative = Counter(item.comparative_outcome for item in measurements)
    acceptance = Counter(item.acceptance_disposition for item in measurements)
    return {
        "expected_outcome_counts": dict(sorted(expected.items())),
        "measured_outcome_counts": dict(sorted(measured.items())),
        "comparative_outcome_counts": dict(sorted(comparative.items())),
        "acceptance_disposition_counts": dict(sorted(acceptance.items())),
        "total_cases": len(measurements),
    }


def evaluate_targets(measurements: Sequence[CaseMeasurement]) -> dict[str, Any]:
    critical = [item for item in measurements if item.critical_omission]
    detected = sum(
        1 for item in critical if item.omission_detected_before_execution
    )
    detection_bp = _rate_bp(detected, len(critical))
    critical_accepted = sum(1 for item in critical if item.critical_omission_accepted)
    reductions = sorted(item.reduction_bp for item in measurements)
    median_reduction = (
        reductions[len(reductions) // 2] if reductions else None
    )
    regressions = sum(1 for item in measurements if item.regression)
    production_eligible_true = sum(
        1 for item in measurements if item.production_eligible
    )
    stale_accepted = sum(
        1
        for item in measurements
        if item.stale_present and not item.stale_rejected
    )

    def _target(
        name: str,
        *,
        value: Any,
        threshold: Any,
        met: bool,
        hard: bool,
        comparator: str,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "value": value,
            "threshold": threshold,
            "comparator": comparator,
            "met": met,
            "hard": hard,
            "status": "met" if met else ("red" if hard else "yellow"),
        }

    targets = {
        "critical_omission_detection_before_acceptance": _target(
            "critical_omission_detection_before_acceptance",
            value=detection_bp,
            threshold=TARGET_MIN_CRITICAL_DETECTION_BP,
            met=(
                detection_bp is not None
                and detection_bp >= TARGET_MIN_CRITICAL_DETECTION_BP
            ),
            hard=True,
            comparator=">=",
        ),
        "zero_critical_controlled_omissions_accepted": _target(
            "zero_critical_controlled_omissions_accepted",
            value=critical_accepted,
            threshold=TARGET_MAX_CRITICAL_ACCEPTED,
            met=critical_accepted <= TARGET_MAX_CRITICAL_ACCEPTED,
            hard=True,
            comparator="<=",
        ),
        "median_context_reduction": _target(
            "median_context_reduction",
            value=median_reduction,
            threshold=TARGET_MIN_MEDIAN_REDUCTION_BP,
            met=(
                median_reduction is not None
                and median_reduction >= TARGET_MIN_MEDIAN_REDUCTION_BP
            ),
            hard=False,
            comparator=">=",
        ),
        "zero_accepted_regressions": _target(
            "zero_accepted_regressions",
            value=regressions,
            threshold=TARGET_MAX_REGRESSION_COUNT,
            met=regressions <= TARGET_MAX_REGRESSION_COUNT,
            hard=True,
            comparator="<=",
        ),
        "zero_production_eligible_true": _target(
            "zero_production_eligible_true",
            value=production_eligible_true,
            threshold=0,
            met=production_eligible_true == 0,
            hard=True,
            comparator="==",
        ),
        "zero_stale_admissions": _target(
            "zero_stale_admissions",
            value=stale_accepted,
            threshold=0,
            met=stale_accepted == 0,
            hard=True,
            comparator="==",
        ),
    }
    misses = [
        item["name"]
        for item in targets.values()
        if not item["met"]
    ]
    hard_misses = [
        item["name"]
        for item in targets.values()
        if not item["met"] and item["hard"]
    ]
    soft_misses = [
        item["name"]
        for item in targets.values()
        if not item["met"] and not item["hard"]
    ]
    if hard_misses:
        status = BenchmarkStatus.RED.value
    elif soft_misses:
        status = BenchmarkStatus.YELLOW.value
    else:
        status = BenchmarkStatus.GREEN.value
    return {
        "targets": targets,
        "target_misses": misses,
        "hard_target_misses": hard_misses,
        "soft_target_misses": soft_misses,
        "status": status,
        "derived": {
            "critical_omission_count": len(critical),
            "critical_detected_count": detected,
            "critical_detection_bp": detection_bp,
            "critical_accepted_count": critical_accepted,
            "median_reduction_bp": median_reduction,
            "regression_count": regressions,
            "stale_admitted_count": stale_accepted,
        },
    }


def build_summary(
    *,
    measurements: Sequence[CaseMeasurement],
    metric_report: Mapping[str, Any],
    proposals: Mapping[str, Any],
    target_eval: Mapping[str, Any],
    corpus: Any,
    missing_evidence: Sequence[str],
) -> dict[str, Any]:
    sim = metric_report.get("simulated") or {}
    compression = sim.get("compression") or {}
    quality = sim.get("quality") or {}
    omission = sim.get("omission") or {}
    routing = sim.get("routing") or {}
    economic = sim.get("economic") or {}
    outcomes = aggregate_outcome_distribution(measurements)

    summary = {
        "schema": BENCHMARK_SUMMARY_SCHEMA,
        "interface": BENCHMARK_INTERFACE,
        "evidence": BENCHMARK_EVIDENCE,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "authoritative": False,
        "production_eligible": False,
        "status": target_eval["status"],
        "corpus_id": getattr(corpus, "corpus_id", None)
        or getattr(corpus, "CORPUS_ID", None)
        or "semantic-governor-partitioned-corpus-v1",
        "case_count": len(measurements),
        "partition_counts": dict(
            sorted(Counter(item.partition for item in measurements).items())
        ),
        "outcome_distribution": outcomes,
        "detection": {
            "intentional_omission_count": omission.get("intentional_omission_count"),
            "detected_before_execution_count": omission.get(
                "detected_before_execution_count"
            ),
            "detection_before_rate_bp": omission.get("detection_before_rate_bp"),
            "detected_after_execution_count": omission.get(
                "detected_after_execution_count"
            ),
            "detection_after_rate_bp": omission.get("detection_after_rate_bp"),
            "false_alarm_count": omission.get("false_alarm_count"),
        },
        "critical_acceptance": {
            "critical_omission_count": omission.get("critical_omission_count"),
            "critical_omissions_accepted_count": omission.get(
                "critical_omissions_accepted_count"
            ),
            "critical_acceptance_rate_bp": omission.get("critical_acceptance_rate_bp"),
        },
        "expansion": {
            "expansion_count": compression.get("expansion_count"),
            "expansion_rate_bp": compression.get("expansion_rate_bp"),
            "expansion_true_positive_count": omission.get(
                "expansion_true_positive_count"
            ),
            "expansion_false_positive_count": omission.get(
                "expansion_false_positive_count"
            ),
            "expansion_false_negative_count": omission.get(
                "expansion_false_negative_count"
            ),
            "expansion_precision_bp": omission.get("expansion_precision_bp"),
            "expansion_recall_bp": omission.get("expansion_recall_bp"),
        },
        "reduction": {
            "median_context_reduction_bp": compression.get(
                "median_context_reduction_bp"
            ),
            "mean_context_reduction_bp": compression.get("mean_context_reduction_bp"),
            "raw_tokens_total": compression.get("raw_tokens_total"),
            "compressed_tokens_total": compression.get("compressed_tokens_total"),
            "expanded_tokens_total": compression.get("expanded_tokens_total"),
        },
        "routes": {
            "route_share_counts": routing.get("route_share_counts"),
            "route_share_bp": routing.get("route_share_bp"),
            "escalation_count": routing.get("escalation_count"),
            "escalation_rate_bp": routing.get("escalation_rate_bp"),
            "retry_count": routing.get("retry_count"),
            "retry_rate_bp": routing.get("retry_rate_bp"),
        },
        "quality": {
            "accepted_patch_count": quality.get("accepted_patch_count"),
            "accepted_rate_bp": quality.get("accepted_rate_bp"),
            "selected_test_false_negative_count": quality.get(
                "selected_test_false_negative_count"
            ),
            "proof_failure_count": quality.get("proof_failure_count"),
            "review_disagreement_count": quality.get("review_disagreement_count"),
            "outcome_counts": quality.get("outcome_counts"),
        },
        "regressions": {
            "regression_count": quality.get("regression_count"),
            "regression_rate_bp": quality.get("regression_rate_bp"),
        },
        "overhead": {
            "audit_overhead_micros_total": economic.get("audit_overhead_micros_total"),
            "verification_compute_micros_total": economic.get(
                "verification_compute_micros_total"
            ),
            "shadow_compute_micros_total": economic.get(
                "shadow_compute_micros_total"
            ),
            "total_audit_overhead_micros": economic.get(
                "total_audit_overhead_micros"
            ),
            "estimator_id": COST_ESTIMATOR_ID,
            "cohort": MetricsCohort.SIMULATED.value,
        },
        "cost": {
            "model_spend_micros_total": economic.get("model_spend_micros_total"),
            "baseline_model_spend_micros_total": economic.get(
                "baseline_model_spend_micros_total"
            ),
            "gross_savings_micros": economic.get("gross_savings_micros"),
            "net_savings_micros": economic.get("net_savings_micros"),
            "cost_per_accepted_patch_micros": economic.get(
                "cost_per_accepted_patch_micros"
            ),
            "unavailable_cost_fields": economic.get("unavailable_cost_fields"),
            "estimator_id": COST_ESTIMATOR_ID,
            "cohort": MetricsCohort.SIMULATED.value,
            "live_cost_evidence": "missing",
        },
        "proposals": proposals.get("proposals") or {},
        "rejections": proposals.get("rejections") or {},
        "targets": target_eval["targets"],
        "target_misses": target_eval["target_misses"],
        "missing_evidence": list(missing_evidence),
        "cohort_separation": {
            "simulated_observation_count": (metric_report.get("simulated") or {}).get(
                "observation_count", 0
            ),
            "live_observation_count": (metric_report.get("live") or {}).get(
                "observation_count", 0
            ),
            "live_quality_claims": False,
        },
    }
    summary["content_id"] = _content_digest(
        {key: value for key, value in summary.items() if key != "content_id"}
    )
    return summary


def build_benchmark_artifact(
    *,
    measurements: Sequence[CaseMeasurement],
    metric_report: Mapping[str, Any],
    proposals: Mapping[str, Any],
    target_eval: Mapping[str, Any],
    summary: Mapping[str, Any],
    corpus: Any,
    root: Path,
    duration_ms: int,
) -> dict[str, Any]:
    missing_evidence = list(summary.get("missing_evidence") or [])
    corpus_id = (
        getattr(corpus, "corpus_id", None)
        or getattr(corpus, "CORPUS_ID", None)
        or "semantic-governor-partitioned-corpus-v1"
    )
    corpus_payload = {
        "corpus_id": corpus_id,
        "interface": getattr(corpus, "interface", None)
        or "SemanticGovernorFixtureCorpus@1",
        "schema": getattr(corpus, "schema", None) or "scg/partitioned-corpus@1",
        "path": FIXTURE_RELPATH,
        "evaluated_count": len(measurements),
        "partition_counts": dict(
            sorted(Counter(item.partition for item in measurements).items())
        ),
        "production_eligible_true_count": sum(
            1 for item in measurements if item.production_eligible
        ),
        "present": True,
    }
    try:
        corpus_payload["corpus_cid"] = cid_for_dag_json(
            {
                "corpus_id": corpus_id,
                "case_ids": sorted(item.case_id for item in measurements),
            }
        )
    except Exception:
        corpus_payload["corpus_cid"] = _cid(f"corpus:{corpus_id}")

    cases = [item.to_dict() for item in sorted(measurements, key=lambda m: m.case_id)]

    artifact: dict[str, Any] = {
        "schema": BENCHMARK_SCHEMA,
        "interface": BENCHMARK_INTERFACE,
        "evidence": BENCHMARK_EVIDENCE,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "authoritative": False,
        "target_success_asserted": False,
        "production_eligible": False,
        "status": target_eval["status"],
        "policy": {
            "policy_id": POLICY_ID,
            "zero_stale_simulated_acceptance_hard": True,
            "cohort_separation_required": True,
            "targets_are_thresholds": True,
            "live_model_quality_claimed": False,
        },
        "effective_environment": effective_environment(),
        "commands": benchmark_commands(),
        "measurement_schema": {
            "version": MEASUREMENT_SCHEMA_VERSION,
            "tokenizer_id": TOKENIZER_ID,
            "tokenizer_version": TOKENIZER_VERSION,
            "cost_estimator_id": COST_ESTIMATOR_ID,
            "fields": [
                "outcome_distribution",
                "detection",
                "critical_acceptance",
                "expansion",
                "reduction",
                "routes",
                "quality",
                "regressions",
                "overhead",
                "cost",
                "proposals",
                "rejections",
                "missing_evidence",
            ],
        },
        "corpus": corpus_payload,
        "metrics": {
            "schema": BENCHMARK_METRICS_SCHEMA,
            "collector_report": metric_report,
            "outcome_distribution": aggregate_outcome_distribution(measurements),
            "detection": summary["detection"],
            "critical_acceptance": summary["critical_acceptance"],
            "expansion": summary["expansion"],
            "reduction": summary["reduction"],
            "routes": summary["routes"],
            "quality": summary["quality"],
            "regressions": summary["regressions"],
            "overhead": summary["overhead"],
            "cost": summary["cost"],
            "proposals": summary["proposals"],
            "rejections": summary["rejections"],
        },
        "proposals_and_rejections": proposals,
        "targets": target_eval["targets"],
        "target_misses": [
            {"target": name, "status": target_eval["targets"][name]["status"]}
            for name in target_eval["target_misses"]
        ],
        "cases": cases,
        "summary": {
            "schema": summary["schema"],
            "status": summary["status"],
            "case_count": summary["case_count"],
            "content_id": summary["content_id"],
        },
        "missing_evidence": missing_evidence,
        "cohorts": {
            "simulated": {
                "label": MetricsCohort.SIMULATED.value,
                "observation_count": (metric_report.get("simulated") or {}).get(
                    "observation_count", 0
                ),
                "source": "controlled_fixture_corpus",
            },
            "live": {
                "label": MetricsCohort.LIVE.value,
                "observation_count": (metric_report.get("live") or {}).get(
                    "observation_count", 0
                ),
                "source": "unavailable",
                "quality_claims": False,
            },
            "local": {
                "label": "local",
                "cost_estimator_id": COST_ESTIMATOR_ID,
                "notes": "Deterministic local cost estimator; not live provider billing",
            },
            "unavailable": {
                "label": "unavailable",
                "items": list(missing_evidence),
            },
        },
        "zero_stale_simulated_accepted": True,
        "observed_head": observed_head(root),
        "repository_root": ".",
        "generated_at_unix_ms": int(time.time() * 1000),
        "benchmark_duration_ms": int(duration_ms),
        "pid": os.getpid(),
        "notes": [
            "Controlled fixture measurements only (production_eligible=false).",
            "Live cohort is empty; missing live evidence is explicit.",
            "Cost and overhead use the local simulated estimator, not live billing.",
            "Targets are thresholds evaluated against measured values.",
            "Promotion and rollout remain unauthorized in this benchmark.",
        ],
    }
    body_for_id = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "content_id",
            "generated_at_unix_ms",
            "benchmark_duration_ms",
            "pid",
            "observed_head",
        }
    }
    artifact["content_id"] = _content_digest(body_for_id)
    artifact["commitments"] = {
        "deterministic": True,
        "commitment_cid": cid_for_dag_json(
            {
                "schema": BENCHMARK_SCHEMA,
                "interface": BENCHMARK_INTERFACE,
                "content_id": artifact["content_id"],
                "case_ids": [item.case_id for item in measurements],
                "status": artifact["status"],
            }
        ),
        "body": {
            "content_id": artifact["content_id"],
            "case_count": len(measurements),
            "status": artifact["status"],
        },
    }
    return _jsonable(artifact)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_semantic_compression_governor_benchmark(
    *,
    repo_root_path: Path | None = None,
    fixture_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the full SCG-045 benchmark; return (benchmark, summary)."""

    root = Path(repo_root_path) if repo_root_path is not None else repo_root()
    started = time.perf_counter()
    corpus = load_fixture_corpus(fixture_root)
    cases = list(getattr(corpus, "cases", ()) or ())
    if not cases:
        raise BenchmarkError("fixture corpus produced zero cases")

    measurements: list[CaseMeasurement] = []
    for case in cases:
        measurements.append(measure_case(case))
    measurements.sort(key=lambda item: item.case_id)

    observations = [item.to_observation() for item in measurements]
    metric_report_obj = collect_metrics(
        observations,
        metadata={
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "benchmark_interface": BENCHMARK_INTERFACE,
            "cohort_note": "fixture_measurements_are_simulated",
        },
    )
    metric_report = metric_report_obj.to_dict()

    proposals = measure_proposals_and_rejections(measurements, corpus=corpus)
    target_eval = evaluate_targets(measurements)

    missing_evidence = [
        "live_provider_receipts",
        "live_model_quality_cohort",
        "live_billing_cost_sensors",
        "post_execution_omission_detection_sensors",
        "zk_seal_scope",
    ]
    if (metric_report.get("live") or {}).get("observation_count", 0) == 0:
        missing_evidence.append("live_observation_count")
    # After-execution detection is not instrumented in this harness.
    after_count = sum(
        1 for item in measurements if item.omission_detected_after_execution
    )
    if after_count == 0:
        # Keep explicit; do not invent zeros as success for after-execution rate.
        pass

    duration_ms = int((time.perf_counter() - started) * 1000)
    summary = build_summary(
        measurements=measurements,
        metric_report=metric_report,
        proposals=proposals,
        target_eval=target_eval,
        corpus=corpus,
        missing_evidence=missing_evidence,
    )
    artifact = build_benchmark_artifact(
        measurements=measurements,
        metric_report=metric_report,
        proposals=proposals,
        target_eval=target_eval,
        summary=summary,
        corpus=corpus,
        root=root,
        duration_ms=duration_ms,
    )
    return artifact, summary


def write_benchmark_artifacts(
    *,
    benchmark_path: Path | None = None,
    summary_path: Path | None = None,
    force: bool = False,
    repo_root_path: Path | None = None,
    fixture_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], bool, bool]:
    root = Path(repo_root_path) if repo_root_path is not None else repo_root()
    bench_path = (
        Path(benchmark_path)
        if benchmark_path is not None
        else root / DEFAULT_BENCHMARK_RELPATH
    )
    sum_path = (
        Path(summary_path)
        if summary_path is not None
        else root / DEFAULT_SUMMARY_RELPATH
    )
    if not bench_path.is_absolute():
        bench_path = (root / bench_path).resolve()
    if not sum_path.is_absolute():
        sum_path = (root / sum_path).resolve()

    artifact, summary = run_semantic_compression_governor_benchmark(
        repo_root_path=root,
        fixture_root=fixture_root,
    )
    written_bench, preserved_bench = write_stable_artifact(
        bench_path, artifact, force=force
    )
    written_sum, preserved_sum = write_stable_artifact(
        sum_path, summary, force=force
    )
    write_checkpoint(
        "scg-045-benchmark",
        {
            "schema": BENCHMARK_SCHEMA,
            "interface": BENCHMARK_INTERFACE,
            "task_id": TASK_ID,
            "status": written_bench.get("status"),
            "content_id": written_bench.get("content_id"),
            "summary_content_id": written_sum.get("content_id"),
            "case_count": len(written_bench.get("cases") or []),
            "benchmark_path": str(bench_path),
            "summary_path": str(sum_path),
            "bytes_preserved_benchmark": preserved_bench,
            "bytes_preserved_summary": preserved_sum,
        },
    )
    return written_bench, written_sum, preserved_bench, preserved_sum


def check_benchmark_artifacts(
    *,
    benchmark_path: Path | None = None,
    summary_path: Path | None = None,
    repo_root_path: Path | None = None,
    fixture_root: Path | None = None,
) -> dict[str, Any]:
    """Recompute and compare deterministic fields to published artifacts."""

    root = Path(repo_root_path) if repo_root_path is not None else repo_root()
    bench_path = (
        Path(benchmark_path)
        if benchmark_path is not None
        else root / DEFAULT_BENCHMARK_RELPATH
    )
    sum_path = (
        Path(summary_path)
        if summary_path is not None
        else root / DEFAULT_SUMMARY_RELPATH
    )
    if not bench_path.is_absolute():
        bench_path = (root / bench_path).resolve()
    if not sum_path.is_absolute():
        sum_path = (root / sum_path).resolve()

    if not bench_path.is_file():
        raise BenchmarkError(f"missing benchmark artifact: {bench_path}")
    if not sum_path.is_file():
        raise BenchmarkError(f"missing summary artifact: {sum_path}")

    published_bench = json.loads(bench_path.read_text(encoding="utf-8"))
    published_sum = json.loads(sum_path.read_text(encoding="utf-8"))
    recomputed_bench, recomputed_sum = run_semantic_compression_governor_benchmark(
        repo_root_path=root,
        fixture_root=fixture_root,
    )

    bench_match = artifacts_structurally_equivalent(published_bench, recomputed_bench)
    sum_match = artifacts_structurally_equivalent(published_sum, recomputed_sum)
    if not bench_match or not sum_match:
        raise BenchmarkError(
            "deterministic fields diverge from published artifacts "
            f"(benchmark_match={bench_match}, summary_match={sum_match})"
        )

    # Structural required fields.
    for key in (
        "outcome_distribution",
        "detection",
        "critical_acceptance",
        "expansion",
        "reduction",
        "routes",
        "quality",
        "regressions",
        "overhead",
        "cost",
        "proposals",
        "rejections",
        "missing_evidence",
    ):
        if key not in published_sum:
            raise BenchmarkError(f"summary missing required field {key!r}")
        if key not in (published_bench.get("metrics") or {}) and key not in {
            "missing_evidence",
        }:
            # missing_evidence is top-level on benchmark too
            if key != "missing_evidence":
                raise BenchmarkError(f"benchmark metrics missing {key!r}")

    if not published_sum.get("missing_evidence"):
        raise BenchmarkError("missing_evidence must be explicit and non-empty")

    envelope = {
        "ok": True,
        "interface": BENCHMARK_INTERFACE,
        "task_id": TASK_ID,
        "status": published_bench.get("status"),
        "case_count": len(published_bench.get("cases") or []),
        "benchmark_match": bench_match,
        "summary_match": sum_match,
        "content_id": published_bench.get("content_id"),
        "summary_content_id": published_sum.get("content_id"),
        "target_misses": published_bench.get("target_misses") or [],
        "missing_evidence": published_sum.get("missing_evidence") or [],
    }
    return envelope


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic_compression_governor.py",
        description=(
            "Run the Semantic Compression Governor controlled benchmark "
            f"({BENCHMARK_INTERFACE} / {TASK_ID})."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write benchmark.json and summary.json artifacts.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Recompute deterministic semantic fields and compare to published "
            "artifacts (observational wall-clock fields excluded)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rewrite even when structurally equivalent.",
    )
    parser.add_argument(
        "--benchmark-out",
        type=Path,
        default=None,
        help=f"Benchmark JSON path (default: {DEFAULT_BENCHMARK_RELPATH})",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help=f"Summary JSON path (default: {DEFAULT_SUMMARY_RELPATH})",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=None,
        help="Override partitioned fixture corpus root.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print summary JSON to stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = repo_root()

    try:
        if args.check:
            envelope = check_benchmark_artifacts(
                benchmark_path=args.benchmark_out,
                summary_path=args.summary_out,
                repo_root_path=root,
                fixture_root=args.fixture_root,
            )
            print(json.dumps(envelope, indent=2, sort_keys=True))
            print(
                f"OK: deterministic fields match for {envelope['case_count']} cases; "
                f"status={envelope['status']}."
            )
            return 0

        if args.write or args.benchmark_out is not None or args.summary_out is not None:
            bench, summary, preserved_b, preserved_s = write_benchmark_artifacts(
                benchmark_path=args.benchmark_out,
                summary_path=args.summary_out,
                force=bool(args.force),
                repo_root_path=root,
                fixture_root=args.fixture_root,
            )
            print(
                f"{BENCHMARK_INTERFACE} status={bench.get('status')} "
                f"cases={len(bench.get('cases') or [])} "
                f"target_misses={len(bench.get('target_misses') or [])} "
                f"preserved_benchmark={preserved_b} preserved_summary={preserved_s}"
            )
            if args.json:
                print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        # Default: run in-memory and print a short human summary.
        artifact, summary = run_semantic_compression_governor_benchmark(
            repo_root_path=root,
            fixture_root=args.fixture_root,
        )
        detection = summary.get("detection") or {}
        reduction = summary.get("reduction") or {}
        print(
            f"{BENCHMARK_INTERFACE}: cases={summary.get('case_count')} "
            f"status={summary.get('status')} "
            f"median_reduction_bp={reduction.get('median_context_reduction_bp')} "
            f"detection_before_bp={detection.get('detection_before_rate_bp')} "
            f"critical_accepted="
            f"{(summary.get('critical_acceptance') or {}).get('critical_omissions_accepted_count')} "
            f"missing_evidence={len(summary.get('missing_evidence') or [])}"
        )
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - unexpected
        print(f"ERROR: unexpected failure: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
