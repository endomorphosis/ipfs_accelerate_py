"""SCG-041: structural omission, stale-artifact, policy, and bounded-expansion.

Static omission conformance over the held-out static adversarial fixture
partition (SCG-040 corpus) and the datasets public governor API (SCG-018):

* Critical intentional omissions are detected before automatic acceptance
  (verification pass alone never authorizes compressed acceptance).
* Exact sufficient context is not needlessly expanded.
* Expansion plans remain hard-bounded (steps, token growth, absolute limits).

Conflict policy: real fixture oracles and canonical governor views only.
No hand-injected passing identities, model outputs, or fabricated receipts.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts import semantic_governor as sg

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "test" / "fixtures" / "semantic_governor"
PACKAGE_NAME = "scg_partitioned_fixture_corpus"

TASK_ID = "SCG-041"
INTERFACE = "static omission conformance"
EVIDENCE_SUBSET = "held-out static fixture partition"

# Effects matrix: structural / static adversarial scenarios (SCG-041).
# Dynamic / injection / selected-vs-full / both-context cases belong to SCG-042.
STATIC_ADVERSARIAL_SCENARIOS: tuple[str, ...] = (
    "hidden_callee_side_effect",
    "caller_exception_contract",
    "config_flag",
    "pytest_fixture",
    "serializer",
    "generated_interface",
    "stale_capsule",
    "confidence_misclassification",
    "behavior_only_dependency",
    "security_invariant",
    "migration_path",
)

_ACCEPTING_STATES = frozenset(
    {
        sg.ContextSufficiencyState.SUFFICIENT.value,
        sg.ContextSufficiencyState.SUFFICIENT_WITH_CAVEATS.value,
    }
)
_ACCEPTING_ACTIONS = frozenset({sg.DecisionAction.ACCEPT_COMPRESSED.value})

_FAMILY_GAP_KIND: Mapping[str, str] = {
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

_CONFIDENCE_BP: Mapping[str, int] = {
    "exact": 10_000,
    "conservative": 7_500,
    "heuristic": 4_500,
    "opaque": 1_000,
}

_TOKEN_SAFE = re.compile(r"[^A-Za-z0-9_.:/+-]+")


# ---------------------------------------------------------------------------
# Fixture corpus loader (mirrors SCG-040 import isolation)
# ---------------------------------------------------------------------------


def _load_fixture_package() -> ModuleType:
    if PACKAGE_NAME in sys.modules and hasattr(
        sys.modules[PACKAGE_NAME], "SemanticGovernorFixtureCorpus"
    ):
        return sys.modules[PACKAGE_NAME]

    init_path = FIXTURE_DIR / "__init__.py"
    if not init_path.is_file():
        raise ImportError(f"missing fixture package init: {init_path}")

    package = ModuleType(PACKAGE_NAME)
    package.__file__ = str(init_path)
    package.__path__ = [str(FIXTURE_DIR)]  # type: ignore[attr-defined]
    sys.modules[PACKAGE_NAME] = package

    def _load_submodule(name: str, filename: str) -> ModuleType:
        qualname = f"{PACKAGE_NAME}.{name}"
        if qualname in sys.modules:
            return sys.modules[qualname]
        path = FIXTURE_DIR / filename
        spec = importlib.util.spec_from_file_location(qualname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = PACKAGE_NAME
        sys.modules[qualname] = module
        spec.loader.exec_module(module)
        setattr(package, name, module)
        return module

    _load_submodule("case_record", "case_record.py")
    _load_submodule("recipes", "recipes.py")
    _load_submodule("corpus", "corpus.py")

    init_spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME, init_path, submodule_search_locations=[str(FIXTURE_DIR)]
    )
    assert init_spec is not None and init_spec.loader is not None
    package.__spec__ = init_spec
    package.__package__ = PACKAGE_NAME
    init_spec.loader.exec_module(package)
    assert hasattr(package, "SemanticGovernorFixtureCorpus")
    return package


@pytest.fixture(scope="module")
def fixture_pkg() -> ModuleType:
    return _load_fixture_package()


@pytest.fixture(scope="module")
def corpus(fixture_pkg: ModuleType) -> Any:
    return fixture_pkg.SemanticGovernorFixtureCorpus.load()


@pytest.fixture(scope="module")
def static_adversarial_cases(corpus: Any) -> tuple[Any, ...]:
    cases = tuple(
        case
        for case in corpus.cases
        if case.adversarial_scenario in STATIC_ADVERSARIAL_SCENARIOS
    )
    assert len(cases) == len(STATIC_ADVERSARIAL_SCENARIOS)
    found = {case.adversarial_scenario for case in cases}
    assert found == set(STATIC_ADVERSARIAL_SCENARIOS)
    for case in cases:
        assert case.partition == "held_out"
        assert case.production_eligible is False
    return cases


@pytest.fixture(scope="module")
def held_out_sufficient_cases(corpus: Any) -> tuple[Any, ...]:
    cases = tuple(
        case
        for case in corpus.cases
        if case.partition == "held_out"
        and case.outcome.expected_outcome == "sufficient"
        and not case.omission.intentional_critical
        and case.outcome.automatic_accept_allowed
    )
    assert cases, "held-out sufficient control cases required"
    return cases


# ---------------------------------------------------------------------------
# Canonical view builders from fixture oracles
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _token_id(prefix: str, *parts: str) -> str:
    raw = "_".join((prefix, *parts))
    cleaned = _TOKEN_SAFE.sub("_", raw).strip("_").lower()
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"id_{cleaned}"
    return cleaned[:128]


def _sym_token(symbol: str) -> str:
    """Map fixture scanner symbols to governor lowercase identity tokens.

    Fixture oracles may use PascalCase type names (e.g. ``UserRecord``).
    Governor graph / symbol tokens require ``^[a-z][a-z0-9_.:/+-]{0,127}$``.
    """

    text = str(symbol).strip().lower()
    text = _TOKEN_SAFE.sub("_", text).strip("._")
    if not text or not text[0].isalpha():
        text = f"sym_{text}"
    return text[:128]


def _path_for_symbol(case: Any, symbol: str) -> str:
    scanner = case.scanner_view
    # Prefer scanner-declared paths for any symbol in the changed set (case-insensitive).
    lowered = {item.lower(): item for item in scanner.changed_symbols}
    if symbol in scanner.changed_symbols or symbol.lower() in lowered:
        if scanner.changed_paths:
            return scanner.changed_paths[0]
    # Derive a stable repo-relative path from the qualified symbol.
    if ":" in symbol:
        # docs.api_reference:fetch_value style
        head = symbol.split(":", 1)[0]
        return head.replace(".", "/") + ".md"
    if symbol.startswith("proof."):
        return "proofs/" + symbol[len("proof.") :].replace(".", "/") + ".lean"
    if symbol.startswith("tests."):
        # tests.conftest.sample_record -> tests/conftest.py
        body = symbol[len("tests.") :]
        module = body.rsplit(".", 1)[0]
        return "tests/" + module.replace(".", "/") + ".py"
    # scg_fixture.core.add -> scg_fixture/core.py
    parts = symbol.split(".")
    if len(parts) >= 2:
        return "/".join(parts[:-1]) + ".py"
    return "scg_fixture/unknown.py"


def _generator(interface_id: str = "evaluate_context_sufficiency@1") -> Any:
    return sg.GeneratorIdentity(
        generator_id="static_omission_conformance",
        generator_version="1.0.0",
        interface_id=interface_id,
    )


def _provenance(*, case_id: str) -> Any:
    return sg.ArtifactProvenance(
        producer_id="semantic_governor",
        producer_version="1",
        execution_mode=sg.ExecutionMode.LIVE,
        authority_source=sg.AuthoritySource.DETERMINISTIC,
        input_cids=(_cid(f"fixture:{case_id}"),),
        tool_ids=("static_omission.v1",),
        policy_cid=_cid("policy:scg-041"),
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
        "terminal_status": sg.GovernorTerminalStatus.COMPLETE,
        "assumptions": (
            sg.GovernorAssumption(
                assumption_id="fixture_oracle_binding",
                kind=sg.AssumptionKind.COVERAGE,
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
            "interface": INTERFACE,
            "evidence_subset": EVIDENCE_SUBSET,
        },
    }
    fields.update(overrides)
    return sg.GovernorArtifactHeader(**fields)  # type: ignore[arg-type]


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


def _inclusion(
    *,
    case: Any,
    symbol: str,
    inclusion_kind: str = sg.InclusionKind.RAW_SOURCE.value,
    token_cost: int = 40,
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
        token_cost=token_cost,
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
    token_cost: int = 50,
    substituted_by: str | None = None,
    repo_cid: str,
) -> Any:
    path = _path_for_symbol(case, symbol)
    primary = case.scanner_view.primary_symbol
    sym = _sym_token(symbol)
    prim = _sym_token(primary)
    nodes = (prim,) if sym == prim else (prim, sym)
    # Capsule substitution requires a substituted_by binding when claimed.
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
        token_cost=token_cost,
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


def _build_compressed_manifest(
    case: Any,
    *,
    repo_cid: str,
    pack_cid: str,
    include_critical: bool = False,
) -> Any:
    """Build a coverage manifest from the fixture omission oracle.

    When ``include_critical`` is False, intentional compressed omissions are
    represented as critical exclusions + known gaps (the adversarial pack).
    When True, expansion targets and critical symbols are raw-included so the
    pack is structurally complete (exact-sufficient control).
    """

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

    # Always represent the primary target when not critically omitted, so the
    # pack has a concrete inclusion cone.
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
            _inclusion(case=case, symbol=symbol, inclusion_kind=kind, token_cost=40)
        )

    critical_omitted = set(omission.critical_omitted_symbols)
    if not include_critical:
        for symbol in sorted(omit_set):
            is_critical = symbol in critical_omitted
            # Structural omission: represent as capsule substitution of a critical
            # subject (not budget overflow), so diagnose_omission ranks OMISSION.
            reason = sg.ExclusionReason.EXACT_CAPSULE_SUBSTITUTED.value
            if case.adversarial_scenario == "stale_capsule":
                reason = sg.ExclusionReason.EXACT_CAPSULE_SUBSTITUTED.value
            elif scanner.confidence == "opaque":
                reason = sg.ExclusionReason.CONSERVATIVE_CAPSULE_SUBSTITUTED.value
            elif not is_critical:
                reason = sg.ExclusionReason.OUTSIDE_AFFECTED_INVALIDATION_CONE.value
            # Critical structural omissions also carry an expansion-forcing gap;
            # budget-truncation gaps force EXPANSION_REQUIRED on sufficiency.
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
                    token_cost=10,
                    repo_cid=repo_cid,
                )
            )

    if not inclusions:
        # Fail closed construction would yield an empty pack; include primary raw.
        inclusions.append(
            _inclusion(
                case=case,
                symbol=primary,
                inclusion_kind=sg.InclusionKind.RAW_SOURCE.value,
                token_cost=40,
            )
        )

    # Stable order by artifact_id for deterministic identity.
    inclusions = tuple(sorted(inclusions, key=lambda item: item.artifact_id))
    exclusions = tuple(sorted(exclusions, key=lambda item: item.artifact_id))
    gaps = tuple(sorted(gaps, key=lambda item: item.gap_id))

    raw_count = sum(
        1
        for item in inclusions
        if item.inclusion_kind
        in {sg.InclusionKind.RAW_SOURCE.value, "raw_source"}
    )
    capsule_count = sum(
        1
        for item in inclusions
        if item.inclusion_kind
        in {
            sg.InclusionKind.EXACT_CAPSULE.value,
            sg.InclusionKind.CONSERVATIVE_CAPSULE.value,
            "exact_capsule",
            "conservative_capsule",
        }
    )
    dep_paths = []
    for item in inclusions:
        if item.dependency_path is not None:
            dep_paths.append(item.dependency_path)
    for item in exclusions:
        if item.dependency_path is not None:
            dep_paths.append(item.dependency_path)
    # Deduplicate by identity payload nodes.
    seen_paths: set[tuple[str, ...]] = set()
    unique_paths: list[Any] = []
    for path in dep_paths:
        key = tuple(path.nodes)
        if key not in seen_paths:
            seen_paths.add(key)
            unique_paths.append(path)

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
        inclusions=inclusions,
        exclusions=exclusions,
        context_budget_tokens=2_000,
        minimum_safe_tokens=40,
        total_included_tokens=sum(item.token_cost for item in inclusions),
        total_excluded_tokens=sum(item.token_cost for item in exclusions),
        raw_inclusion_count=raw_count,
        capsule_inclusion_count=capsule_count,
        exclusion_count=len(exclusions),
        known_gaps=gaps,
        opaque_dependency_ids=tuple(sorted(set(opaque_ids))),
        dependency_paths=tuple(unique_paths),
        policy_cid=_cid("policy:scg-041"),
        notes=None,
        metadata={
            "case_id": case.case_id,
            "adversarial_scenario": case.adversarial_scenario,
            "include_critical": include_critical,
        },
    )


def _acceptance_for_case(case: Any) -> Any:
    require_proofs = bool(case.outcome.proof_obligations)
    require_review = case.outcome.expected_diagnosis in {
        "security",
        "confidence_error",
    } or case.outcome.expected_outcome == "human_review_required"
    # Security / confidence policy cases keep review required only when the
    # oracle forbids auto-accept and names those diagnoses.
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
            "adversarial_scenario": case.adversarial_scenario,
            "tree_digest_label": f"tree:{case.case_id}",
        },
    )


def _pack_for_case(
    case: Any,
    *,
    pack_cid: str,
    manifest: Any,
    risk_class: str | None = None,
) -> Any:
    risk = risk_class or _acceptance_for_case(case).risk_class
    return sg.ContextPackView(
        context_pack_cid=pack_cid,
        coverage_manifest=manifest,
        task_class=case.family,
        risk_class=risk,
        route_tier=sg.RouteTier.SMALL,
        notes=None,
        metadata={
            "case_id": case.case_id,
            "adversarial_scenario": case.adversarial_scenario,
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


def _evaluate_case(
    case: Any,
    *,
    include_critical: bool = False,
    verification_passed: bool = True,
) -> Any:
    pack_cid = _cid(f"pack:{case.case_id}:{'full' if include_critical else 'compressed'}")
    repo_cid = _cid(f"repo:{case.case_id}")
    manifest = _build_compressed_manifest(
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
    # Full packs that previously required review solely due to omission must
    # not keep a hard human-review matrix once structural coverage is complete.
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
    return sg.evaluate_context_sufficiency(pack, repo, policy, cal)


def _audit_case(case: Any, *, pack_cid: str, repo_cid: str) -> Any:
    return sg.CompressionAuditCase(
        header=_header(
            "compression_audit_case",
            case_id=case.case_id,
            repo_cid=repo_cid,
            pack_cid=pack_cid,
            interface_id="diagnose_omission@1",
        ),
        case_id=_token_id("case", case.case_id),
        task_id=_token_id("task", case.case_id),
        task_class=case.family,
        risk_class=_acceptance_for_case(case).risk_class,
        coverage_manifest_cid=_cid(f"manifest:{case.case_id}"),
        sufficiency_claim_cid=_cid(f"claim:{case.case_id}"),
        decision_cid=_cid(f"decision:{case.case_id}"),
        run_receipt_cid=None,
        expansion_plan_cid=None,
        omission_evidence_cid=_cid(f"omission-evidence:{case.case_id}"),
        shadow_plan_cid=_cid(f"shadow-plan:{case.case_id}"),
        shadow_result_cid=_cid(f"shadow-result:{case.case_id}"),
        differential_report_cid=_cid(f"differential:{case.case_id}"),
        policy_cid=_cid("policy:scg-041"),
        benchmark_partition=case.partition,
        notes=None,
        metadata={
            "adversarial_scenario": case.adversarial_scenario,
            "case_id": case.case_id,
        },
    )


def _omission_repo_view(
    case: Any,
    *,
    pack_cid: str,
    repo_cid: str,
    exclusions: Sequence[Any],
    differential_outcome: str,
) -> dict[str, Any]:
    expanded = [
        item.artifact_id for item in exclusions if getattr(item, "critical", False)
    ]
    return {
        "repository_state_cid": repo_cid,
        "context_pack_cid": pack_cid,
        "verification_bundle_cid": _cid(f"verification:{case.case_id}"),
        "differential_outcome": differential_outcome,
        "exclusions": tuple(exclusions),
        "target_symbol_ids": (_sym_token(case.scanner_view.primary_symbol),),
        "counterexample_cids": (
            (_cid(f"counterexample:{case.case_id}"),)
            if differential_outcome
            in set(sg.omission_supporting_outcomes())
            else ()
        ),
        "minimized_failure_cids": (
            (_cid(f"minimized:{case.case_id}"),)
            if differential_outcome
            in set(sg.omission_supporting_outcomes())
            else ()
        ),
        "model_insufficiency_evidence_cids": (),
        "expanded_artifact_ids": tuple(sorted(expanded)),
        "coverage_manifest_cid": _cid(f"manifest:{case.case_id}"),
        "policy_cid": _cid("policy:scg-041"),
        "notes": None,
        "metadata": {"case_id": case.case_id},
    }


def _dependency_graph(
    case: Any,
    *,
    repo_cid: str,
    exclusions: Sequence[Any],
    inclusions: Sequence[Any],
) -> dict[str, Any]:
    node_map: dict[str, str] = {}
    paths: list[Any] = []
    for item in inclusions:
        if item.symbol_id:
            node_map[_sym_token(item.symbol_id)] = item.artifact_id
        if item.dependency_path is not None:
            paths.append(item.dependency_path)
    for item in exclusions:
        if item.symbol_id:
            node_map[_sym_token(item.symbol_id)] = item.artifact_id
        if item.dependency_path is not None:
            paths.append(item.dependency_path)
    return {
        "repository_state_cid": repo_cid,
        "paths": tuple(paths)
        or (_graph_path(_sym_token(case.scanner_view.primary_symbol)),),
        "node_artifact_ids": node_map,
        "notes": None,
        "metadata": {"case_id": case.case_id},
    }


def _hypotheses_from_case(case: Any, exclusions: Sequence[Any]) -> list[Any]:
    hyps: list[Any] = []
    critical_symbols = {
        _sym_token(symbol)
        for symbol in (
            set(case.omission.critical_omitted_symbols)
            | set(case.omission.expansion_targets)
        )
    }
    ranked = [
        item
        for item in exclusions
        if (item.symbol_id and _sym_token(item.symbol_id) in critical_symbols)
        or item.critical
    ]
    ranked = sorted(ranked, key=lambda item: item.artifact_id)
    pack_cid = _cid(f"pack:{case.case_id}:compressed")
    repo_cid = _cid(f"repo:{case.case_id}")

    for rank, item in enumerate(ranked):
        if case.outcome.expected_diagnosis == "stale_artifact":
            cause = sg.HypothesisCause.STALE_ARTIFACT
            action = sg.ExpansionAction.REQUEST_HUMAN_REVIEW
        elif case.outcome.expected_diagnosis in {"security", "confidence_error"}:
            cause = sg.HypothesisCause.POLICY_BOUNDARY
            action = sg.ExpansionAction.REQUEST_HUMAN_REVIEW
        else:
            cause = sg.HypothesisCause.OMISSION
            if case.family == "configuration":
                action = sg.ExpansionAction.INCLUDE_CONFIGURATION
            elif case.family == "fixture":
                action = sg.ExpansionAction.INCLUDE_FIXTURE
            elif case.family in {"schema_migration", "api_migration"}:
                action = sg.ExpansionAction.INCLUDE_SCHEMA
            else:
                action = sg.ExpansionAction.INCLUDE_RAW_SOURCE
        hyps.append(
            sg.OmissionHypothesis(
                header=_header(
                    "omission_hypothesis",
                    case_id=case.case_id,
                    repo_cid=repo_cid,
                    pack_cid=pack_cid,
                    interface_id="diagnose_omission@1",
                ),
                hypothesis_id=_token_id("hyp", case.case_id, item.artifact_id),
                cause=cause,
                subject_artifact_id=item.artifact_id,
                subject_kind=item.artifact_kind,
                rank=rank,
                expected_relevance_bp=9_000 if item.critical else 6_000,
                inclusion_cost_tokens=item.token_cost,
                confidence_bp=item.confidence_bp,
                expansion_action=action,
                exclusion_reason=item.exclusion_reason,
                capsule_class=(
                    "exact_capsule"
                    if item.exclusion_reason
                    == sg.ExclusionReason.EXACT_CAPSULE_SUBSTITUTED.value
                    else None
                ),
                path=item.path,
                source_span=item.source_span,
                dependency_path=item.dependency_path,
                supporting_evidence_cids=(_cid(f"counterexample:{case.case_id}"),),
                proposed_rule_change=(
                    "prefer_raw_source_for_critical_omitted_subjects"
                    if cause == sg.HypothesisCause.OMISSION
                    else None
                ),
                notes=None,
                metadata={"symbol_id": item.symbol_id},
            )
        )
    return hyps


def _recommended_action(claim: Any) -> str:
    return str(claim.metadata.get("recommended_action") or "")


def _is_auto_accept(claim: Any) -> bool:
    return (
        claim.sufficiency_state in _ACCEPTING_STATES
        and _recommended_action(claim) in _ACCEPTING_ACTIONS
    )


# ---------------------------------------------------------------------------
# Surface / partition wiring
# ---------------------------------------------------------------------------


def test_static_scenario_matrix_is_held_out_complete(
    static_adversarial_cases: Sequence[Any],
) -> None:
    assert len(static_adversarial_cases) == 11
    for case in static_adversarial_cases:
        assert case.omission is not None
        assert case.outcome is not None
        assert case.scanner_view is not None
        assert case.outcome.automatic_accept_allowed is False
        assert case.omission.intentional_critical is True
        assert case.omission.critical_omitted_symbols
        assert case.omission.expansion_targets


def test_public_governor_apis_are_bound() -> None:
    for name in (
        "evaluate_context_sufficiency",
        "diagnose_omission",
        "plan_context_expansion",
    ):
        assert name in sg.REQUIRED_PUBLIC_APIS
        assert callable(getattr(sg, name))


# ---------------------------------------------------------------------------
# Critical omissions detected before automatic acceptance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", STATIC_ADVERSARIAL_SCENARIOS)
def test_critical_omission_blocks_automatic_acceptance(
    static_adversarial_cases: Sequence[Any],
    scenario: str,
) -> None:
    case = next(
        item for item in static_adversarial_cases if item.adversarial_scenario == scenario
    )
    # Verification pass is present — still must not auto-accept.
    claim = _evaluate_case(case, include_critical=False, verification_passed=True)

    assert claim.sufficiency_state not in _ACCEPTING_STATES, (
        scenario,
        claim.sufficiency_state,
        claim.blocking_reason_codes,
    )
    assert not _is_auto_accept(claim), (scenario, _recommended_action(claim))
    assert _recommended_action(claim) not in _ACCEPTING_ACTIONS
    assert claim.blocking_reason_codes or claim.known_gap_ids

    # Map fixture oracle outcome to closed sufficiency states.
    expected = case.outcome.expected_outcome
    if expected == "reject_stale":
        assert claim.sufficiency_state == sg.ContextSufficiencyState.STALE.value
        assert _recommended_action(claim) == sg.DecisionAction.MARK_STALE.value
        assert any("stale" in code for code in claim.blocking_reason_codes)
    elif expected == "human_review_required":
        assert claim.sufficiency_state == (
            sg.ContextSufficiencyState.HUMAN_REVIEW_REQUIRED.value
        )
        assert _recommended_action(claim) == (
            sg.DecisionAction.REQUIRE_HUMAN_REVIEW.value
        )
    elif expected == "insufficient_omission":
        assert claim.sufficiency_state == (
            sg.ContextSufficiencyState.EXPANSION_REQUIRED.value
        )
        assert _recommended_action(claim) == (
            sg.DecisionAction.REQUIRE_EXPANSION.value
        )
    else:  # pragma: no cover - static matrix is closed
        raise AssertionError(f"unexpected static outcome {expected!r}")


def test_verification_pass_never_authorizes_critical_omission(
    static_adversarial_cases: Sequence[Any],
) -> None:
    for case in static_adversarial_cases:
        claim = _evaluate_case(case, verification_passed=True)
        assert claim.verification_passed is True
        assert claim.sufficiency_state not in _ACCEPTING_STATES
        # Structural evidence must remain present; verification is not sole basis.
        bases = set(claim.evidence_bases)
        assert bases != {sg.SufficiencyEvidenceBasis.VERIFICATION_PASS.value}


def test_stale_capsule_case_forces_raw_regeneration(
    static_adversarial_cases: Sequence[Any],
) -> None:
    case = next(
        item
        for item in static_adversarial_cases
        if item.adversarial_scenario == "stale_capsule"
    )
    claim = _evaluate_case(case)
    assert claim.sufficiency_state == sg.ContextSufficiencyState.STALE.value
    assert any("stale" in code for code in claim.blocking_reason_codes)
    assert sg.SufficiencyEvidenceBasis.FRESHNESS.value in claim.evidence_bases


def test_policy_and_confidence_cases_require_human_review(
    static_adversarial_cases: Sequence[Any],
) -> None:
    for scenario in ("confidence_misclassification", "security_invariant"):
        case = next(
            item
            for item in static_adversarial_cases
            if item.adversarial_scenario == scenario
        )
        claim = _evaluate_case(case)
        assert claim.sufficiency_state == (
            sg.ContextSufficiencyState.HUMAN_REVIEW_REQUIRED.value
        )
        assert _recommended_action(claim) == (
            sg.DecisionAction.REQUIRE_HUMAN_REVIEW.value
        )
        assert case.outcome.automatic_accept_allowed is False


# ---------------------------------------------------------------------------
# Omission diagnosis on structural adversarial cases
# ---------------------------------------------------------------------------


def test_omission_diagnosis_ranks_critical_subjects_for_structural_cases(
    static_adversarial_cases: Sequence[Any],
) -> None:
    omission_cases = [
        case
        for case in static_adversarial_cases
        if case.outcome.expected_outcome == "insufficient_omission"
        and case.outcome.expected_diagnosis == "omission"
    ]
    assert omission_cases

    for case in omission_cases:
        pack_cid = _cid(f"pack:{case.case_id}:compressed")
        repo_cid = _cid(f"repo:{case.case_id}")
        manifest = _build_compressed_manifest(
            case, repo_cid=repo_cid, pack_cid=pack_cid, include_critical=False
        )
        audit = _audit_case(case, pack_cid=pack_cid, repo_cid=repo_cid)
        repo_view = _omission_repo_view(
            case,
            pack_cid=pack_cid,
            repo_cid=repo_cid,
            exclusions=manifest.exclusions,
            differential_outcome=(
                sg.ComparativeOutcome.COMPRESSED_FAILED_EXPANDED_SUCCEEDED.value
            ),
        )
        graph = _dependency_graph(
            case,
            repo_cid=repo_cid,
            exclusions=manifest.exclusions,
            inclusions=manifest.inclusions,
        )
        result = sg.diagnose_omission(audit, repo_view, graph)
        assert result.ranked_omission_supported is True, case.case_id
        assert result.primary_cause == sg.PrimaryDiagnosisCause.OMISSION.value
        assert result.evidence is not None
        assert result.hypotheses
        subject_ids = {hyp.subject_artifact_id for hyp in result.hypotheses}
        critical_exc = {
            item.artifact_id for item in manifest.exclusions if item.critical
        }
        assert subject_ids & critical_exc, case.case_id
        # Hypotheses stay ranked (non-decreasing rank order already enforced).
        ranks = [hyp.rank for hyp in result.hypotheses]
        assert ranks == sorted(ranks)


def test_stale_and_policy_cases_do_not_claim_compression_omission_without_support(
    static_adversarial_cases: Sequence[Any],
) -> None:
    """Without expanded-success differential, compression is not blamed."""

    for scenario in (
        "stale_capsule",
        "confidence_misclassification",
        "security_invariant",
    ):
        case = next(
            item
            for item in static_adversarial_cases
            if item.adversarial_scenario == scenario
        )
        pack_cid = _cid(f"pack:{case.case_id}:compressed")
        repo_cid = _cid(f"repo:{case.case_id}")
        manifest = _build_compressed_manifest(
            case, repo_cid=repo_cid, pack_cid=pack_cid, include_critical=False
        )
        audit = _audit_case(case, pack_cid=pack_cid, repo_cid=repo_cid)
        repo_view = _omission_repo_view(
            case,
            pack_cid=pack_cid,
            repo_cid=repo_cid,
            exclusions=manifest.exclusions,
            differential_outcome=(
                sg.ComparativeOutcome.VERIFICATION_INCONCLUSIVE.value
            ),
        )
        graph = _dependency_graph(
            case,
            repo_cid=repo_cid,
            exclusions=manifest.exclusions,
            inclusions=manifest.inclusions,
        )
        result = sg.diagnose_omission(audit, repo_view, graph)
        assert result.ranked_omission_supported is False, scenario
        assert result.evidence is None
        assert result.primary_cause != sg.PrimaryDiagnosisCause.OMISSION.value


# ---------------------------------------------------------------------------
# Exact sufficient context is not needlessly expanded
# ---------------------------------------------------------------------------


def test_held_out_sufficient_cases_accept_without_expansion(
    held_out_sufficient_cases: Sequence[Any],
) -> None:
    sample = held_out_sufficient_cases[:8]
    for case in sample:
        claim = _evaluate_case(case, include_critical=True, verification_passed=False)
        assert claim.sufficiency_state in _ACCEPTING_STATES, (
            case.case_id,
            claim.sufficiency_state,
            claim.blocking_reason_codes,
        )
        assert _is_auto_accept(claim)
        assert claim.blocking_reason_codes == ()

        # No omission hypotheses → expansion plan is empty / no-op.
        pack_cid = _cid(f"pack:{case.case_id}:full")
        repo_cid = _cid(f"repo:{case.case_id}")
        audit = _audit_case(case, pack_cid=pack_cid, repo_cid=repo_cid)
        plan = sg.plan_context_expansion(audit, (), token_budget=500, max_steps=4)
        assert plan.step_count == 0
        assert plan.total_token_increase == 0
        assert plan.total_token_increase <= plan.max_token_growth


def test_repairing_critical_omission_removes_needless_expansion_pressure(
    static_adversarial_cases: Sequence[Any],
) -> None:
    """Including expansion targets yields sufficiency for pure-omission cases."""

    pure_omission = [
        case
        for case in static_adversarial_cases
        if case.outcome.expected_outcome == "insufficient_omission"
        and case.outcome.expected_diagnosis == "omission"
    ]
    assert pure_omission

    for case in pure_omission:
        compressed = _evaluate_case(case, include_critical=False)
        assert compressed.sufficiency_state == (
            sg.ContextSufficiencyState.EXPANSION_REQUIRED.value
        )

        repaired = _evaluate_case(case, include_critical=True, verification_passed=False)
        assert repaired.sufficiency_state in _ACCEPTING_STATES, (
            case.case_id,
            repaired.sufficiency_state,
            repaired.blocking_reason_codes,
        )
        assert _is_auto_accept(repaired)
        # Exact sufficient pack must not request expansion.
        assert _recommended_action(repaired) == (
            sg.DecisionAction.ACCEPT_COMPRESSED.value
        )


# ---------------------------------------------------------------------------
# Bounded expansion limits
# ---------------------------------------------------------------------------


def test_expansion_plan_respects_step_and_token_limits(
    static_adversarial_cases: Sequence[Any],
) -> None:
    limits = sg.default_expansion_limits()
    absolute_max_steps = limits["max_expansion_steps_absolute"]

    for case in static_adversarial_cases:
        if case.outcome.expected_outcome != "insufficient_omission":
            continue
        pack_cid = _cid(f"pack:{case.case_id}:compressed")
        repo_cid = _cid(f"repo:{case.case_id}")
        manifest = _build_compressed_manifest(
            case, repo_cid=repo_cid, pack_cid=pack_cid, include_critical=False
        )
        audit = _audit_case(case, pack_cid=pack_cid, repo_cid=repo_cid)
        hyps = _hypotheses_from_case(case, manifest.exclusions)
        if not hyps:
            continue

        token_budget = 80
        max_steps = 2
        result = sg.plan_context_expansion(
            audit,
            hyps,
            token_budget=token_budget,
            max_steps=max_steps,
            return_result=True,
        )
        plan = result.plan
        assert plan.step_count <= max_steps
        assert plan.step_count <= plan.max_steps
        assert plan.max_steps <= absolute_max_steps
        assert plan.total_token_increase <= token_budget
        assert plan.total_token_increase <= plan.max_token_growth
        assert plan.max_token_growth == token_budget
        # Only ranked subject artifacts — never a repository dump.
        for step in plan.steps:
            if step.action in sg.context_expansion_actions():
                assert step.artifact_ids_added
                for aid in step.artifact_ids_added:
                    assert aid.startswith("exc_") or aid.startswith("inc_")


def test_zero_budget_with_required_omission_is_not_unbounded(
    static_adversarial_cases: Sequence[Any],
) -> None:
    case = next(
        item
        for item in static_adversarial_cases
        if item.adversarial_scenario == "hidden_callee_side_effect"
    )
    pack_cid = _cid(f"pack:{case.case_id}:compressed")
    repo_cid = _cid(f"repo:{case.case_id}")
    manifest = _build_compressed_manifest(
        case, repo_cid=repo_cid, pack_cid=pack_cid, include_critical=False
    )
    audit = _audit_case(case, pack_cid=pack_cid, repo_cid=repo_cid)
    hyps = _hypotheses_from_case(case, manifest.exclusions)
    assert hyps

    result = sg.plan_context_expansion(
        audit, hyps, token_budget=0, max_steps=4, return_result=True
    )
    assert result.requires_human_review is True
    plan = result.plan
    assert plan.total_token_increase == 0
    assert plan.total_token_increase <= plan.max_token_growth
    assert plan.step_count >= 1
    assert all(
        step.action != sg.ExpansionAction.INCLUDE_RAW_SOURCE.value
        or step.token_delta == 0
        for step in plan.steps
    )
    # Prefer explicit human review over unbounded raw growth.
    assert any(
        step.action == sg.ExpansionAction.REQUEST_HUMAN_REVIEW.value
        for step in plan.steps
    ) or result.disposition == sg.ExpansionDisposition.HUMAN_REVIEW.value


def test_expansion_prefers_context_before_model_escalation(
    static_adversarial_cases: Sequence[Any],
) -> None:
    case = next(
        item
        for item in static_adversarial_cases
        if item.adversarial_scenario == "behavior_only_dependency"
    )
    pack_cid = _cid(f"pack:{case.case_id}:compressed")
    repo_cid = _cid(f"repo:{case.case_id}")
    manifest = _build_compressed_manifest(
        case, repo_cid=repo_cid, pack_cid=pack_cid, include_critical=False
    )
    audit = _audit_case(case, pack_cid=pack_cid, repo_cid=repo_cid)
    hyps = _hypotheses_from_case(case, manifest.exclusions)
    # Append an explicit route-escalation hypothesis after omission repair.
    hyps.append(
        sg.OmissionHypothesis(
            header=_header(
                "omission_hypothesis",
                case_id=case.case_id,
                repo_cid=repo_cid,
                pack_cid=pack_cid,
                interface_id="diagnose_omission@1",
            ),
            hypothesis_id=_token_id("hyp", case.case_id, "model_route"),
            cause=sg.HypothesisCause.MODEL_INSUFFICIENCY,
            subject_artifact_id=_token_id("route", case.case_id),
            subject_kind=sg.CoveredArtifactKind.SYMBOL,
            rank=len(hyps),
            expected_relevance_bp=5_000,
            inclusion_cost_tokens=0,
            confidence_bp=6_000,
            expansion_action=sg.ExpansionAction.ESCALATE_ROUTE,
            exclusion_reason=None,
            capsule_class=None,
            path=None,
            source_span=None,
            dependency_path=None,
            supporting_evidence_cids=(_cid(f"model:{case.case_id}"),),
            proposed_rule_change="escalate_route_after_context_expansion",
            notes=None,
            metadata={"route_hypothesis": True},
        )
    )
    result = sg.plan_context_expansion(
        audit, hyps, token_budget=200, max_steps=6, return_result=True
    )
    assert result.context_before_model_escalation is True
    actions = [step.action for step in result.plan.steps]
    context_actions = set(sg.context_expansion_actions())
    if any(action in context_actions for action in actions) and (
        sg.ExpansionAction.ESCALATE_ROUTE.value in actions
    ):
        first_context = min(
            i for i, action in enumerate(actions) if action in context_actions
        )
        first_escalation = actions.index(sg.ExpansionAction.ESCALATE_ROUTE.value)
        assert first_context < first_escalation


def test_expansion_identity_is_deterministic(
    static_adversarial_cases: Sequence[Any],
) -> None:
    case = next(
        item
        for item in static_adversarial_cases
        if item.adversarial_scenario == "serializer"
    )
    pack_cid = _cid(f"pack:{case.case_id}:compressed")
    repo_cid = _cid(f"repo:{case.case_id}")
    manifest = _build_compressed_manifest(
        case, repo_cid=repo_cid, pack_cid=pack_cid, include_critical=False
    )
    audit = _audit_case(case, pack_cid=pack_cid, repo_cid=repo_cid)
    hyps = _hypotheses_from_case(case, manifest.exclusions)
    a = sg.plan_context_expansion(audit, hyps, token_budget=100, max_steps=3)
    b = sg.plan_context_expansion(audit, hyps, token_budget=100, max_steps=3)
    assert a.plan_cid == b.plan_cid
    assert a.step_count == b.step_count
    assert a.total_token_increase == b.total_token_increase


def test_sufficiency_identity_is_deterministic(
    static_adversarial_cases: Sequence[Any],
) -> None:
    case = static_adversarial_cases[0]
    a = _evaluate_case(case)
    b = _evaluate_case(case)
    assert a.claim_cid == b.claim_cid
    assert a.sufficiency_state == b.sufficiency_state


# ---------------------------------------------------------------------------
# Fixture materialisation remains oracle-only (no forbidden artifacts)
# ---------------------------------------------------------------------------


def test_static_cases_materialize_without_forbidden_artifacts(
    corpus: Any,
    static_adversarial_cases: Sequence[Any],
    fixture_pkg: ModuleType,
    tmp_path: Path,
) -> None:
    forbidden = (
        "model_output",
        "completion_receipt",
        "state.db",
        "duckdb",
        "provider_response",
    )
    for case in static_adversarial_cases:
        root = tmp_path / case.case_id.replace(".", "_")
        corpus.materialize_case(case.case_id, root)
        tree = fixture_pkg.read_tree_bytes(root)
        assert tree
        for rel, payload in tree.items():
            blob = f"{rel}\n{payload.decode('utf-8', errors='replace')}".lower()
            for marker in forbidden:
                assert marker not in blob, (case.case_id, rel, marker)
        # Scanner oracle paths must exist after materialisation.
        for path in case.scanner_view.changed_paths:
            assert path in tree, (case.case_id, path)


def test_no_static_case_allows_auto_accept_under_oracle(
    static_adversarial_cases: Sequence[Any],
) -> None:
    """Oracle-level invariant: every static adversarial case forbids auto-accept."""

    for case in static_adversarial_cases:
        assert case.outcome.automatic_accept_allowed is False
        assert case.omission.intentional_critical is True
        # Governor evaluation agrees.
        claim = _evaluate_case(case, verification_passed=True)
        assert not _is_auto_accept(claim)
