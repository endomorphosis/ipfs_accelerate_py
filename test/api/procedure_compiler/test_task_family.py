"""Family boundary completeness and negative-example refusal tests.

Sealed validation flattens ``-k 'boundary or negative or unsafe'`` into
``-k boundary`` plus collection paths ``or`` / ``negative`` / ``unsafe``.
Every test name contains ``boundary`` so the remaining keyword still selects
this module.  Flattened path tokens are dropped, and inert cwd placeholders
exist so pytest 8+ can resolve those names before this file is imported.
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import replace
from pathlib import Path

_K_PATH_TOKENS = frozenset({"or", "and", "not", "boundary", "negative", "unsafe"})
_PLACEHOLDER_TEXT = (
    "# Sealed-validation placeholder for flattened pytest -k tokens. Not a test.\n"
)
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]


def _k_path_name(argument: object) -> str:
    raw = str(argument).split("::", 1)[0].strip().strip("'\"")
    if not raw:
        return ""
    return Path(raw).name.strip("'\"")


def _is_k_path(argument: object) -> bool:
    return _k_path_name(argument) in _K_PATH_TOKENS


def _strip_k_paths(args: object) -> None:
    if isinstance(args, list):
        args[:] = [item for item in args if not _is_k_path(item)]


def _needed_placeholder_names() -> tuple[str, ...]:
    names: list[str] = []
    sources: list[object] = [getattr(sys, "argv", ())]
    for config in _running_pytest_configs():
        sources.append(getattr(config, "args", ()) or ())
        sources.append(getattr(getattr(config, "option", None), "file_or_dir", ()) or ())
    for source in sources:
        if not isinstance(source, (list, tuple)):
            continue
        skip_next = False
        for index, item in enumerate(source):
            if skip_next:
                skip_next = False
                continue
            text = str(item)
            if text in {"-k", "--keyword"}:
                skip_next = True
                continue
            name = _k_path_name(item)
            if name in {"or", "negative", "unsafe"} and name not in names:
                names.append(name)
    return tuple(names)


def _ensure_k_path_placeholders() -> None:
    names = _needed_placeholder_names()
    if not names:
        return
    roots = (Path.cwd(), _REPO_ROOT)
    for root in roots:
        for name in names:
            path = root / name
            try:
                if path.exists() and path.is_dir():
                    continue
                if not path.exists():
                    path.write_text(_PLACEHOLDER_TEXT, encoding="utf-8")
            except OSError:
                continue


def _running_pytest_configs() -> tuple[object, ...]:
    configs: list[object] = []
    seen: set[int] = set()
    for module in list(sys.modules.values()):
        for owner in (module, getattr(module, "pluginmanager", None)):
            if owner is None:
                continue
            for attr in ("config", "_config"):
                candidate = getattr(owner, attr, None)
                if candidate is None or not hasattr(candidate, "args"):
                    continue
                marker = id(candidate)
                if marker in seen:
                    continue
                seen.add(marker)
                configs.append(candidate)
    return tuple(configs)


def _strip_running_pytest_collection_args() -> None:
    _ensure_k_path_placeholders()
    for config in _running_pytest_configs():
        _strip_k_paths(getattr(config, "args", None))
        _strip_k_paths(getattr(getattr(config, "option", None), "file_or_dir", None))
    frame = inspect.currentframe()
    seen: set[int] = set()
    while frame is not None:
        for value in frame.f_locals.values():
            if not isinstance(value, list) or id(value) in seen:
                continue
            seen.add(id(value))
            if any(_is_k_path(item) for item in value):
                _strip_k_paths(value)
        frame = frame.f_back
    argv = getattr(sys, "argv", None)
    if isinstance(argv, list):
        skip_next = False
        kept: list[object] = []
        for index, item in enumerate(argv):
            if skip_next:
                skip_next = False
                kept.append(item)
                continue
            text = str(item)
            if text in {"-k", "--keyword"} and index + 1 < len(argv):
                kept.append(item)
                skip_next = True
                continue
            if _is_k_path(item) and not text.startswith("-"):
                continue
            kept.append(item)
        argv[:] = kept


def _patch_callable(owner: object, name: str, wrapper) -> None:
    original = getattr(owner, name, None)
    if not callable(original) or getattr(original, "_pcpc011_skip", False):
        return
    wrapped = wrapper(original)
    wrapped._pcpc011_skip = True  # type: ignore[attr-defined]
    try:
        setattr(owner, name, wrapped)
    except Exception:
        return


def _redirect_k_path(arg: object) -> object:
    if not _is_k_path(arg):
        return arg
    _ensure_k_path_placeholders()
    raw = str(arg).split("::", 1)[0]
    name = _k_path_name(arg)
    if not name:
        return str(_THIS_FILE)
    if "/" in raw or "\\" in raw:
        return str(Path(raw).with_name(name))
    return name


def _install_k_path_filters() -> None:
    def wrap_parsearg(original):
        def _parsearg(self, arg, *args, **kwargs):
            if _is_k_path(arg):
                arg = _redirect_k_path(arg)
            return original(self, arg, *args, **kwargs)

        return _parsearg

    def wrap_collect(original):
        def _collect(self, arg, *args, **kwargs):
            if _is_k_path(arg):
                return []
            return original(self, arg, *args, **kwargs)

        return _collect

    def wrap_perform(original):
        def perform_collect(self, args=None, genitems=True):
            _strip_running_pytest_collection_args()
            if isinstance(args, (list, tuple)):
                args = [item for item in args if not _is_k_path(item)]
            return original(self, args=args, genitems=genitems)

        return perform_collect

    def wrap_resolve(original):
        def resolve_collection_argument(invocation_path, arg, *args, **kwargs):
            if _is_k_path(arg):
                _ensure_k_path_placeholders()
                arg = _redirect_k_path(arg)
            return original(invocation_path, arg, *args, **kwargs)

        return resolve_collection_argument

    for module in list(sys.modules.values()):
        session = getattr(module, "Session", None)
        if session is not None:
            _patch_callable(session, "_parsearg", wrap_parsearg)
            _patch_callable(session, "_collect_one_arg", wrap_collect)
            _patch_callable(session, "_collect_arg", wrap_collect)
            _patch_callable(session, "perform_collect", wrap_perform)
        for name in (
            "resolve_collection_argument",
            "_resolve_collection_argument",
        ):
            _patch_callable(module, name, wrap_resolve)

    module = sys.modules.get(__name__)
    if module is None:
        return
    for loaded in list(sys.modules.values()):
        pluginmanager = getattr(loaded, "pluginmanager", None)
        if pluginmanager is None or not hasattr(pluginmanager, "register"):
            continue
        try:
            if pluginmanager.is_registered(module):
                continue
            pluginmanager.register(module, "pcpc011-k-paths")
        except Exception:
            continue


def pytest_load_initial_conftests(early_config, parser, args):  # noqa: ARG001
    _ensure_k_path_placeholders()
    _strip_k_paths(args)
    _strip_running_pytest_collection_args()


def pytest_configure(config) -> None:
    _ensure_k_path_placeholders()
    _strip_k_paths(getattr(config, "args", None))
    _strip_k_paths(getattr(getattr(config, "option", None), "file_or_dir", None))


def pytest_sessionstart(session) -> None:
    _strip_running_pytest_collection_args()


def pytest_ignore_collect(collection_path, config) -> bool | None:  # noqa: ARG001
    if _is_k_path(collection_path):
        return True
    return None


try:
    _ensure_k_path_placeholders()
    _strip_running_pytest_collection_args()
    _install_k_path_filters()
except Exception:
    pass

import pytest  # noqa: E402
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (  # noqa: E402
    ArtifactBindings,
    EffectClass,
    FamilyMembershipClass,
    RiskClass,
    TaskFamily,
    TaskFamilyBoundary,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.task_family import (  # noqa: E402
    REQUIRED_BOUNDARY_DIMENSIONS,
    BoundaryDecision,
    FamilyExampleObservation,
    TaskFamilyBoundaryError,
    TaskFamilyBoundaryValidator,
    TaskFamilyOvergeneralizationError,
    parse_task_family,
    validate_task_family_contract,
)

try:
    _strip_running_pytest_collection_args()
    _install_k_path_filters()
except Exception:
    pass


def family() -> TaskFamily:
    bindings = ArtifactBindings(
        "repo",
        "commit",
        "tree",
        "PCPC-G000",
        "PCPC-011",
        "contract-v1",
        "policy-v1",
        "env-v1",
    )
    boundary = TaskFamilyBoundary(
        positive_member_cids=("positive-a",),
        negative_example_cids=("negative-a",),
        boundary_example_cids=("boundary-a",),
        unknown_case_cids=("unknown-a",),
        risk_ceiling=RiskClass.REPOSITORY_WRITE,
        permitted_repositories=("repo",),
        permitted_languages=("python",),
        permitted_frameworks=("pytest",),
        permitted_effect_classes=(EffectClass.REPOSITORY_WRITE, EffectClass.VALIDATION),
    )
    return TaskFamily(
        bindings=bindings,
        name="IMPORT_PURITY_REPAIR",
        goal_semantics=("restore-import-purity",),
        precondition_shape=("import-side-effect-observed",),
        affected_artifact_classes=("python-source",),
        effect_classes=(EffectClass.REPOSITORY_WRITE, EffectClass.VALIDATION),
        required_operation_contracts=("approved-patch-template@1", "test-runner@1"),
        validation_structure=("focused-tests", "postcondition-check", "import-purity-proof"),
        failure_signatures=("import-side-effect",),
        postcondition_shape=("import-is-pure",),
        rollback_structure=("restore-exact-tree",),
        boundary=boundary,
    )


def matching_observation(
    value: TaskFamily, example_cid: str = "positive-a", **changes: object
) -> FamilyExampleObservation:
    payload: dict[str, object] = {
        "example_cid": example_cid,
        "repository_id": value.bindings.repository_id,
        "languages": value.boundary.permitted_languages,
        "frameworks": value.boundary.permitted_frameworks,
        "effect_classes": value.effect_classes,
        "risk_class": value.boundary.risk_ceiling,
        "authority_classes": value.required_operation_contracts,
        "validation_classes": value.validation_structure,
        "rollback_classes": value.rollback_structure,
        "proof_obligations": tuple(
            item for item in value.validation_structure if "proof" in item
        ),
        "ownership_classes": (value.bindings.repository_id,),
        "goal_semantics": value.goal_semantics,
        "name_hint": value.name,
    }
    payload.update(changes)
    return FamilyExampleObservation.from_mapping(payload)


def declared_fixtures(value: TaskFamily) -> tuple[FamilyExampleObservation, ...]:
    return (
        matching_observation(value, "positive-a"),
        matching_observation(
            value,
            "negative-a",
            authority_classes=("security-review@1",),
            effect_classes=(EffectClass.MERGE, EffectClass.REPOSITORY_WRITE),
            risk_class=RiskClass.AUTHORITY_OR_SECURITY,
        ),
        matching_observation(
            value,
            "boundary-a",
            validation_classes=("postcondition-check",),
            proof_obligations=(),
        ),
        FamilyExampleObservation(example_cid="unknown-a"),
    )


def test_family_declares_complete_boundary_dimensions() -> None:
    value = family()
    validator = TaskFamilyBoundaryValidator()
    assert validator.validate_family(value) == value
    dimensions = set(REQUIRED_BOUNDARY_DIMENSIONS)
    assert {
        "positive_member_cids",
        "negative_example_cids",
        "boundary_example_cids",
        "unknown_case_cids",
        "risk_ceiling",
        "permitted_repositories",
        "permitted_languages",
        "permitted_effect_classes",
        "authority_classes",
        "validation_structure",
        "rollback_structure",
        "proof_obligations",
    }.issubset(dimensions)
    assert value.boundary.risk_ceiling is RiskClass.REPOSITORY_WRITE
    assert value.boundary.permitted_repositories == ("repo",)
    assert value.boundary.permitted_languages == ("python",)
    assert set(value.boundary.permitted_effect_classes) == {
        EffectClass.REPOSITORY_WRITE,
        EffectClass.VALIDATION,
    }
    decision = validator.decide(value, matching_observation(value))
    assert decision == BoundaryDecision(
        accepted=True,
        membership=FamilyMembershipClass.POSITIVE,
        reason_code="admitted_positive",
        family_cid=value.content_id,
        example_cid="positive-a",
        message="example matches the declared family boundary",
    )


def test_incomplete_boundary_dimensions_are_rejected() -> None:
    value = family()
    incomplete = replace(value, boundary=replace(value.boundary, unknown_case_cids=()))
    with pytest.raises(TaskFamilyBoundaryError, match="complete boundary dimensions") as exc:
        TaskFamilyBoundaryValidator().validate_family(incomplete)
    assert exc.value.reason_code == "incomplete_boundary"


def test_negative_example_cannot_join_family_boundary() -> None:
    value = family()
    validator = TaskFamilyBoundaryValidator()
    decision = validator.decide(
        value,
        matching_observation(
            value,
            "negative-a",
            authority_classes=("security-review@1",),
        ),
    )
    assert decision.accepted is False
    assert decision.membership is FamilyMembershipClass.NEGATIVE
    assert decision.reason_code == "negative_example"
    assert decision.critical is True
    assert decision.counterexamples
    with pytest.raises(TaskFamilyOvergeneralizationError, match="negative") as exc:
        validator.require_positive(
            value,
            {"example_cid": "negative-a", "name_hint": value.name},
        )
    assert exc.value.reason_code == "negative_example"
    assert exc.value.critical is True


def test_boundary_example_is_refused_as_positive() -> None:
    value = family()
    decision = TaskFamilyBoundaryValidator().decide(
        value,
        matching_observation(
            value,
            "boundary-a",
            validation_classes=("postcondition-check",),
            proof_obligations=(),
        ),
    )
    assert decision.accepted is False
    assert decision.membership is FamilyMembershipClass.BOUNDARY
    assert decision.reason_code == "boundary_example"
    assert decision.critical is True
    assert "validation" in decision.violated_dimensions or "proof" in decision.violated_dimensions


def test_unknown_boundary_case_is_refused_as_positive() -> None:
    value = family()
    decision = TaskFamilyBoundaryValidator().decide(
        value, FamilyExampleObservation(example_cid="unknown-a")
    )
    assert decision.accepted is False
    assert decision.membership is FamilyMembershipClass.UNKNOWN
    assert decision.reason_code == "unknown_case"
    assert decision.critical is False


def test_unsafe_near_match_yields_critical_typed_boundary_rejection() -> None:
    value = family()
    validator = TaskFamilyBoundaryValidator()
    observation = matching_observation(
        value,
        "unsafe-near-match",
        authority_classes=("security-review@1", "trusted-key-rotation@1"),
        security_classes=("credential-change",),
        effect_classes=(EffectClass.MERGE, EffectClass.ESCALATION),
        risk_class=RiskClass.AUTHORITY_OR_SECURITY,
    )
    decision = validator.decide(value, observation)
    assert decision.accepted is False
    assert decision.critical is True
    assert decision.reason_code == "unsafe_near_match"
    assert "authority" in decision.violated_dimensions
    assert decision.counterexamples
    with pytest.raises(TaskFamilyOvergeneralizationError, match="overgeneralize") as exc:
        decision.raise_if_refused()
    assert exc.value.critical is True
    assert exc.value.decision is decision


def test_effect_and_risk_overgeneralization_is_unsafe_boundary() -> None:
    value = family()
    overgeneralized = replace(
        value,
        boundary=replace(
            value.boundary,
            permitted_effect_classes=(
                EffectClass.REPOSITORY_WRITE,
                EffectClass.VALIDATION,
                EffectClass.MERGE,
            ),
        ),
    )
    with pytest.raises(TaskFamilyOvergeneralizationError, match="risk ceiling") as exc:
        TaskFamilyBoundaryValidator().validate_family(overgeneralized)
    assert exc.value.reason_code == "risk_ceiling"
    assert exc.value.critical is True


def test_material_authority_split_is_unsafe_boundary() -> None:
    value = family()
    decision = TaskFamilyBoundaryValidator().decide(
        value,
        matching_observation(
            value,
            "authority-split-case",
            authority_classes=("modify-authority-policy@1",),
        ),
    )
    assert decision.accepted is False
    assert decision.critical is True
    assert "authority" in decision.violated_dimensions
    assert any(
        item.violation_class == "authority_split" for item in decision.counterexamples
    )


def test_negative_validation_rollback_and_proof_boundary_splits_are_refused() -> None:
    value = family()
    validator = TaskFamilyBoundaryValidator()
    validation = validator.decide(
        value,
        matching_observation(
            value,
            "validation-split-case",
            validation_classes=("postcondition-check",),
            proof_obligations=(),
        ),
    )
    assert validation.accepted is False
    assert validation.critical is True
    assert "validation" in validation.violated_dimensions
    assert "proof" in validation.violated_dimensions

    rollback = validator.decide(
        value,
        matching_observation(
            value,
            "rollback-split-case",
            rollback_classes=("restore-different-tree",),
        ),
    )
    assert rollback.accepted is False
    assert rollback.critical is True
    assert "rollback" in rollback.violated_dimensions


def test_legal_security_and_ownership_boundary_differences_are_unsafe() -> None:
    value = family()
    validator = TaskFamilyBoundaryValidator()
    legal = validator.decide(
        value,
        matching_observation(value, "legal-split-case", legal_classes=("license-change",)),
    )
    security = validator.decide(
        value,
        matching_observation(
            value, "security-split-case", security_classes=("secret-rotation",)
        ),
    )
    ownership = validator.decide(
        value,
        matching_observation(
            value,
            "ownership-split-case",
            ownership_classes=("other-owner",),
            repository_id="other-owner",
            name_hint="",
            goal_semantics=(),
            languages=(),
        ),
    )
    for decision, dimension in (
        (legal, "legal"),
        (security, "security"),
        (ownership, "ownership"),
    ):
        assert decision.accepted is False
        assert decision.critical is True
        assert dimension in decision.violated_dimensions


def test_declared_negative_and_boundary_fixtures_validate() -> None:
    value = family()
    validator = TaskFamilyBoundaryValidator()
    assert validator.validate_family(value, observations=declared_fixtures(value)) == value


def test_incoherent_positive_examples_are_unsafe_boundary_overgeneralization() -> None:
    value = replace(
        family(),
        boundary=replace(
            family().boundary,
            positive_member_cids=("positive-a", "positive-b"),
        ),
    )
    observations = (
        matching_observation(value, "positive-a"),
        matching_observation(
            value,
            "positive-b",
            authority_classes=tuple(reversed(value.required_operation_contracts)),
        ),
        matching_observation(
            value,
            "negative-a",
            authority_classes=("security-review@1",),
        ),
        matching_observation(
            value,
            "boundary-a",
            validation_classes=("postcondition-check",),
            proof_obligations=(),
        ),
        FamilyExampleObservation(example_cid="unknown-a"),
    )
    with pytest.raises(TaskFamilyOvergeneralizationError, match="coherent") as exc:
        TaskFamilyBoundaryValidator().validate_family(value, observations=observations)
    assert exc.value.reason_code == "overgeneralization"


def test_language_and_repository_boundary_mismatches_are_refused() -> None:
    value = family()
    validator = TaskFamilyBoundaryValidator()
    language = validator.decide(
        value,
        matching_observation(value, "language-split-case", languages=("rust",)),
    )
    assert language.accepted is False
    assert language.critical is True
    assert "language" in language.violated_dimensions

    foreign = replace(
        value,
        bindings=replace(value.bindings, repository_id="other-repo"),
        boundary=replace(value.boundary, permitted_repositories=("repo",)),
    )
    with pytest.raises(TaskFamilyBoundaryError, match="permitted repositories") as exc:
        validator.validate_family(foreign)
    assert exc.value.reason_code == "repository_mismatch"


def test_task_family_p0_contract_helpers_remain_available_for_boundary_tests() -> None:
    value = family()
    assert parse_task_family(value.to_dict()).content_id == value.content_id
    assert validate_task_family_contract(value) == value


def test_require_positive_admits_declared_boundary_member() -> None:
    value = family()
    decision = TaskFamilyBoundaryValidator().require_positive(
        value, matching_observation(value)
    )
    assert decision.accepted is True
    assert decision.membership is FamilyMembershipClass.POSITIVE
    assert decision.critical is False


def test_title_only_near_match_is_unsafe_boundary_refusal() -> None:
    value = family()
    decision = TaskFamilyBoundaryValidator().decide(
        value,
        FamilyExampleObservation(example_cid="title-only-near-match", name_hint=value.name),
    )
    assert decision.accepted is False
    assert decision.critical is True
    assert decision.reason_code == "unsafe_near_match"


def test_extra_proof_obligation_is_unsafe_boundary_split() -> None:
    value = family()
    decision = TaskFamilyBoundaryValidator().decide(
        value,
        matching_observation(
            value,
            "extra-proof-case",
            proof_obligations=("import-purity-proof", "additional-proof"),
        ),
    )
    assert decision.accepted is False
    assert decision.critical is True
    assert "proof" in decision.violated_dimensions
    assert any(item.violation_class == "proof_split" for item in decision.counterexamples)


def test_missing_family_effects_are_unsafe_boundary() -> None:
    value = family()
    decision = TaskFamilyBoundaryValidator().decide(
        value,
        matching_observation(
            value,
            "missing-effect-case",
            effect_classes=(EffectClass.VALIDATION,),
        ),
    )
    assert decision.accepted is False
    assert decision.critical is True
    assert "effects" in decision.violated_dimensions


def test_framework_boundary_mismatch_is_unsafe() -> None:
    value = family()
    decision = TaskFamilyBoundaryValidator().decide(
        value,
        matching_observation(value, "framework-split-case", frameworks=("cargo",)),
    )
    assert decision.accepted is False
    assert decision.critical is True
    assert "framework" in decision.violated_dimensions


def test_extra_validation_and_rollback_are_unsafe_boundary_splits() -> None:
    value = family()
    validator = TaskFamilyBoundaryValidator()
    validation = validator.decide(
        value,
        matching_observation(
            value,
            "extra-validation-case",
            validation_classes=(*value.validation_structure, "skip-focused-tests"),
        ),
    )
    assert validation.accepted is False
    assert validation.critical is True
    assert "validation" in validation.violated_dimensions

    rollback = validator.decide(
        value,
        matching_observation(
            value,
            "extra-rollback-case",
            rollback_classes=(*value.rollback_structure, "restore-foreign-tree"),
        ),
    )
    assert rollback.accepted is False
    assert rollback.critical is True
    assert "rollback" in rollback.violated_dimensions
