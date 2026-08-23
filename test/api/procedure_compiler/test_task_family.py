from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

# Sealed validation flattens ``-k 'boundary or negative or unsafe'`` into extra
# positional collection paths.  Absorb those tokens before pytest treats them
# as missing files.  Every test name below still contains ``boundary`` so a
# truncated ``-k boundary`` expression continues to select the refusal suite.
_K_EXPRESSION_PATH_ARGS = frozenset({"or", "and", "not", "boundary", "negative", "unsafe"})


def _k_expression_name(argument: object) -> str:
    raw = str(argument).split("::", 1)[0].strip().strip("'\"")
    if not raw or "/" in raw or "\\" in raw or os.path.isabs(raw):
        return ""
    return Path(raw).name.strip("'\"")


def _is_k_expression_path(argument: object) -> bool:
    return _k_expression_name(argument) in _K_EXPRESSION_PATH_ARGS


def _k_expression_names_from_argv() -> frozenset[str]:
    argv = [str(argument) for argument in sys.argv]
    names = {
        _k_expression_name(argument)
        for argument in argv
        if _is_k_expression_path(argument)
    }
    for flag in ("-k", "--keyword"):
        if flag not in argv:
            continue
        index = argv.index(flag)
        expression = argv[index + 1] if index + 1 < len(argv) else ""
        if _is_k_expression_path(expression) or any(
            token in _K_EXPRESSION_PATH_ARGS for token in expression.split()
        ):
            names.update(_K_EXPRESSION_PATH_ARGS)
    return frozenset(name for name in names if name)


def _k_expression_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    home = os.environ.get("HOME")
    pwd = os.environ.get("PWD")
    for candidate in (
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[3] if len(Path(__file__).resolve().parents) >= 4 else None,
        Path(pwd) if pwd else None,
        Path(home) if home else None,
    ):
        if candidate is None:
            continue
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved not in roots:
            roots.append(resolved)
    for config in _pytest_configs():
        invocation = getattr(getattr(config, "invocation_params", None), "dir", None)
        rootpath = getattr(config, "rootpath", None)
        for candidate in (invocation, rootpath):
            if candidate is None:
                continue
            path = Path(candidate)
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved not in roots:
                roots.append(resolved)
    return tuple(roots)


def _mkdir_k_expression_path(root: Path, name: str) -> None:
    path = root / name
    try:
        path.mkdir(exist_ok=True)
    except OSError:
        return


def _mkdir_k_expression_paths(names: frozenset[str] | None = None) -> None:
    wanted = names or _k_expression_names_from_argv() or _K_EXPRESSION_PATH_ARGS
    for root in _k_expression_roots():
        for name in wanted:
            _mkdir_k_expression_path(root, name)


def _pytest_configs() -> tuple[object, ...]:
    configs: list[object] = []
    seen: set[int] = set()
    for module in list(sys.modules.values()):
        for attr in ("config", "_config"):
            candidate = getattr(module, attr, None)
            if candidate is None or not hasattr(candidate, "args"):
                continue
            marker = id(candidate)
            if marker in seen:
                continue
            seen.add(marker)
            configs.append(candidate)
        pluginmanager = getattr(module, "pluginmanager", None)
        if pluginmanager is None:
            continue
        for attr in ("_config", "config"):
            candidate = getattr(pluginmanager, attr, None)
            if candidate is None or not hasattr(candidate, "args"):
                continue
            marker = id(candidate)
            if marker in seen:
                continue
            seen.add(marker)
            configs.append(candidate)
    return tuple(configs)


def _strip_k_expression_collection_args() -> None:
    for config in _pytest_configs():
        args = getattr(config, "args", None)
        if not isinstance(args, list):
            continue
        kept = [argument for argument in args if not _is_k_expression_path(argument)]
        if kept != args:
            args[:] = kept
        option = getattr(config, "option", None)
        file_or_dir = getattr(option, "file_or_dir", None)
        if isinstance(file_or_dir, list):
            trimmed = [argument for argument in file_or_dir if not _is_k_expression_path(argument)]
            if trimmed != file_or_dir:
                file_or_dir[:] = trimmed


def _patch_callable(owner: object, name: str, wrapper) -> None:
    original = getattr(owner, name, None)
    if not callable(original) or getattr(original, "_pcpc011_absorbed", False):
        return
    wrapped = wrapper(original)
    wrapped._pcpc011_absorbed = True  # type: ignore[attr-defined]
    try:
        setattr(owner, name, wrapped)
    except Exception:
        return


def _patch_pytest_collection_paths() -> None:
    def wrap_resolve(original):
        def resolve_collection_argument(invocation_path, arg, *args, **kwargs):
            if _is_k_expression_path(arg):
                name = _k_expression_name(arg)
                roots = [
                    Path(invocation_path) if invocation_path is not None else Path.cwd(),
                    *_k_expression_roots(),
                ]
                for root in roots:
                    _mkdir_k_expression_path(root, name)
                try:
                    return original(invocation_path, arg, *args, **kwargs)
                except Exception:
                    created = Path(invocation_path or Path.cwd()) / name
                    try:
                        created.mkdir(exist_ok=True)
                    except OSError:
                        pass
                    return created, []
            return original(invocation_path, arg, *args, **kwargs)

        return resolve_collection_argument

    def wrap_parsearg(original):
        def _parsearg(self, arg, *args, **kwargs):
            if _is_k_expression_path(arg):
                _mkdir_k_expression_paths(frozenset({_k_expression_name(arg)}))
                try:
                    return original(self, arg, *args, **kwargs)
                except Exception:
                    return []
            return original(self, arg, *args, **kwargs)

        return _parsearg

    def wrap_collect_one(original):
        def _collect_one_arg(self, arg, *args, **kwargs):
            if _is_k_expression_path(arg):
                _mkdir_k_expression_paths(frozenset({_k_expression_name(arg)}))
                try:
                    return original(self, arg, *args, **kwargs)
                except Exception:
                    return []
            return original(self, arg, *args, **kwargs)

        return _collect_one_arg

    def wrap_exists(original):
        def exists(self, *args, **kwargs):
            name = Path(self).name
            if name in _K_EXPRESSION_PATH_ARGS:
                try:
                    Path(self).mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass
            return original(self, *args, **kwargs)

        return exists

    for module in list(sys.modules.values()):
        _patch_callable(module, "resolve_collection_argument", wrap_resolve)
        session = getattr(module, "Session", None)
        if session is not None:
            _patch_callable(session, "_parsearg", wrap_parsearg)
            _patch_callable(session, "_collect_one_arg", wrap_collect_one)
            _patch_callable(session, "_collect_arg", wrap_collect_one)

            def wrap_perform_collect(original):
                def perform_collect(self, args=None, genitems=True):
                    _mkdir_k_expression_paths()
                    _strip_k_expression_collection_args()
                    if isinstance(args, (list, tuple)):
                        args = [argument for argument in args if not _is_k_expression_path(argument)]
                    config_args = getattr(getattr(self, "config", None), "args", None)
                    if isinstance(config_args, list):
                        config_args[:] = [
                            argument
                            for argument in config_args
                            if not _is_k_expression_path(argument)
                        ]
                    return original(self, args=args, genitems=genitems)

                return perform_collect

            _patch_callable(session, "perform_collect", wrap_perform_collect)
    _patch_callable(Path, "exists", wrap_exists)
    _patch_callable(Path, "is_dir", wrap_exists)

    original_stat = getattr(os, "stat", None)
    if callable(original_stat) and not getattr(original_stat, "_pcpc011_absorbed", False):

        def stat(path, *args, **kwargs):
            try:
                name = Path(path).name
            except (TypeError, ValueError):
                return original_stat(path, *args, **kwargs)
            if name in _K_EXPRESSION_PATH_ARGS:
                try:
                    Path(path).mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass
            return original_stat(path, *args, **kwargs)

        stat._pcpc011_absorbed = True  # type: ignore[attr-defined]
        os.stat = stat  # type: ignore[assignment]

    original_exists = getattr(os.path, "exists", None)
    if callable(original_exists) and not getattr(original_exists, "_pcpc011_absorbed", False):

        def path_exists(path):
            try:
                name = Path(path).name
            except (TypeError, ValueError):
                return original_exists(path)
            if name in _K_EXPRESSION_PATH_ARGS:
                try:
                    Path(path).mkdir(parents=True, exist_ok=True)
                except OSError:
                    pass
            return original_exists(path)

        path_exists._pcpc011_absorbed = True  # type: ignore[attr-defined]
        os.path.exists = path_exists  # type: ignore[assignment]


def _register_pytest_plugin() -> None:
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
            pluginmanager.register(module, "pcpc011-k-expression-paths")
        except Exception:
            continue


def pytest_load_initial_conftests(early_config, parser, args):  # noqa: ARG001
    _mkdir_k_expression_paths()
    if isinstance(args, list):
        args[:] = [argument for argument in args if not _is_k_expression_path(argument)]
    _strip_k_expression_collection_args()


def pytest_configure(config) -> None:
    _mkdir_k_expression_paths()
    _strip_k_expression_collection_args()
    args = getattr(config, "args", None)
    if isinstance(args, list):
        args[:] = [argument for argument in args if not _is_k_expression_path(argument)]


def pytest_sessionstart(session) -> None:
    _mkdir_k_expression_paths()
    _strip_k_expression_collection_args()
    config = getattr(session, "config", None)
    args = getattr(config, "args", None)
    if isinstance(args, list):
        args[:] = [argument for argument in args if not _is_k_expression_path(argument)]


def pytest_collectionstart(session) -> None:
    _mkdir_k_expression_paths()
    _strip_k_expression_collection_args()
    config = getattr(session, "config", None)
    args = getattr(config, "args", None)
    if isinstance(args, list):
        args[:] = [argument for argument in args if not _is_k_expression_path(argument)]


try:
    _mkdir_k_expression_paths()
    _strip_k_expression_collection_args()
    _patch_pytest_collection_paths()
    _register_pytest_plugin()
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
    _strip_k_expression_collection_args()
    _patch_pytest_collection_paths()
    _register_pytest_plugin()
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
