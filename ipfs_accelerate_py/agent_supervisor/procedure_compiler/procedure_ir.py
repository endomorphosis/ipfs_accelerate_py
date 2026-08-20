"""Parser and fail-closed static verifier for declarative ``ProcedureIR``.

The verifier proves structural properties of the IR only.  It does not treat
the procedure's conditions, observations, or certificate claims as true.
Runtime producers must independently establish those facts.
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Mapping
from functools import cache
from typing import Any, Final

from .contracts import (
    ALLOWED_STEP_OPERATIONS,
    FORBIDDEN_STEP_OPERATIONS,
    ArtifactBindings,
    EffectClass,
    FailureTransition,
    ProcedureAuthorityEnvelope,
    ProcedureBoundsError,
    ProcedureBranch,
    ProcedureContractError,
    ProcedureEffect,
    ProcedureFallback,
    ProcedureHole,
    ProcedureInvariant,
    ProcedureLocal,
    ProcedureLoop,
    ProcedureObservation,
    ProcedureParameter,
    ProcedurePostcondition,
    ProcedurePrecondition,
    ProcedureResourceEnvelope,
    ProcedureRollback,
    ProcedureSafetyError,
    ProcedureSpec,
    ProcedureStep,
    ProcedureValidationPlan,
    ProcedureVersion,
    ProviderClass,
    StepOperation,
    canonical_json_bytes,
)


class ProcedureIRValidationError(ProcedureContractError):
    """The ProcedureIR is structurally or semantically unsafe."""


class ProcedureGraphError(ProcedureIRValidationError):
    """Control flow is missing, cyclic outside a loop, or ambiguous."""


class ProcedureDataflowError(ProcedureIRValidationError):
    """A step consumes an undeclared or not-definitely-initialized value."""


class ProcedureEffectError(ProcedureIRValidationError):
    """An operation's declared effects exceed its closed effect vocabulary."""


class ProcedureScopeError(ProcedureIRValidationError):
    """A declared read or effect escapes the procedure's repository scope."""


class ProcedureValidationRetentionError(ProcedureIRValidationError):
    """A control-flow path can bypass a required validation step."""


_BINDING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "repository_id",
        "repository_commit",
        "tree_id",
        "objective_id",
        "task_id",
        "contract_revision",
        "policy_revision",
        "environment_id",
    }
)

_OPERATION_EFFECT_CLASSES: Final[Mapping[StepOperation, frozenset[EffectClass]]] = {
    StepOperation.READ_STATE: frozenset({EffectClass.OBSERVE}),
    StepOperation.QUERY_AST_INDEX: frozenset({EffectClass.OBSERVE}),
    StepOperation.QUERY_DEPENDENCY_GRAPH: frozenset({EffectClass.OBSERVE}),
    StepOperation.QUERY_SEMANTIC_INDEX: frozenset({EffectClass.OBSERVE}),
    StepOperation.QUERY_RECEIPT_CACHE: frozenset({EffectClass.OBSERVE}),
    StepOperation.SELECT_EVIDENCE: frozenset({EffectClass.OBSERVE}),
    StepOperation.EXPAND_CONTEXT_REFERENCE: frozenset({EffectClass.OBSERVE}),
    StepOperation.CHECK_CAPABILITY: frozenset({EffectClass.OBSERVE}),
    StepOperation.CHECK_POLICY: frozenset({EffectClass.OBSERVE}),
    StepOperation.CHECK_AUTHORITY: frozenset({EffectClass.OBSERVE}),
    StepOperation.CREATE_ISOLATED_WORKTREE: frozenset(
        {EffectClass.OBSERVE, EffectClass.WORKTREE_CREATE}
    ),
    StepOperation.APPLY_APPROVED_PATCH_TEMPLATE: frozenset(
        {EffectClass.OBSERVE, EffectClass.REPOSITORY_WRITE}
    ),
    StepOperation.REQUEST_TYPED_MODEL_HOLE: frozenset(
        {EffectClass.OBSERVE, EffectClass.MODEL_REQUEST}
    ),
    StepOperation.RUN_STATIC_ANALYSIS: frozenset({EffectClass.OBSERVE, EffectClass.VALIDATION}),
    StepOperation.RUN_TYPE_CHECK: frozenset({EffectClass.OBSERVE, EffectClass.VALIDATION}),
    StepOperation.RUN_SELECTED_TESTS: frozenset({EffectClass.OBSERVE, EffectClass.VALIDATION}),
    StepOperation.RUN_FULL_TEST_FALLBACK: frozenset({EffectClass.OBSERVE, EffectClass.VALIDATION}),
    StepOperation.RUN_PROOF: frozenset(
        {EffectClass.OBSERVE, EffectClass.PROOF, EffectClass.VALIDATION}
    ),
    StepOperation.RUN_ADVERSARIAL_ASSURANCE: frozenset(
        {EffectClass.OBSERVE, EffectClass.VALIDATION}
    ),
    StepOperation.CHECK_DIFF: frozenset({EffectClass.OBSERVE, EffectClass.VALIDATION}),
    StepOperation.CHECK_SCOPE: frozenset({EffectClass.OBSERVE, EffectClass.VALIDATION}),
    StepOperation.CHECK_POSTCONDITION: frozenset({EffectClass.OBSERVE, EffectClass.VALIDATION}),
    StepOperation.PREPARE_MERGE: frozenset({EffectClass.OBSERVE, EffectClass.MERGE_PREPARE}),
    StepOperation.MERGE_IN_ISOLATED_TRAIN: frozenset({EffectClass.OBSERVE, EffectClass.MERGE}),
    StepOperation.VERIFY_MERGED_TREE: frozenset({EffectClass.OBSERVE, EffectClass.VALIDATION}),
    StepOperation.PERSIST_ARTIFACT: frozenset({EffectClass.OBSERVE, EffectClass.ARTIFACT_PERSIST}),
    StepOperation.EMIT_RECEIPT: frozenset({EffectClass.OBSERVE, EffectClass.RECEIPT_EMIT}),
    StepOperation.ROLLBACK: frozenset({EffectClass.OBSERVE, EffectClass.ROLLBACK}),
    StepOperation.ESCALATE: frozenset({EffectClass.OBSERVE, EffectClass.ESCALATION}),
}

_REQUIRED_PRIMARY_EFFECT: Final[Mapping[StepOperation, EffectClass]] = {
    StepOperation.CREATE_ISOLATED_WORKTREE: EffectClass.WORKTREE_CREATE,
    StepOperation.APPLY_APPROVED_PATCH_TEMPLATE: EffectClass.REPOSITORY_WRITE,
    StepOperation.REQUEST_TYPED_MODEL_HOLE: EffectClass.MODEL_REQUEST,
    StepOperation.RUN_STATIC_ANALYSIS: EffectClass.VALIDATION,
    StepOperation.RUN_TYPE_CHECK: EffectClass.VALIDATION,
    StepOperation.RUN_SELECTED_TESTS: EffectClass.VALIDATION,
    StepOperation.RUN_FULL_TEST_FALLBACK: EffectClass.VALIDATION,
    StepOperation.RUN_PROOF: EffectClass.PROOF,
    StepOperation.RUN_ADVERSARIAL_ASSURANCE: EffectClass.VALIDATION,
    StepOperation.CHECK_DIFF: EffectClass.VALIDATION,
    StepOperation.CHECK_SCOPE: EffectClass.VALIDATION,
    StepOperation.CHECK_POSTCONDITION: EffectClass.VALIDATION,
    StepOperation.PREPARE_MERGE: EffectClass.MERGE_PREPARE,
    StepOperation.MERGE_IN_ISOLATED_TRAIN: EffectClass.MERGE,
    StepOperation.VERIFY_MERGED_TREE: EffectClass.VALIDATION,
    StepOperation.PERSIST_ARTIFACT: EffectClass.ARTIFACT_PERSIST,
    StepOperation.EMIT_RECEIPT: EffectClass.RECEIPT_EMIT,
    StepOperation.ROLLBACK: EffectClass.ROLLBACK,
    StepOperation.ESCALATE: EffectClass.ESCALATION,
}

_VALIDATION_OPERATIONS: Final[frozenset[StepOperation]] = frozenset(
    {
        StepOperation.RUN_STATIC_ANALYSIS,
        StepOperation.RUN_TYPE_CHECK,
        StepOperation.RUN_SELECTED_TESTS,
        StepOperation.RUN_FULL_TEST_FALLBACK,
        StepOperation.RUN_PROOF,
        StepOperation.RUN_ADVERSARIAL_ASSURANCE,
        StepOperation.CHECK_DIFF,
        StepOperation.CHECK_SCOPE,
        StepOperation.CHECK_POSTCONDITION,
        StepOperation.VERIFY_MERGED_TREE,
    }
)


def _duplicates(values: list[str] | tuple[str, ...]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _require_unique(values: list[str] | tuple[str, ...], field_name: str) -> None:
    if _duplicates(values):
        raise ProcedureIRValidationError(f"{field_name} contains duplicate identities")


def _path_pattern_matches(path: str, pattern: str) -> bool:
    """Match a repository path without allowing ``*`` to cross ``/``.

    Python's generic ``fnmatch`` treats slashes as ordinary characters, so a
    scope like ``src/*`` would otherwise authorize ``src/private/key.py``.
    Only a complete ``**`` segment is recursive in ProcedureIR scopes.
    """

    path_parts = tuple(path.split("/"))
    pattern_parts = tuple(pattern.split("/"))

    @cache
    def matches(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        pattern_part = pattern_parts[pattern_index]
        if pattern_part == "**":
            return matches(path_index, pattern_index + 1) or (
                path_index < len(path_parts) and matches(path_index + 1, pattern_index)
            )
        return (
            path_index < len(path_parts)
            and fnmatch.fnmatchcase(path_parts[path_index], pattern_part)
            and matches(path_index + 1, pattern_index + 1)
        )

    return matches(0, 0)


def _path_is_in_scope(path: str, scopes: tuple[str, ...]) -> bool:
    for scope in scopes:
        if scope == ".":
            return True
        if any(marker in scope for marker in "*?["):
            if _path_pattern_matches(path, scope):
                return True
            continue
        base = scope.rstrip("/")
        if path == base or path.startswith(base + "/"):
            return True
    return False


def _source_variable(source: str) -> str | None:
    """Return the procedure variable referenced by one declarative binding.

    Literals are inert data and exact artifact-binding fields are initialized
    by the invocation.  No expression evaluation is supported.
    """

    if source.startswith("literal:"):
        return None
    if source.startswith("binding:"):
        name = source[len("binding:") :]
        if name not in _BINDING_FIELDS:
            raise ProcedureDataflowError("input references an unknown exact binding")
        return None
    for prefix in ("parameter:", "param:", "local:"):
        if source.startswith(prefix):
            name = source[len(prefix) :]
            if not name:
                raise ProcedureDataflowError("input contains an empty variable reference")
            return name
    if any(character.isspace() for character in source):
        raise ProcedureDataflowError("input binding is not a declarative variable or literal")
    return source


def _output_variable(target: str) -> str:
    if target.startswith("local:"):
        target = target[len("local:") :]
    if not target or ":" in target or any(character.isspace() for character in target):
        raise ProcedureDataflowError("output binding must name one declared local")
    return target


def _normal_successors(
    spec: ProcedureSpec,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    successors: dict[str, tuple[str, ...]] = {}
    failure_successors: dict[str, tuple[str, ...]] = {}
    rollback_by_id = {rollback.rollback_id: rollback for rollback in spec.rollback}
    fallback_by_id = {fallback.fallback_id: fallback for fallback in spec.fallback}

    for step in spec.steps:
        successors[step.step_id] = (step.next_step_id,) if step.next_step_id else ()
        failure: tuple[str, ...] = ()
        if step.failure_transition is FailureTransition.ROLLBACK:
            rollback = rollback_by_id.get(step.failure_target)
            if rollback is None:
                raise ProcedureGraphError("step names an unknown rollback target")
            failure = (rollback.step_ids[0],)
        elif step.failure_transition is FailureTransition.FALLBACK:
            fallback = fallback_by_id.get(step.failure_target)
            if fallback is None:
                raise ProcedureGraphError("step names an unknown fallback target")
            failure = (fallback.entry_step_id,)
        failure_successors[step.step_id] = failure
    for branch in spec.branches:
        successors[branch.branch_id] = (branch.true_step_id, branch.false_step_id)
        failure_successors[branch.branch_id] = ()
    for loop in spec.loops:
        successors[loop.loop_id] = (loop.body_step_id, loop.exit_step_id)
        failure_successors[loop.loop_id] = ()
    return successors, failure_successors


def _reachable(entry: str, successors: Mapping[str, tuple[str, ...]]) -> set[str]:
    reached: set[str] = set()
    pending = [entry]
    while pending:
        node = pending.pop()
        if node in reached:
            continue
        reached.add(node)
        pending.extend(successors.get(node, ()))
    return reached


def _strongly_connected_components(
    nodes: set[str], successors: Mapping[str, tuple[str, ...]]
) -> tuple[tuple[str, ...], ...]:
    """Tarjan SCCs in deterministic node order."""

    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for successor in sorted(successors.get(node, ())):
            if successor not in nodes:
                continue
            if successor not in indices:
                visit(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[successor])
        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(tuple(sorted(component)))

    for node in sorted(nodes):
        if node not in indices:
            visit(node)
    return tuple(components)


def _validate_names_and_references(spec: ProcedureSpec) -> None:
    parameters = tuple(parameter.name for parameter in spec.parameters)
    locals_ = tuple(local.name for local in spec.locals)
    _require_unique(parameters, "parameters")
    _require_unique(locals_, "locals")
    if set(parameters).intersection(locals_):
        raise ProcedureDataflowError("parameters and locals must have distinct names")

    effect_ids = tuple(effect.effect_id for effect in spec.declared_effects)
    observation_ids = tuple(observation.observation_id for observation in spec.observations)
    condition_ids = tuple(
        condition.condition_id
        for condition in (*spec.preconditions, *spec.invariants, *spec.postconditions)
    )
    hole_ids = tuple(hole.hole_id for hole in spec.holes)
    rollback_ids = tuple(rollback.rollback_id for rollback in spec.rollback)
    fallback_ids = tuple(fallback.fallback_id for fallback in spec.fallback)
    for values, name in (
        (effect_ids, "declared_effects"),
        (observation_ids, "observations"),
        (condition_ids, "conditions"),
        (hole_ids, "holes"),
        (rollback_ids, "rollback"),
        (fallback_ids, "fallback"),
    ):
        _require_unique(values, name)

    step_ids = tuple(step.step_id for step in spec.steps)
    branch_ids = tuple(branch.branch_id for branch in spec.branches)
    loop_ids = tuple(loop.loop_id for loop in spec.loops)
    all_node_ids = (*step_ids, *branch_ids, *loop_ids)
    _require_unique(all_node_ids, "control-flow nodes")
    nodes = set(all_node_ids)
    if spec.entry_step_id not in nodes:
        raise ProcedureGraphError("entry_step_id does not name a control-flow node")
    if not set(spec.terminal_step_ids).issubset(set(step_ids)):
        raise ProcedureGraphError("terminal_step_ids must name executable steps")

    effects = set(effect_ids)
    observations = set(observation_ids)
    holes = set(hole_ids)
    authority_ids = set(spec.authority.requirement_ids)
    for step in spec.steps:
        if not set(step.declared_effect_ids).issubset(effects):
            raise ProcedureEffectError("step references an undeclared effect")
        if not set(step.required_authority_ids).issubset(authority_ids):
            raise ProcedureEffectError("step references an undeclared authority requirement")
        if step.hole_id and step.hole_id not in holes:
            raise ProcedureIRValidationError("step references an undeclared typed hole")
        if step.next_step_id and step.next_step_id not in nodes:
            raise ProcedureGraphError("step names an unknown next control-flow node")
    for branch in spec.branches:
        if branch.observation_id not in observations:
            raise ProcedureGraphError("branch names an unknown observation")
        if branch.true_step_id not in nodes or branch.false_step_id not in nodes:
            raise ProcedureGraphError("branch names an unknown target")
    for loop in spec.loops:
        if loop.condition_observation_id not in observations:
            raise ProcedureGraphError("loop names an unknown observation")
        if loop.body_step_id not in nodes or loop.exit_step_id not in nodes:
            raise ProcedureGraphError("loop names an unknown target")
    for rollback in spec.rollback:
        if not set(rollback.trigger_effect_ids).issubset(effects):
            raise ProcedureEffectError("rollback references an undeclared effect")
        if not set(rollback.step_ids).issubset(set(step_ids)):
            raise ProcedureGraphError("rollback references an unknown step")
        if not set(rollback.verification_observation_ids).issubset(observations):
            raise ProcedureGraphError("rollback references an unknown observation")
    for fallback in spec.fallback:
        if fallback.entry_step_id not in nodes:
            raise ProcedureGraphError("fallback references an unknown entry step")
    for hole in spec.holes:
        if not set(hole.authority_requirement_ids).issubset(authority_ids):
            raise ProcedureEffectError("hole authority exceeds the procedure envelope")
        if not set(hole.validation_observation_ids).issubset(observations):
            raise ProcedureIRValidationError("hole references unknown validation")
        if hole.fallback_step_id not in nodes:
            raise ProcedureGraphError("hole references an unknown fallback step")
    if not set(spec.validation.required_step_ids).issubset(set(step_ids)):
        raise ProcedureGraphError("validation plan references an unknown step")
    if not set(spec.validation.required_observation_ids).issubset(observations):
        raise ProcedureIRValidationError("validation plan references an unknown observation")
    evidence_producers: dict[str, set[str]] = {}
    for step in spec.steps:
        for evidence_id in step.evidence_outputs:
            evidence_producers.setdefault(evidence_id, set()).add(step.operation_contract)
    for observation in spec.observations:
        producers = evidence_producers.get(observation.observation_id, set())
        if observation.observation_id in set(spec.validation.required_observation_ids):
            if observation.producer_contract not in producers:
                raise ProcedureIRValidationError(
                    "required observation is not emitted by its declared external producer"
                )


def _validate_graph(
    spec: ProcedureSpec,
) -> tuple[
    dict[str, tuple[str, ...]],
    set[str],
    dict[str, tuple[str, ...]],
    set[str],
]:
    normal, failures = _normal_successors(spec)
    nodes = set(normal)
    for node, targets in normal.items():
        for target in targets:
            if target not in nodes:
                raise ProcedureGraphError(f"control-flow node {node} has an unknown successor")
    full = {node: tuple(dict.fromkeys((*normal[node], *failures[node]))) for node in sorted(nodes)}
    reachable = _reachable(spec.entry_step_id, full)
    if reachable != nodes:
        raise ProcedureGraphError("ProcedureIR contains unreachable control-flow nodes")
    normal_reachable = _reachable(spec.entry_step_id, normal)
    for node in normal_reachable:
        if not normal[node] and node not in set(spec.terminal_step_ids):
            raise ProcedureGraphError("a normal path ends outside a declared terminal step")
    if not set(spec.terminal_step_ids).issubset(normal_reachable):
        raise ProcedureGraphError("a declared terminal is not normally reachable")
    for terminal in spec.terminal_step_ids:
        if normal[terminal]:
            raise ProcedureGraphError("terminal steps may not have normal successors")

    loop_ids = {loop.loop_id for loop in spec.loops}
    for component in _strongly_connected_components(nodes, full):
        cyclic = len(component) > 1 or (len(component) == 1 and component[0] in full[component[0]])
        if cyclic and len(set(component).intersection(loop_ids)) != 1:
            raise ProcedureGraphError(
                "every control-flow cycle must pass through exactly one bounded loop"
            )
    return normal, normal_reachable, full, reachable


def _validate_effects_and_scope(spec: ProcedureSpec) -> None:
    effect_by_id = {effect.effect_id: effect for effect in spec.declared_effects}
    used_effects: set[str] = set()
    if spec.authority.authority_policy_revision != spec.bindings.policy_revision:
        raise ProcedureEffectError("procedure authority envelope is stale for exact bindings")
    allowed_operations = set(spec.authority.allowed_operations)
    for path in spec.declared_reads:
        if not _path_is_in_scope(path, spec.scope_paths):
            raise ProcedureScopeError("a declared read escapes procedure scope")
    for effect in spec.declared_effects:
        for target in effect.targets:
            if not _path_is_in_scope(target, spec.scope_paths):
                raise ProcedureScopeError("a declared effect target escapes procedure scope")

    for step in spec.steps:
        if step.operation.value in FORBIDDEN_STEP_OPERATIONS:
            raise ProcedureSafetyError("ProcedureIR contains a forbidden operation")
        if step.operation.value not in ALLOWED_STEP_OPERATIONS:
            raise ProcedureSafetyError("ProcedureIR contains a non-allowlisted operation")
        if step.operation not in allowed_operations:
            raise ProcedureEffectError("step operation exceeds its authority envelope")
        declared = tuple(effect_by_id[effect_id] for effect_id in step.declared_effect_ids)
        declared_classes = {effect.effect_class for effect in declared}
        permitted = _OPERATION_EFFECT_CLASSES[step.operation]
        if not declared_classes.issubset(permitted):
            raise ProcedureEffectError("step effect class is not permitted for its operation")
        primary = _REQUIRED_PRIMARY_EFFECT.get(step.operation)
        if primary is not None and primary not in declared_classes:
            raise ProcedureEffectError("effectful operation omits its required primary effect")
        if primary is not None:
            if not step.required_authority_ids:
                raise ProcedureEffectError("effectful operation lacks admitted authority")
            if not step.evidence_outputs:
                raise ProcedureEffectError(
                    "effectful operation lacks an observation/evidence output"
                )
        if (
            step.failure_transition is FailureTransition.RETRY
            and step.retry_policy.max_attempts < 2
        ):
            raise ProcedureBoundsError("retry transition requires a bounded retry allowance")
        used_effects.update(step.declared_effect_ids)
        if step.timeout_ms > spec.resources.wall_time_ms:
            raise ProcedureBoundsError("step timeout exceeds the procedure resource envelope")
    if set(effect_by_id).difference(used_effects):
        raise ProcedureEffectError("procedure declares unused effects")

    hole_by_id = {hole.hole_id: hole for hole in spec.holes}
    for step in spec.steps:
        if not step.hole_id:
            continue
        hole = hole_by_id[step.hole_id]
        actual = {effect_by_id[item].effect_class for item in step.declared_effect_ids}
        if not actual.issubset(set(hole.effect_classes) | {EffectClass.OBSERVE}):
            raise ProcedureEffectError("typed hole effects exceed its declared envelope")
    model_holes = [
        hole
        for hole in spec.holes
        if any(
            provider
            in {
                ProviderClass.LOCAL_SMALL_MODEL,
                ProviderClass.REMOTE_STANDARD_MODEL,
                ProviderClass.REMOTE_STRONG_MODEL,
            }
            for provider in hole.allowed_provider_classes
        )
    ]
    if sum(hole.maximum_attempts for hole in model_holes) > spec.resources.model_call_limit:
        raise ProcedureBoundsError("typed-hole attempts exceed the model-call envelope")
    if model_holes and spec.resources.model_token_limit == 0:
        raise ProcedureBoundsError("model-backed holes require a nonzero token envelope")


def _node_requirements(
    spec: ProcedureSpec,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    requirements: dict[str, set[str]] = {}
    productions: dict[str, set[str]] = {}
    observation_by_id = {
        observation.observation_id: observation for observation in spec.observations
    }
    declared_variables = {parameter.name for parameter in spec.parameters} | {
        local.name for local in spec.locals
    }
    local_variables = {local.name for local in spec.locals}
    for step in spec.steps:
        required: set[str] = set()
        for source in step.input_bindings.values():
            variable = _source_variable(source)
            if variable is not None:
                if variable not in declared_variables:
                    raise ProcedureDataflowError("step input references an undeclared variable")
                required.add(variable)
        produced: set[str] = set()
        for target in step.output_bindings.values():
            variable = _output_variable(target)
            if variable not in local_variables:
                raise ProcedureDataflowError("step output must bind a declared local")
            produced.add(variable)
        requirements[step.step_id] = required
        productions[step.step_id] = produced
    for branch in spec.branches:
        variable = _source_variable(observation_by_id[branch.observation_id].output_binding)
        requirements[branch.branch_id] = {variable} if variable is not None else set()
        productions[branch.branch_id] = set()
    for loop in spec.loops:
        variable = _source_variable(observation_by_id[loop.condition_observation_id].output_binding)
        requirements[loop.loop_id] = {variable} if variable is not None else set()
        productions[loop.loop_id] = set()
    for node, values in requirements.items():
        if not values.issubset(declared_variables):
            raise ProcedureDataflowError(
                f"control-flow node {node} references an undeclared variable"
            )
    for condition in (*spec.preconditions, *spec.invariants, *spec.postconditions):
        variable = _source_variable(condition.binding)
        if variable is not None and variable not in declared_variables:
            raise ProcedureDataflowError("condition references an undeclared variable")
    return requirements, productions


def _validate_dataflow(
    spec: ProcedureSpec,
    successors: Mapping[str, tuple[str, ...]],
    reachable: set[str],
    *,
    normal_successors: Mapping[str, tuple[str, ...]],
) -> None:
    requirements, productions = _node_requirements(spec)
    predecessors: dict[str, set[str]] = {node: set() for node in successors}
    for node, targets in successors.items():
        for target in targets:
            predecessors[target].add(node)
    parameters = {parameter.name for parameter in spec.parameters}
    all_variables = parameters | {local.name for local in spec.locals}
    in_sets: dict[str, set[str]] = {
        node: (set(parameters) if node == spec.entry_step_id else set(all_variables))
        for node in reachable
    }
    changed = True
    while changed:
        changed = False
        for node in sorted(reachable):
            if node == spec.entry_step_id:
                candidate = set(parameters)
            else:
                incoming = []
                for parent in predecessors[node]:
                    if parent not in reachable:
                        continue
                    values = set(in_sets[parent])
                    # A failed operation cannot establish its advertised
                    # outputs merely because recovery control transfers.
                    if node in normal_successors[parent]:
                        values.update(productions[parent])
                    incoming.append(values)
                candidate = set.intersection(*incoming) if incoming else set()
            if candidate != in_sets[node]:
                in_sets[node] = candidate
                changed = True
    for node in sorted(reachable):
        missing = requirements[node].difference(in_sets[node])
        if missing:
            raise ProcedureDataflowError(
                "a control-flow path consumes a local before definite initialization"
            )


def _validate_validation_retention(
    spec: ProcedureSpec,
    successors: Mapping[str, tuple[str, ...]],
    reachable: set[str],
) -> None:
    required = set(spec.validation.required_step_ids)
    step_by_id = {step.step_id: step for step in spec.steps}
    if any(step_by_id[step_id].operation not in _VALIDATION_OPERATIONS for step_id in required):
        raise ProcedureValidationRetentionError(
            "required validation step does not use a validation operation"
        )
    predecessors: dict[str, set[str]] = {node: set() for node in successors}
    for node, targets in successors.items():
        for target in targets:
            predecessors[target].add(node)
    dominators: dict[str, set[str]] = {
        node: ({node} if node == spec.entry_step_id else set(reachable)) for node in reachable
    }
    changed = True
    while changed:
        changed = False
        for node in sorted(reachable):
            if node == spec.entry_step_id:
                candidate = {node}
            else:
                incoming = [
                    dominators[parent] for parent in predecessors[node] if parent in reachable
                ]
                candidate = {node} | (set.intersection(*incoming) if incoming else set())
            if candidate != dominators[node]:
                dominators[node] = candidate
                changed = True
    for terminal in spec.terminal_step_ids:
        if not required.issubset(dominators[terminal]):
            raise ProcedureValidationRetentionError("a terminal path bypasses required validation")
    if not any(
        step.operation is StepOperation.CHECK_POSTCONDITION
        for step in spec.steps
        if step.step_id in required
    ):
        raise ProcedureValidationRetentionError(
            "validation must retain an independently observed postcondition check"
        )


def validate_procedure_spec(spec: ProcedureSpec) -> ProcedureSpec:
    """Validate the complete static graph and return the same immutable value."""

    if not isinstance(spec, ProcedureSpec):
        raise ProcedureIRValidationError("spec must be a ProcedureSpec")
    _validate_names_and_references(spec)
    normal_successors, normal_reachable, full_successors, full_reachable = _validate_graph(spec)
    _validate_effects_and_scope(spec)
    _validate_dataflow(
        spec,
        full_successors,
        full_reachable,
        normal_successors=normal_successors,
    )
    _validate_validation_retention(spec, normal_successors, normal_reachable)
    # Force canonical serialization here so unsupported values cannot survive
    # a subclass or mapping implementation at this final boundary.
    canonical_json_bytes(spec.to_dict())
    return spec


def _reject_float(_: str) -> Any:
    raise ProcedureIRValidationError("ProcedureIR JSON cannot contain floating point values")


def _reject_constant(_: str) -> Any:
    raise ProcedureIRValidationError("ProcedureIR JSON cannot contain nonfinite values")


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProcedureIRValidationError("ProcedureIR JSON contains a duplicate object field")
        result[key] = value
    return result


def parse_procedure_spec(value: Any) -> ProcedureSpec:
    """Parse mapping/JSON input deterministically and perform static validation."""

    if isinstance(value, ProcedureSpec):
        return validate_procedure_spec(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ProcedureIRValidationError("ProcedureIR bytes must be UTF-8") from exc
    if isinstance(value, str):
        try:
            value = json.loads(
                value,
                object_pairs_hook=_closed_object,
                parse_float=_reject_float,
                parse_constant=_reject_constant,
            )
        except json.JSONDecodeError as exc:
            raise ProcedureIRValidationError("ProcedureIR JSON is malformed") from exc
    if not isinstance(value, Mapping):
        raise ProcedureIRValidationError("ProcedureIR must be an object or canonical JSON object")
    return validate_procedure_spec(ProcedureSpec.from_dict(value))


parse_procedure_ir = parse_procedure_spec


class ProcedureIRParser:
    """Stateless parser object for dependency-injected control surfaces."""

    @staticmethod
    def parse(value: Any) -> ProcedureSpec:
        return parse_procedure_spec(value)

    @staticmethod
    def validate(value: ProcedureSpec) -> ProcedureSpec:
        return validate_procedure_spec(value)


__all__ = [
    "ArtifactBindings",
    "EffectClass",
    "ProcedureAuthorityEnvelope",
    "ProcedureBranch",
    "ProcedureDataflowError",
    "ProcedureEffect",
    "ProcedureEffectError",
    "ProcedureFallback",
    "ProcedureGraphError",
    "ProcedureHole",
    "ProcedureIRParser",
    "ProcedureIRValidationError",
    "ProcedureInvariant",
    "ProcedureLocal",
    "ProcedureLoop",
    "ProcedureObservation",
    "ProcedureParameter",
    "ProcedurePostcondition",
    "ProcedurePrecondition",
    "ProcedureResourceEnvelope",
    "ProcedureRollback",
    "ProcedureScopeError",
    "ProcedureSpec",
    "ProcedureStep",
    "ProcedureValidationPlan",
    "ProcedureValidationRetentionError",
    "ProcedureVersion",
    "StepOperation",
    "parse_procedure_ir",
    "parse_procedure_spec",
    "validate_procedure_spec",
]
