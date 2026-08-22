"""Deterministic decision/uncertainty graph operations.

This module is a pure orchestration layer over the immutable contracts in
``autonomy.contracts``.  It does not plan work, persist state, admit effects,
or execute resolution actions.  In particular, effect admission remains the
authority of :mod:`agent_supervisor.context.decision_runtime`.

``DecisionQuestion.question_id`` is the content identity of a *version* of a
question.  Evidence and disposition changes therefore produce a new ID.  The
stable key used for deduplication and dependency-preserving reconstruction is
``semantic_question_id``.  That key deliberately excludes criteria,
evidence, risk annotations, deadlines, and current disposition: equivalent
questions from several tasks collapse to one conservative question whose
requirements are the union of their inputs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from itertools import islice
from types import MappingProxyType
from typing import Any

from ..proof.formal_verification_contracts import (
    CONTRACT_VERSION,
    CanonicalContract,
    canonical_json,
    content_identity,
)
from .contracts import (
    AutonomyContractError,
    DecisionGraph,
    DecisionQuestion,
    QuestionDisposition,
    RiskClass,
)

SEMANTIC_QUESTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/semantic-question@1"
)
DECISION_GRAPH_CHANGE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/decision-graph-change-receipt@1"
)
MAX_GRAPH_QUESTIONS = 1_024
MAX_GRAPH_EVIDENCE_INDEXES = 256
MAX_GRAPH_ID_BYTES = 512
MAX_GRAPH_RECEIPT_BYTES = 262_144


class DecisionGraphError(AutonomyContractError):
    """Raised when a graph transition is not closed or admissible."""


class DecisionGraphChangeKind(str, Enum):  # noqa: UP042 - package supports Python 3.8
    COMPILED = "compiled"
    EVIDENCE_RECORDED = "evidence_recorded"
    QUESTION_RESOLVED = "question_resolved"
    EVIDENCE_INVALIDATED = "evidence_invalidated"
    CONTRADICTION_RECORDED = "contradiction_recorded"
    NO_OP = "no_op"


def semantic_question_id(question: DecisionQuestion) -> str:
    """Return the stable identity of the decision being asked.

    Alternative order is not semantic.  All other omitted fields are merged
    conservatively by :func:`compile_decision_graph` rather than being allowed
    to split one underlying decision into several model calls.
    """

    if not isinstance(question, DecisionQuestion):
        raise DecisionGraphError("question must be a DecisionQuestion")
    return content_identity(
        {
            "schema": SEMANTIC_QUESTION_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "objective_id": question.objective_id,
            "question_type": question.question_type.value,
            "current_alternatives": sorted(question.current_alternatives),
            "terminal_decision_rule": question.terminal_decision_rule,
        }
    )


def _risk_max(values: Iterable[RiskClass]) -> RiskClass:
    return max(values, key=lambda value: value.rank)


def _minimum_nonzero(values: Iterable[int]) -> int:
    nonzero = [value for value in values if value > 0]
    return min(nonzero) if nonzero else 0


def _merge_question_group(group: Sequence[DecisionQuestion]) -> DecisionQuestion:
    """Merge semantically equal versions without weakening any requirement."""

    if not group:
        raise DecisionGraphError("cannot merge an empty question group")
    representative = min(group, key=lambda item: item.question_id)
    known = set().union(*(set(item.known_evidence_ids) for item in group))
    contradictory = set().union(
        *(set(item.contradictory_evidence_ids) for item in group)
    )
    overlap = known.intersection(contradictory)
    # Conflicting duplicate observations fail closed.  The contradictory
    # classification wins and the question cannot remain terminal.
    known.difference_update(overlap)

    answers = {
        item.terminal_answer
        for item in group
        if item.disposition is QuestionDisposition.RESOLVED
    }
    all_resolved = all(
        item.disposition is QuestionDisposition.RESOLVED for item in group
    )
    if (
        overlap
        or len(answers) > 1
        or any(item.disposition is QuestionDisposition.BLOCKED for item in group)
    ):
        disposition = QuestionDisposition.BLOCKED
        terminal_answer = ""
    elif all_resolved and len(answers) == 1 and not contradictory:
        disposition = QuestionDisposition.RESOLVED
        terminal_answer = next(iter(answers))
    elif any(item.disposition is QuestionDisposition.INVALIDATED for item in group):
        disposition = QuestionDisposition.INVALIDATED
        terminal_answer = ""
    else:
        disposition = QuestionDisposition.UNRESOLVED
        terminal_answer = ""

    return DecisionQuestion(
        objective_id=representative.objective_id,
        acceptance_criterion_ids=tuple(
            sorted(set().union(*(set(item.acceptance_criterion_ids) for item in group)))
        ),
        question_type=representative.question_type,
        current_alternatives=tuple(sorted(representative.current_alternatives)),
        required_evidence_ids=tuple(
            sorted(set().union(*(set(item.required_evidence_ids) for item in group)))
        ),
        known_evidence_ids=tuple(sorted(known)),
        contradictory_evidence_ids=tuple(sorted(contradictory)),
        residual_uncertainty_bp=max(item.residual_uncertainty_bp for item in group),
        decision_deadline_ms=_minimum_nonzero(
            item.decision_deadline_ms for item in group
        ),
        risk_if_incorrect=_risk_max(item.risk_if_incorrect for item in group),
        risk_if_left_unresolved=_risk_max(
            item.risk_if_left_unresolved for item in group
        ),
        possible_resolution_action_ids=tuple(
            sorted(
                set().union(
                    *(set(item.possible_resolution_action_ids) for item in group)
                )
            )
        ),
        # Rewritten from semantic dependencies after every transition.
        dependency_question_ids=(),
        terminal_decision_rule=representative.terminal_decision_rule,
        mandatory=any(item.mandatory for item in group),
        disposition=disposition,
        terminal_answer=terminal_answer,
    )


def _topological_semantic_keys(
    states: Mapping[str, DecisionQuestion],
    dependencies: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    order: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(key: str) -> None:
        if key in visiting:
            raise DecisionGraphError("semantic question dependencies must be acyclic")
        if key in visited:
            return
        if key not in states:
            raise DecisionGraphError(
                "question dependency references an unknown question"
            )
        visiting.add(key)
        for dependency in sorted(dependencies.get(key, ())):
            visit(dependency)
        visiting.remove(key)
        visited.add(key)
        order.append(key)

    for semantic_id in sorted(states):
        visit(semantic_id)
    return tuple(order)


def _build_graph(
    *,
    repository_id: str,
    tree_id: str,
    objective_id: str,
    objective_revision: str,
    graph_revision: int,
    states: Mapping[str, DecisionQuestion],
    dependencies: Mapping[str, tuple[str, ...]],
    evidence_dependencies: Mapping[str, tuple[str, ...]],
) -> DecisionGraph:
    """Rebase content IDs while retaining stable semantic dependencies."""

    built: dict[str, DecisionQuestion] = {}
    for semantic_id in _topological_semantic_keys(states, dependencies):
        state = states[semantic_id]
        if semantic_question_id(state) != semantic_id:
            raise DecisionGraphError(
                "question state does not match its semantic identity"
            )
        dependency_ids = tuple(
            sorted(built[key].question_id for key in dependencies.get(semantic_id, ()))
        )
        built[semantic_id] = replace(
            state,
            dependency_question_ids=dependency_ids,
        )

    ordered_keys = tuple(sorted(built))
    evidence_by_version = {
        built[key].question_id: tuple(sorted(set(evidence_dependencies.get(key, ()))))
        for key in ordered_keys
        if evidence_dependencies.get(key, ())
    }
    return DecisionGraph(
        repository_id=repository_id,
        tree_id=tree_id,
        objective_id=objective_id,
        objective_revision=objective_revision,
        graph_revision=graph_revision,
        questions=tuple(built[key] for key in ordered_keys),
        evidence_dependencies=evidence_by_version,
    )


def _semantic_state(
    graph: DecisionGraph,
) -> tuple[
    dict[str, DecisionQuestion],
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
    dict[str, str],
]:
    """Project a versioned graph onto stable semantic keys."""

    by_version = {question.question_id: question for question in graph.questions}
    version_to_semantic = {
        question_id: semantic_question_id(question)
        for question_id, question in by_version.items()
    }
    states = {
        version_to_semantic[question.question_id]: question
        for question in graph.questions
    }
    if len(states) != len(graph.questions):
        raise DecisionGraphError(
            "graph contains semantic duplicates; compile it before transition"
        )
    dependencies = {
        version_to_semantic[question.question_id]: tuple(
            sorted(
                version_to_semantic[item] for item in question.dependency_question_ids
            )
        )
        for question in graph.questions
    }
    evidence_dependencies = {
        version_to_semantic[question_id]: tuple(values)
        for question_id, values in graph.evidence_dependencies.items()
    }
    return states, dependencies, evidence_dependencies, version_to_semantic


def compile_decision_graph(
    *,
    repository_id: str,
    tree_id: str,
    objective_id: str,
    objective_revision: str,
    questions: Iterable[DecisionQuestion],
    evidence_dependencies: Mapping[str, Sequence[str]] | None = None,
    graph_revision: int = 0,
) -> DecisionGraph:
    """Deduplicate questions and return a deterministically ordered graph.

    Evidence dependency keys may be input question version IDs or stable
    semantic IDs.  References are rebound to the merged question version IDs.
    """

    try:
        raw_questions = tuple(islice(iter(questions), MAX_GRAPH_QUESTIONS + 1))
    except TypeError as exc:
        raise DecisionGraphError("questions must be an iterable") from exc
    if len(raw_questions) > MAX_GRAPH_QUESTIONS:
        raise DecisionGraphError("decision graph contains too many questions")
    groups: dict[str, list[DecisionQuestion]] = {}
    aliases: dict[str, str] = {}
    for question in raw_questions:
        if not isinstance(question, DecisionQuestion):
            raise DecisionGraphError("questions must contain DecisionQuestion values")
        if question.objective_id != objective_id:
            raise DecisionGraphError(
                "question objective does not match graph objective"
            )
        semantic_id = semantic_question_id(question)
        groups.setdefault(semantic_id, []).append(question)
        aliases[question.question_id] = semantic_id

    states = {
        semantic_id: _merge_question_group(group)
        for semantic_id, group in groups.items()
    }
    dependencies: dict[str, tuple[str, ...]] = {}
    for semantic_id, group in groups.items():
        dependency_keys: set[str] = set()
        for question in group:
            for dependency_id in question.dependency_question_ids:
                try:
                    dependency_keys.add(aliases[dependency_id])
                except KeyError as exc:
                    raise DecisionGraphError(
                        "question dependency references an unknown input question"
                    ) from exc
        if semantic_id in dependency_keys:
            raise DecisionGraphError(
                "deduplication cannot make a question depend on itself"
            )
        dependencies[semantic_id] = tuple(sorted(dependency_keys))

    evidence_by_semantic: dict[str, set[str]] = {}
    raw_evidence_dependencies = evidence_dependencies or {}
    if not isinstance(raw_evidence_dependencies, Mapping):
        raise DecisionGraphError("evidence_dependencies must be a bounded mapping")
    if len(raw_evidence_dependencies) > MAX_GRAPH_EVIDENCE_INDEXES:
        raise DecisionGraphError("evidence_dependencies contains too many entries")
    for raw_key, values in raw_evidence_dependencies.items():
        semantic_id = aliases.get(raw_key, raw_key)
        if semantic_id not in states:
            raise DecisionGraphError(
                "evidence dependency references an unknown input question"
            )
        if isinstance(values, str):
            values = (values,)
        try:
            bounded_values = tuple(islice(iter(values), MAX_GRAPH_QUESTIONS + 1))
        except TypeError as exc:
            raise DecisionGraphError(
                "evidence dependency values must be bounded iterables"
            ) from exc
        if len(bounded_values) > MAX_GRAPH_QUESTIONS:
            raise DecisionGraphError("evidence dependency contains too many identities")
        evidence_by_semantic.setdefault(semantic_id, set()).update(bounded_values)
    # Observed evidence is always an invalidation dependency, even when a
    # caller omitted the optional index.
    for semantic_id, state in states.items():
        evidence_by_semantic.setdefault(semantic_id, set()).update(
            state.known_evidence_ids
        )
        evidence_by_semantic[semantic_id].update(state.contradictory_evidence_ids)

    return _build_graph(
        repository_id=repository_id,
        tree_id=tree_id,
        objective_id=objective_id,
        objective_revision=objective_revision,
        graph_revision=graph_revision,
        states=states,
        dependencies=dependencies,
        evidence_dependencies={
            key: tuple(sorted(values)) for key, values in evidence_by_semantic.items()
        },
    )


def _dependency_descendants(
    dependencies: Mapping[str, tuple[str, ...]], direct: set[str]
) -> set[str]:
    affected = set(direct)
    changed = True
    while changed:
        changed = False
        for key, question_dependencies in dependencies.items():
            if key not in affected and affected.intersection(question_dependencies):
                affected.add(key)
                changed = True
    return affected


def _resolve_semantic_reference(
    reference: str,
    states: Mapping[str, DecisionQuestion],
    version_to_semantic: Mapping[str, str],
) -> str:
    if reference in states:
        return reference
    try:
        return version_to_semantic[reference]
    except KeyError as exc:
        raise DecisionGraphError("unknown question reference") from exc


@dataclass(frozen=True)
class DecisionGraphChangeReceipt(CanonicalContract):
    """Compact, content-addressed audit receipt for one pure transition."""

    SCHEMA = DECISION_GRAPH_CHANGE_RECEIPT_SCHEMA

    before_graph_id: str
    after_graph_id: str
    change_kind: DecisionGraphChangeKind
    evidence_ids: tuple[str, ...]
    direct_question_ids: tuple[str, ...]
    affected_question_ids: tuple[str, ...]
    invalidated_question_ids: tuple[str, ...]
    preserved_question_ids: tuple[str, ...]
    disposition_changes: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        for name in ("before_graph_id", "after_graph_id"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value
                or len(value.encode("utf-8")) > MAX_GRAPH_ID_BYTES
                or any(char.isspace() for char in value)
            ):
                raise DecisionGraphError(f"{name} must be a compact identity")
        if not isinstance(self.change_kind, DecisionGraphChangeKind):
            try:
                object.__setattr__(
                    self, "change_kind", DecisionGraphChangeKind(str(self.change_kind))
                )
            except ValueError as exc:
                raise DecisionGraphError("invalid graph change kind") from exc
        for name in (
            "evidence_ids",
            "direct_question_ids",
            "affected_question_ids",
            "invalidated_question_ids",
            "preserved_question_ids",
        ):
            values = getattr(self, name)
            if isinstance(values, str) or not isinstance(values, Sequence):
                raise DecisionGraphError(f"{name} must be a sequence")
            if len(values) > MAX_GRAPH_QUESTIONS:
                raise DecisionGraphError(f"{name} contains too many identities")
            normalized = tuple(sorted(set(values)))
            if any(
                not isinstance(value, str)
                or not value
                or len(value.encode("utf-8")) > MAX_GRAPH_ID_BYTES
                or any(char.isspace() for char in value)
                for value in normalized
            ):
                raise DecisionGraphError(f"{name} contains an invalid identity")
            object.__setattr__(self, name, normalized)
        if len(self.disposition_changes) > MAX_GRAPH_QUESTIONS:
            raise DecisionGraphError("disposition_changes contains too many entries")
        normalized_changes: list[tuple[str, str, str]] = []
        for change in self.disposition_changes:
            if not isinstance(change, Sequence) or len(change) != 3:
                raise DecisionGraphError("disposition change must contain three fields")
            semantic_id, before, after = change
            if (
                not isinstance(semantic_id, str)
                or not semantic_id
                or len(semantic_id.encode("utf-8")) > MAX_GRAPH_ID_BYTES
                or any(char.isspace() for char in semantic_id)
            ):
                raise DecisionGraphError("disposition change has an invalid identity")
            QuestionDisposition(before)
            QuestionDisposition(after)
            normalized_changes.append((str(semantic_id), str(before), str(after)))
        object.__setattr__(
            self, "disposition_changes", tuple(sorted(set(normalized_changes)))
        )
        if set(self.invalidated_question_ids).difference(self.affected_question_ids):
            raise DecisionGraphError("invalidated questions must be affected")
        if set(self.preserved_question_ids).intersection(self.affected_question_ids):
            raise DecisionGraphError("a question cannot be affected and preserved")
        if len(canonical_json(self.to_dict()).encode("utf-8")) > MAX_GRAPH_RECEIPT_BYTES:
            raise DecisionGraphError("decision graph receipt exceeds its bounded size")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "before_graph_id": self.before_graph_id,
            "after_graph_id": self.after_graph_id,
            "change_kind": self.change_kind,
            "evidence_ids": self.evidence_ids,
            "direct_question_ids": self.direct_question_ids,
            "affected_question_ids": self.affected_question_ids,
            "invalidated_question_ids": self.invalidated_question_ids,
            "preserved_question_ids": self.preserved_question_ids,
            "disposition_changes": self.disposition_changes,
        }

    @property
    def receipt_id(self) -> str:
        return self.content_id


@dataclass(frozen=True)
class DecisionGraphTransition:
    graph: DecisionGraph
    receipt: DecisionGraphChangeReceipt


def _transition(
    graph: DecisionGraph,
    *,
    states: Mapping[str, DecisionQuestion],
    dependencies: Mapping[str, tuple[str, ...]],
    evidence_dependencies: Mapping[str, tuple[str, ...]],
    tree_id: str,
    change_kind: DecisionGraphChangeKind,
    evidence_ids: Iterable[str],
    direct: set[str],
    affected: set[str],
) -> DecisionGraphTransition:
    before_states, _, _, _ = _semantic_state(graph)
    # ``affected`` is the dependency cone considered by the operation;
    # ``changed`` is the smaller set whose immutable question body actually
    # changed.  Keeping those distinct makes exact replay a true no-op.
    changed = {key for key in states if states[key] != before_states[key]}
    if not changed.issubset(affected):
        raise DecisionGraphError(
            "transition changed a question outside its affected cone"
        )
    if not changed and tree_id == graph.tree_id:
        after = graph
        actual_kind = DecisionGraphChangeKind.NO_OP
    else:
        after = _build_graph(
            repository_id=graph.repository_id,
            tree_id=tree_id,
            objective_id=graph.objective_id,
            objective_revision=graph.objective_revision,
            graph_revision=graph.graph_revision + 1,
            states=states,
            dependencies=dependencies,
            evidence_dependencies=evidence_dependencies,
        )
        actual_kind = change_kind
    invalidated = {
        key
        for key in changed
        if states[key].disposition is QuestionDisposition.INVALIDATED
    }
    disposition_changes = tuple(
        (
            key,
            before_states[key].disposition.value,
            states[key].disposition.value,
        )
        for key in sorted(changed)
        if before_states[key].disposition != states[key].disposition
    )
    all_keys = set(states)
    receipt = DecisionGraphChangeReceipt(
        before_graph_id=graph.graph_id,
        after_graph_id=after.graph_id,
        change_kind=actual_kind,
        evidence_ids=tuple(evidence_ids),
        direct_question_ids=tuple(direct),
        affected_question_ids=tuple(changed),
        invalidated_question_ids=tuple(invalidated),
        preserved_question_ids=tuple(all_keys.difference(changed)),
        disposition_changes=disposition_changes,
    )
    return DecisionGraphTransition(graph=after, receipt=receipt)


def resolve_question(
    graph: DecisionGraph,
    question_id: str,
    *,
    terminal_answer: str,
    evidence_ids: Iterable[str],
) -> DecisionGraphTransition:
    """Resolve one named question using explicit, attributable evidence."""

    states, dependencies, evidence_dependencies, aliases = _semantic_state(graph)
    semantic_id = _resolve_semantic_reference(question_id, states, aliases)
    evidence = tuple(sorted(set(evidence_ids)))
    if not evidence:
        raise DecisionGraphError("resolution requires attributable evidence")
    state = states[semantic_id]
    if terminal_answer not in state.current_alternatives:
        raise DecisionGraphError("terminal answer is outside the bounded alternatives")
    if state.contradictory_evidence_ids:
        raise DecisionGraphError(
            "contradictory evidence must be resolved before terminality"
        )
    if set(evidence).intersection(state.contradictory_evidence_ids):
        raise DecisionGraphError("resolution evidence is contradictory")
    known = tuple(sorted(set(state.known_evidence_ids).union(evidence)))
    states[semantic_id] = replace(
        state,
        known_evidence_ids=known,
        residual_uncertainty_bp=0,
        disposition=QuestionDisposition.RESOLVED,
        terminal_answer=terminal_answer,
    )
    evidence_dependencies[semantic_id] = tuple(
        sorted(set(evidence_dependencies.get(semantic_id, ())).union(evidence))
    )
    return _transition(
        graph,
        states=states,
        dependencies=dependencies,
        evidence_dependencies=evidence_dependencies,
        tree_id=graph.tree_id,
        change_kind=DecisionGraphChangeKind.QUESTION_RESOLVED,
        evidence_ids=evidence,
        direct={semantic_id},
        affected={semantic_id},
    )


def record_question_evidence(
    graph: DecisionGraph,
    question_id: str,
    *,
    evidence_ids: Iterable[str],
    contradictory: bool = False,
) -> DecisionGraphTransition:
    """Attach evidence, invalidating only a contradicted dependency suffix."""

    states, dependencies, evidence_dependencies, aliases = _semantic_state(graph)
    semantic_id = _resolve_semantic_reference(question_id, states, aliases)
    evidence = tuple(sorted(set(evidence_ids)))
    if not evidence:
        raise DecisionGraphError("at least one evidence identity is required")
    state = states[semantic_id]
    direct = {semantic_id}
    if contradictory:
        affected = _dependency_descendants(dependencies, direct)
        for key in affected:
            current = states[key]
            if key == semantic_id:
                known = tuple(
                    sorted(set(current.known_evidence_ids).difference(evidence))
                )
                contradictions = tuple(
                    sorted(set(current.contradictory_evidence_ids).union(evidence))
                )
            else:
                known = current.known_evidence_ids
                contradictions = current.contradictory_evidence_ids
            states[key] = replace(
                current,
                known_evidence_ids=known,
                contradictory_evidence_ids=contradictions,
                residual_uncertainty_bp=max(current.residual_uncertainty_bp, 1),
                disposition=QuestionDisposition.INVALIDATED,
                terminal_answer="",
            )
        change_kind = DecisionGraphChangeKind.CONTRADICTION_RECORDED
    else:
        affected = direct
        if set(evidence).intersection(state.contradictory_evidence_ids):
            raise DecisionGraphError(
                "evidence already classified as contradictory; invalidate it first"
            )
        disposition = state.disposition
        if disposition is QuestionDisposition.INVALIDATED:
            disposition = QuestionDisposition.UNRESOLVED
        states[semantic_id] = replace(
            state,
            known_evidence_ids=tuple(
                sorted(set(state.known_evidence_ids).union(evidence))
            ),
            disposition=disposition,
            terminal_answer=(
                state.terminal_answer
                if disposition is QuestionDisposition.RESOLVED
                else ""
            ),
        )
        change_kind = DecisionGraphChangeKind.EVIDENCE_RECORDED
    evidence_dependencies[semantic_id] = tuple(
        sorted(set(evidence_dependencies.get(semantic_id, ())).union(evidence))
    )
    return _transition(
        graph,
        states=states,
        dependencies=dependencies,
        evidence_dependencies=evidence_dependencies,
        tree_id=graph.tree_id,
        change_kind=change_kind,
        evidence_ids=evidence,
        direct=direct,
        affected=affected,
    )


def invalidate_evidence(
    graph: DecisionGraph,
    evidence_ids: Iterable[str],
    *,
    new_tree_id: str | None = None,
) -> DecisionGraphTransition:
    """Invalidate the exact evidence consumers and their dependent suffix."""

    evidence = tuple(sorted(set(evidence_ids)))
    if not evidence:
        raise DecisionGraphError(
            "at least one invalidated evidence identity is required"
        )
    states, dependencies, evidence_dependencies, _ = _semantic_state(graph)
    evidence_set = set(evidence)
    direct = {
        key
        for key, state in states.items()
        if evidence_set.intersection(state.known_evidence_ids)
        or evidence_set.intersection(state.contradictory_evidence_ids)
        or evidence_set.intersection(evidence_dependencies.get(key, ()))
    }
    affected = _dependency_descendants(dependencies, direct)
    for key in affected:
        state = states[key]
        states[key] = replace(
            state,
            known_evidence_ids=tuple(
                sorted(set(state.known_evidence_ids).difference(evidence_set))
            ),
            contradictory_evidence_ids=tuple(
                sorted(set(state.contradictory_evidence_ids).difference(evidence_set))
            ),
            residual_uncertainty_bp=max(state.residual_uncertainty_bp, 1),
            disposition=QuestionDisposition.INVALIDATED,
            terminal_answer="",
        )
        evidence_dependencies[key] = tuple(
            sorted(set(evidence_dependencies.get(key, ())).difference(evidence_set))
        )
    return _transition(
        graph,
        states=states,
        dependencies=dependencies,
        evidence_dependencies=evidence_dependencies,
        tree_id=new_tree_id or graph.tree_id,
        change_kind=DecisionGraphChangeKind.EVIDENCE_INVALIDATED,
        evidence_ids=evidence,
        direct=direct,
        affected=affected,
    )


def question_is_admissibly_terminal(question: DecisionQuestion) -> bool:
    """Fail-closed terminal predicate independent of model confidence."""

    return (
        question.disposition is QuestionDisposition.RESOLVED
        and bool(question.terminal_answer)
        and question.terminal_answer in question.current_alternatives
        and question.residual_uncertainty_bp == 0
        and not question.contradictory_evidence_ids
        and set(question.required_evidence_ids).issubset(question.known_evidence_ids)
    )


def graph_is_complete(graph: DecisionGraph) -> bool:
    """Return whether every mandatory question and its dependencies terminate."""

    by_id = {question.question_id: question for question in graph.questions}
    required: set[str] = {
        question.question_id for question in graph.questions if question.mandatory
    }
    if not required:
        # An empty decision graph is healthy exhaustion, not evidence that an
        # objective or task met its acceptance criteria.
        return False
    pending = list(required)
    while pending:
        question_id = pending.pop()
        for dependency_id in by_id[question_id].dependency_question_ids:
            if dependency_id not in required:
                required.add(dependency_id)
                pending.append(dependency_id)
    return all(question_is_admissibly_terminal(by_id[item]) for item in required)


def _deep_freeze_snapshot(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze_snapshot(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_snapshot(item) for item in value)
    return value


class DecisionGraphController:
    """Small in-memory facade whose durable unit is an immutable snapshot.

    Persistence, CAS, and leases intentionally remain outside this class.  A
    caller stores :meth:`snapshot_json` through the existing artifact/state
    authority and reconstructs the controller with :meth:`from_snapshot`.
    """

    def __init__(self, graph: DecisionGraph) -> None:
        if not isinstance(graph, DecisionGraph):
            raise DecisionGraphError("graph must be a DecisionGraph")
        # Reject semantic duplicates rather than silently changing a recovered
        # signed/content-addressed snapshot.
        _semantic_state(graph)
        self._graph = graph
        self._aliases = {
            question.question_id: semantic_question_id(question)
            for question in graph.questions
        }

    @classmethod
    def compile(cls, **kwargs: Any) -> DecisionGraphController:
        # Freeze one-shot iterables and retain their version IDs as convenience
        # aliases for the lifetime of this in-memory controller.
        try:
            raw_questions = tuple(
                islice(iter(kwargs.get("questions", ())), MAX_GRAPH_QUESTIONS + 1)
            )
        except TypeError as exc:
            raise DecisionGraphError("questions must be an iterable") from exc
        if len(raw_questions) > MAX_GRAPH_QUESTIONS:
            raise DecisionGraphError("decision graph contains too many questions")
        kwargs = {**kwargs, "questions": raw_questions}
        controller = cls(compile_decision_graph(**kwargs))
        controller._aliases.update(
            {
                question.question_id: semantic_question_id(question)
                for question in raw_questions
            }
        )
        return controller

    @classmethod
    def from_snapshot(
        cls, snapshot: DecisionGraph | Mapping[str, Any] | str | bytes
    ) -> DecisionGraphController:
        if isinstance(snapshot, DecisionGraph):
            graph = snapshot
        elif isinstance(snapshot, bytes):
            graph = DecisionGraph.from_json(snapshot.decode("utf-8"))
        elif isinstance(snapshot, str):
            graph = DecisionGraph.from_json(snapshot)
        elif isinstance(snapshot, Mapping):
            graph = DecisionGraph.from_dict(snapshot)
        else:
            raise DecisionGraphError("unsupported decision graph snapshot")
        return cls(graph)

    @property
    def graph(self) -> DecisionGraph:
        return self._graph

    @property
    def complete(self) -> bool:
        return graph_is_complete(self._graph)

    @property
    def unresolved_questions(self) -> tuple[DecisionQuestion, ...]:
        return tuple(
            question
            for question in self._graph.questions
            if not question_is_admissibly_terminal(question)
        )

    def snapshot(self) -> Mapping[str, Any]:
        snapshot = _deep_freeze_snapshot(self._graph.to_record())
        assert isinstance(snapshot, Mapping)
        return snapshot

    def snapshot_json(self) -> str:
        return canonical_json(self._graph.to_record())

    def _current_reference(self, question_id: str) -> str:
        if question_id in self._aliases:
            return self._aliases[question_id]
        return question_id

    def _accept(
        self, transition: DecisionGraphTransition
    ) -> DecisionGraphChangeReceipt:
        old_aliases = dict(self._aliases)
        self._graph = transition.graph
        self._aliases = {
            question.question_id: semantic_question_id(question)
            for question in self._graph.questions
        }
        # Retain prior version IDs while this controller instance lives.  A
        # restart deliberately restores only canonical snapshot identities.
        self._aliases.update(old_aliases)
        return transition.receipt

    def resolve(
        self,
        question_id: str,
        *,
        terminal_answer: str,
        evidence_ids: Iterable[str],
    ) -> DecisionGraphChangeReceipt:
        return self._accept(
            resolve_question(
                self._graph,
                self._current_reference(question_id),
                terminal_answer=terminal_answer,
                evidence_ids=evidence_ids,
            )
        )

    def record_evidence(
        self,
        question_id: str,
        *,
        evidence_ids: Iterable[str],
        contradictory: bool = False,
    ) -> DecisionGraphChangeReceipt:
        return self._accept(
            record_question_evidence(
                self._graph,
                self._current_reference(question_id),
                evidence_ids=evidence_ids,
                contradictory=contradictory,
            )
        )

    def invalidate_evidence(
        self,
        evidence_ids: Iterable[str],
        *,
        new_tree_id: str | None = None,
    ) -> DecisionGraphChangeReceipt:
        return self._accept(
            invalidate_evidence(self._graph, evidence_ids, new_tree_id=new_tree_id)
        )


__all__ = [
    "DecisionGraphChangeKind",
    "DecisionGraphChangeReceipt",
    "DecisionGraphController",
    "DecisionGraphError",
    "DecisionGraphTransition",
    "compile_decision_graph",
    "graph_is_complete",
    "invalidate_evidence",
    "question_is_admissibly_terminal",
    "record_question_evidence",
    "resolve_question",
    "semantic_question_id",
]
