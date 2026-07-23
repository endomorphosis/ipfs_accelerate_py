"""Reviewed templates for code-invariant proof obligations.

The registry in this module is deliberately closed and exact.  It is not a
natural-language classifier and it does not guess a template from symbol
names.  A caller may select a template by its reviewed identifier, or request
an exact code-shape match.  Unknown and multiply-matching shapes stay
unsupported.

Each template includes executable, fail-closed Python reference semantics.
Those predicates are useful as an oracle for mutation and fallback tests; a
successful predicate evaluation is not, by itself, a formal proof.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

from .formal_verification_contracts import canonical_json, content_identity


PROOF_OBLIGATION_TEMPLATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-obligation-template@1"
)
PROOF_OBLIGATION_TEMPLATE_REGISTRY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-obligation-template-registry@1"
)
TEMPLATE_REGISTRY_VERSION = "1"


class TemplateValidationError(ValueError):
    """A reviewed template declaration is malformed."""


class UnsupportedProofTemplateError(LookupError):
    """No reviewed template exactly supports the requested input."""


class AmbiguousProofTemplateError(LookupError):
    """More than one reviewed template exactly matches the request."""


class TemplateSelectionStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"


class ReviewedCodeShape(str, Enum):
    """Exact shape identifiers emitted by reviewed code-shape analyzers."""

    LEGAL_STATE_TRANSITION = "state_machine.transition_function"
    LEASE_UNIQUENESS_AND_FENCING = "lease_store.acquire_or_mutate"
    DAG_ACYCLICITY = "directed_acyclic_graph.edge_update"
    MERGE_IDEMPOTENCE = "merge_operator.apply"
    CACHE_KEY_COMPLETENESS = "content_cache.key_builder"
    EVIDENCE_FRESHNESS = "proof_evidence.validity_check"
    PROJECTION_EQUIVALENCE = "projection.materializer"
    UNSUPPORTED_PROOF_FAIL_CLOSED = "proof_dispatch.unsupported_gate"


ReferencePredicate = Callable[..., bool]


def _strings(
    values: Iterable[Any], *, field_name: str, required: bool = True
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        values = (values,)
    result = tuple(
        sorted({str(value).strip() for value in values if str(value).strip()})
    )
    if required and not result:
        raise TemplateValidationError(f"{field_name} must not be empty")
    return result


def _strict_mapping(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TemplateValidationError(f"{field_name} must be a mapping")
    # The contract canonicalizer rejects floats, non-string keys, and opaque
    # Python values.  Round-tripping also detaches mutable caller containers.
    import json

    try:
        result = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise TemplateValidationError(
            f"{field_name} must contain canonical JSON values"
        ) from exc
    if not isinstance(result, dict):  # pragma: no cover - guarded by dict()
        raise TemplateValidationError(f"{field_name} must be an object")
    return result


def _predicate_semantics(predicate: ReferencePredicate) -> str:
    """Return a location-independent digest input for a Python predicate."""

    if not callable(predicate):
        raise TemplateValidationError("python_reference_predicate must be callable")
    try:
        source = textwrap.dedent(inspect.getsource(predicate))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as exc:
        raise TemplateValidationError(
            "python_reference_predicate must have inspectable Python source"
        ) from exc
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _mapping_rows(value: Any) -> tuple[Mapping[str, Any], ...] | None:
    if isinstance(value, Mapping) or isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        rows = tuple(value)
    except TypeError:
        return None
    return rows if all(isinstance(row, Mapping) for row in rows) else None


def legal_state_transition(
    previous_state: Any,
    next_state: Any,
    allowed_transitions: Mapping[Any, Iterable[Any]],
) -> bool:
    """Whether an exact before/after state pair is in the reviewed relation."""

    try:
        allowed = allowed_transitions.get(previous_state, ())
        return next_state in allowed
    except (AttributeError, TypeError):
        return False


def lease_uniqueness_and_fencing(
    active_leases: Iterable[Mapping[str, Any]],
    *,
    resource_field: str = "resource_id",
    fencing_token_field: str = "fencing_token",
    mutation_fencing_token: Any = None,
    current_fencing_token: Any = None,
) -> bool:
    """Check single active ownership and exact fencing of a lease mutation."""

    rows = _mapping_rows(active_leases)
    if rows is None or not resource_field or not fencing_token_field:
        return False
    resources: set[Any] = set()
    try:
        for row in rows:
            if not bool(row.get("active", True)):
                continue
            resource = row[resource_field]
            token = row[fencing_token_field]
            if resource in resources or isinstance(token, bool) or not isinstance(token, int):
                return False
            resources.add(resource)
        return (
            not isinstance(mutation_fencing_token, bool)
            and isinstance(mutation_fencing_token, int)
            and mutation_fencing_token == current_fencing_token
        )
    except (KeyError, TypeError):
        return False


def dag_is_acyclic(adjacency: Mapping[Any, Iterable[Any]]) -> bool:
    """Check a finite adjacency mapping using a three-colour DFS."""

    if not isinstance(adjacency, Mapping):
        return False
    graph: dict[Any, tuple[Any, ...]] = {}
    try:
        for node, children in adjacency.items():
            if isinstance(children, (str, bytes, bytearray)):
                return False
            graph[node] = tuple(children)
        for children in tuple(graph.values()):
            for child in children:
                graph.setdefault(child, ())
    except (TypeError, RuntimeError):
        return False
    colour: dict[Any, int] = {}

    def visit(node: Any) -> bool:
        state = colour.get(node, 0)
        if state == 1:
            return False
        if state == 2:
            return True
        colour[node] = 1
        if not all(visit(child) for child in graph[node]):
            return False
        colour[node] = 2
        return True

    try:
        return all(visit(node) for node in graph)
    except (KeyError, TypeError, RecursionError):
        return False


def merge_is_idempotent(merge_once: Any, merge_twice: Any) -> bool:
    """Whether an observed merge result is unchanged by reapplication."""

    try:
        return type(merge_once) is type(merge_twice) and merge_once == merge_twice
    except BaseException:
        return False


def cache_key_is_complete(
    semantic_inputs: Mapping[str, Any],
    cache_key_inputs: Mapping[str, Any],
) -> bool:
    """Whether every semantic input is represented exactly in a cache key."""

    if not isinstance(semantic_inputs, Mapping) or not isinstance(
        cache_key_inputs, Mapping
    ):
        return False
    try:
        return all(
            key in cache_key_inputs and cache_key_inputs[key] == value
            for key, value in semantic_inputs.items()
        )
    except BaseException:
        return False


def evidence_is_fresh(
    expected_bindings: Mapping[str, Any],
    evidence_bindings: Mapping[str, Any],
    *,
    verified_at: int,
    not_before: int,
    expires_at: int,
) -> bool:
    """Whether evidence has exact semantic bindings and a valid time window."""

    if (
        not isinstance(expected_bindings, Mapping)
        or not isinstance(evidence_bindings, Mapping)
        or isinstance(verified_at, bool)
        or isinstance(not_before, bool)
        or isinstance(expires_at, bool)
        or not all(isinstance(value, int) for value in (verified_at, not_before, expires_at))
    ):
        return False
    return (
        dict(expected_bindings) == dict(evidence_bindings)
        and not_before <= verified_at <= expires_at
    )


def projection_is_equivalent(
    source: Mapping[str, Any],
    projection: Mapping[str, Any],
    projected_fields: Iterable[str],
) -> bool:
    """Whether a declared projection equals the corresponding source view."""

    if not isinstance(source, Mapping) or not isinstance(projection, Mapping):
        return False
    try:
        fields = tuple(projected_fields)
        if (
            not fields
            or any(not isinstance(field, str) or not field for field in fields)
            or set(projection) != set(fields)
        ):
            return False
        return all(field in source and projection[field] == source[field] for field in fields)
    except (TypeError, KeyError):
        return False


def unsupported_proof_fails_closed(
    support_status: Any,
    matching_template_ids: Iterable[str],
    *,
    gate_satisfied: bool,
) -> bool:
    """Whether only one explicitly supported template can satisfy a gate."""

    try:
        matches = tuple(
            value for value in matching_template_ids if isinstance(value, str) and value
        )
    except TypeError:
        return False
    should_satisfy = support_status == TemplateSelectionStatus.SUPPORTED.value and len(matches) == 1
    return gate_satisfied is should_satisfy


REFERENCE_PREDICATES: Mapping[str, ReferencePredicate] = {
    f"{predicate.__module__}.{predicate.__qualname__}": predicate
    for predicate in (
        legal_state_transition,
        lease_uniqueness_and_fencing,
        dag_is_acyclic,
        merge_is_idempotent,
        cache_key_is_complete,
        evidence_is_fresh,
        projection_is_equivalent,
        unsupported_proof_fails_closed,
    )
}


@dataclass(frozen=True)
class TemplateMutationCase:
    """A reviewed counterexample or positive control for reference semantics."""

    case_id: str
    description: str
    arguments: Mapping[str, Any]
    expected: bool

    def __post_init__(self) -> None:
        for name in ("case_id", "description"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise TemplateValidationError(f"mutation case {name} is required")
            object.__setattr__(self, name, value)
        if not isinstance(self.expected, bool):
            raise TemplateValidationError("mutation case expected must be boolean")
        object.__setattr__(
            self,
            "arguments",
            _strict_mapping(self.arguments, field_name="mutation case arguments"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "description": self.description,
            "arguments": dict(self.arguments),
            "expected": self.expected,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TemplateMutationCase":
        return cls(
            case_id=str(value.get("case_id") or ""),
            description=str(value.get("description") or ""),
            arguments=value.get("arguments") or {},
            expected=value.get("expected"),  # type: ignore[arg-type]
        )

# Compatibility spelling used by callers describing fixture mutations.
MutationCase = TemplateMutationCase


@dataclass(frozen=True)
class ProofObligationTemplate:
    """One versioned and reviewed invariant declaration."""

    template_id: str
    version: str
    invariant_class: str
    canonical_statement: str
    python_reference_predicate: ReferencePredicate = field(repr=False, compare=False)
    supported_backends: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    mutation_cases: tuple[TemplateMutationCase, ...] = ()
    fallback_tests: tuple[str, ...] = ()
    supported_code_shapes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("template_id", "version", "invariant_class", "canonical_statement"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise TemplateValidationError(f"{name} is required")
            object.__setattr__(self, name, value)
        predicate = self.python_reference_predicate
        _predicate_semantics(predicate)
        cases = tuple(
            case
            if isinstance(case, TemplateMutationCase)
            else TemplateMutationCase.from_dict(case)
            for case in self.mutation_cases
        )
        case_ids = [case.case_id for case in cases]
        if not cases or len(case_ids) != len(set(case_ids)):
            raise TemplateValidationError(
                "mutation_cases must be non-empty and have unique case ids"
            )
        object.__setattr__(self, "mutation_cases", tuple(sorted(cases, key=lambda item: item.case_id)))
        for name in (
            "supported_backends",
            "assumptions",
            "fallback_tests",
            "supported_code_shapes",
        ):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), field_name=name),
            )

    @property
    def reference_predicate(self) -> ReferencePredicate:
        return self.python_reference_predicate

    @property
    def template_version(self) -> str:
        return self.version

    @property
    def python_reference_predicate_id(self) -> str:
        return (
            f"{self.python_reference_predicate.__module__}."
            f"{self.python_reference_predicate.__qualname__}"
        )

    def _semantic_payload(self) -> dict[str, Any]:
        return {
            "schema": PROOF_OBLIGATION_TEMPLATE_SCHEMA,
            "template_id": self.template_id,
            "version": self.version,
            "invariant_class": self.invariant_class,
            "canonical_statement": self.canonical_statement,
            "python_reference_predicate_id": self.python_reference_predicate_id,
            "python_reference_semantics": _predicate_semantics(
                self.python_reference_predicate
            ),
            "supported_backends": self.supported_backends,
            "assumptions": self.assumptions,
            "mutation_cases": tuple(case.to_dict() for case in self.mutation_cases),
            "fallback_tests": self.fallback_tests,
            "supported_code_shapes": self.supported_code_shapes,
        }

    @property
    def semantic_hash(self) -> str:
        digest = hashlib.sha256(
            canonical_json(self._semantic_payload()).encode("utf-8")
        ).hexdigest()
        return f"sha256:{digest}"

    @property
    def template_semantic_hash(self) -> str:
        return self.semantic_hash

    @property
    def content_id(self) -> str:
        return content_identity(self._semantic_payload())

    @property
    def identity(self) -> str:
        return self.content_id

    def supports_backend(self, backend_id: str) -> bool:
        return str(backend_id or "").strip() in self.supported_backends

    def supports_code_shape(self, code_shape: str | ReviewedCodeShape) -> bool:
        value = str(getattr(code_shape, "value", code_shape) or "").strip()
        return value in self.supported_code_shapes

    def evaluate(self, **arguments: Any) -> bool:
        """Evaluate reference semantics; every exception fails closed."""

        try:
            result = self.python_reference_predicate(**arguments)
        except BaseException:
            return False
        return result if isinstance(result, bool) else False

    def verify_mutation_cases(self) -> dict[str, bool]:
        return {
            case.case_id: self.evaluate(**case.arguments) is case.expected
            for case in self.mutation_cases
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._semantic_payload()
        payload.pop("python_reference_semantics")
        payload.update(
            {
                "semantic_hash": self.semantic_hash,
                "content_id": self.content_id,
            }
        )
        return payload

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        predicates: Mapping[str, ReferencePredicate] | None = None,
    ) -> "ProofObligationTemplate":
        schema = str(payload.get("schema") or PROOF_OBLIGATION_TEMPLATE_SCHEMA)
        if schema != PROOF_OBLIGATION_TEMPLATE_SCHEMA:
            raise TemplateValidationError(f"unsupported template schema: {schema}")
        predicate_id = str(
            payload.get("python_reference_predicate_id") or ""
        ).strip()
        predicate_index = REFERENCE_PREDICATES if predicates is None else predicates
        predicate = predicate_index.get(predicate_id)
        if predicate is None:
            raise TemplateValidationError(
                f"unknown reviewed Python reference predicate: {predicate_id!r}"
            )
        result = cls(
            template_id=str(payload.get("template_id") or ""),
            version=str(
                payload.get("version") or payload.get("template_version") or ""
            ),
            invariant_class=str(payload.get("invariant_class") or ""),
            canonical_statement=str(payload.get("canonical_statement") or ""),
            python_reference_predicate=predicate,
            supported_backends=tuple(payload.get("supported_backends") or ()),
            assumptions=tuple(payload.get("assumptions") or ()),
            mutation_cases=tuple(
                TemplateMutationCase.from_dict(case)
                for case in payload.get("mutation_cases") or ()
            ),
            fallback_tests=tuple(payload.get("fallback_tests") or ()),
            supported_code_shapes=tuple(
                payload.get("supported_code_shapes") or ()
            ),
        )
        claimed_hash = str(
            payload.get("semantic_hash")
            or payload.get("template_semantic_hash")
            or ""
        )
        if claimed_hash and claimed_hash != result.semantic_hash:
            raise TemplateValidationError(
                "template semantic hash does not match declaration"
            )
        claimed_id = str(payload.get("content_id") or "")
        if claimed_id and claimed_id != result.content_id:
            raise TemplateValidationError(
                "template content identity does not match declaration"
            )
        return result

    @classmethod
    def from_json(
        cls,
        text: str,
        *,
        predicates: Mapping[str, ReferencePredicate] | None = None,
    ) -> "ProofObligationTemplate":
        import json

        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise TemplateValidationError("template JSON is malformed") from exc
        if not isinstance(payload, Mapping):
            raise TemplateValidationError("template JSON must be an object")
        return cls.from_dict(payload, predicates=predicates)


ObligationTemplate = ProofObligationTemplate
CodeProofObligationTemplate = ProofObligationTemplate


@dataclass(frozen=True)
class TemplateSelection:
    status: TemplateSelectionStatus
    template: ProofObligationTemplate | None = None
    candidate_template_ids: tuple[str, ...] = ()
    reason: str = ""

    @property
    def supported(self) -> bool:
        return self.status is TemplateSelectionStatus.SUPPORTED and self.template is not None

    def require_supported(self) -> ProofObligationTemplate:
        if self.supported:
            assert self.template is not None
            return self.template
        error = (
            AmbiguousProofTemplateError
            if self.status is TemplateSelectionStatus.AMBIGUOUS
            else UnsupportedProofTemplateError
        )
        raise error(self.reason or self.status.value)


class ProofObligationTemplateRegistry:
    """Immutable exact-match index of reviewed obligation templates."""

    def __init__(
        self,
        templates: Iterable[ProofObligationTemplate],
        *,
        registry_version: str = TEMPLATE_REGISTRY_VERSION,
    ) -> None:
        self.registry_version = str(registry_version or "").strip()
        if not self.registry_version:
            raise TemplateValidationError("registry_version is required")
        values = tuple(templates)
        if not values or not all(isinstance(item, ProofObligationTemplate) for item in values):
            raise TemplateValidationError(
                "registry templates must be non-empty ProofObligationTemplate values"
            )
        keys = [(item.template_id, item.version) for item in values]
        if len(keys) != len(set(keys)):
            raise TemplateValidationError("template id/version pairs must be unique")
        self._templates = tuple(
            sorted(values, key=lambda item: (item.template_id, item.version))
        )
        self._by_key = {
            (item.template_id, item.version): item for item in self._templates
        }

    @property
    def templates(self) -> tuple[ProofObligationTemplate, ...]:
        return self._templates

    @property
    def template_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.template_id for item in self._templates}))

    @property
    def registry_id(self) -> str:
        return content_identity(
            {
                "schema": PROOF_OBLIGATION_TEMPLATE_REGISTRY_SCHEMA,
                "registry_version": self.registry_version,
                "templates": tuple(
                    {
                        "template_id": item.template_id,
                        "version": item.version,
                        "semantic_hash": item.semantic_hash,
                    }
                    for item in self._templates
                ),
            }
        )

    def get(
        self, template_id: str, version: str | None = None
    ) -> ProofObligationTemplate | None:
        template_id = str(template_id or "").strip()
        if not template_id:
            return None
        if version is not None:
            return self._by_key.get((template_id, str(version).strip()))
        matches = [item for item in self._templates if item.template_id == template_id]
        return matches[0] if len(matches) == 1 else None

    def require(
        self, template_id: str, version: str | None = None
    ) -> ProofObligationTemplate:
        template = self.get(template_id, version)
        if template is not None:
            return template
        versions = [
            item.version for item in self._templates if item.template_id == template_id
        ]
        if len(versions) > 1 and version is None:
            raise AmbiguousProofTemplateError(
                f"template {template_id!r} has multiple versions; select one exactly"
            )
        suffix = f" version {version!r}" if version is not None else ""
        raise UnsupportedProofTemplateError(
            f"unknown reviewed template {template_id!r}{suffix}"
        )

    def select_for_code_shape(
        self,
        code_shape: str | ReviewedCodeShape,
        *,
        backend_id: str | None = None,
    ) -> TemplateSelection:
        shape = str(getattr(code_shape, "value", code_shape) or "").strip()
        if not shape:
            return TemplateSelection(
                status=TemplateSelectionStatus.UNSUPPORTED,
                reason="an exact reviewed code shape is required",
            )
        matches = [
            item
            for item in self._templates
            if item.supports_code_shape(shape)
            and (backend_id is None or item.supports_backend(backend_id))
        ]
        ids = tuple(sorted(f"{item.template_id}@{item.version}" for item in matches))
        if not matches:
            return TemplateSelection(
                status=TemplateSelectionStatus.UNSUPPORTED,
                reason=f"no reviewed template supports exact code shape {shape!r}",
            )
        if len(matches) != 1:
            return TemplateSelection(
                status=TemplateSelectionStatus.AMBIGUOUS,
                candidate_template_ids=ids,
                reason=f"multiple reviewed templates support exact code shape {shape!r}",
            )
        return TemplateSelection(
            status=TemplateSelectionStatus.SUPPORTED,
            template=matches[0],
            candidate_template_ids=ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_OBLIGATION_TEMPLATE_REGISTRY_SCHEMA,
            "registry_version": self.registry_version,
            "registry_id": self.registry_id,
            "templates": [item.to_dict() for item in self._templates],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        predicates: Mapping[str, ReferencePredicate] | None = None,
    ) -> "ProofObligationTemplateRegistry":
        schema = str(
            payload.get("schema") or PROOF_OBLIGATION_TEMPLATE_REGISTRY_SCHEMA
        )
        if schema != PROOF_OBLIGATION_TEMPLATE_REGISTRY_SCHEMA:
            raise TemplateValidationError(f"unsupported registry schema: {schema}")
        raw_templates = payload.get("templates")
        if not isinstance(raw_templates, Sequence) or isinstance(
            raw_templates, (str, bytes, bytearray)
        ):
            raise TemplateValidationError("registry templates must be a sequence")
        result = cls(
            (
                ProofObligationTemplate.from_dict(
                    item,
                    predicates=predicates,
                )
                for item in raw_templates
                if isinstance(item, Mapping)
            ),
            registry_version=str(payload.get("registry_version") or ""),
        )
        if len(result.templates) != len(raw_templates):
            raise TemplateValidationError("registry template entries must be objects")
        claimed_id = str(payload.get("registry_id") or "")
        if claimed_id and claimed_id != result.registry_id:
            raise TemplateValidationError(
                "template registry identity does not match declaration"
            )
        return result

    @classmethod
    def from_json(
        cls,
        text: str,
        *,
        predicates: Mapping[str, ReferencePredicate] | None = None,
    ) -> "ProofObligationTemplateRegistry":
        import json

        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise TemplateValidationError("template registry JSON is malformed") from exc
        if not isinstance(payload, Mapping):
            raise TemplateValidationError("template registry JSON must be an object")
        return cls.from_dict(payload, predicates=predicates)


TemplateRegistry = ProofObligationTemplateRegistry


def _case(
    case_id: str, description: str, expected: bool, **arguments: Any
) -> TemplateMutationCase:
    return TemplateMutationCase(case_id, description, arguments, expected)


_BACKENDS = ("lean4", "python-reference", "smtlib2")


DEFAULT_PROOF_OBLIGATION_TEMPLATES: tuple[ProofObligationTemplate, ...] = (
    ProofObligationTemplate(
        template_id="legal-state-transitions",
        version="1.0.0",
        invariant_class="state_transition",
        canonical_statement=(
            "Every observed state change belongs to the explicitly declared legal "
            "transition relation."
        ),
        python_reference_predicate=legal_state_transition,
        supported_backends=_BACKENDS,
        assumptions=(
            "States have stable equality semantics.",
            "The allowed transition relation is complete for the bound state machine.",
        ),
        mutation_cases=(
            _case(
                "legal-edge",
                "A declared transition is accepted.",
                True,
                previous_state="ready",
                next_state="running",
                allowed_transitions={"ready": ["running"]},
            ),
            _case(
                "undeclared-edge",
                "An undeclared transition is rejected.",
                False,
                previous_state="ready",
                next_state="complete",
                allowed_transitions={"ready": ["running"]},
            ),
        ),
        fallback_tests=(
            "pytest:state-machine-valid-transitions",
            "pytest:state-machine-invalid-transition-rejected",
        ),
        supported_code_shapes=(ReviewedCodeShape.LEGAL_STATE_TRANSITION.value,),
    ),
    ProofObligationTemplate(
        template_id="lease-uniqueness-and-fencing",
        version="1.0.0",
        invariant_class="lease_safety",
        canonical_statement=(
            "At most one active lease exists per resource and every lease mutation "
            "presents the exact current fencing token."
        ),
        python_reference_predicate=lease_uniqueness_and_fencing,
        supported_backends=_BACKENDS,
        assumptions=(
            "Lease rows are read from one serializable logical snapshot.",
            "Fencing tokens are monotonic non-boolean integers.",
        ),
        mutation_cases=(
            _case(
                "unique-current-token",
                "Unique leases with the current token are accepted.",
                True,
                active_leases=[
                    {"resource_id": "r1", "fencing_token": 4, "active": True}
                ],
                mutation_fencing_token=4,
                current_fencing_token=4,
            ),
            _case(
                "duplicate-active-lease",
                "Two active leases for one resource are rejected.",
                False,
                active_leases=[
                    {"resource_id": "r1", "fencing_token": 4, "active": True},
                    {"resource_id": "r1", "fencing_token": 5, "active": True},
                ],
                mutation_fencing_token=5,
                current_fencing_token=5,
            ),
            _case(
                "stale-fencing-token",
                "A stale mutation token is rejected.",
                False,
                active_leases=[],
                mutation_fencing_token=3,
                current_fencing_token=4,
            ),
        ),
        fallback_tests=(
            "pytest:lease-single-winner",
            "pytest:lease-stale-fencing-token-rejected",
        ),
        supported_code_shapes=(
            ReviewedCodeShape.LEASE_UNIQUENESS_AND_FENCING.value,
        ),
    ),
    ProofObligationTemplate(
        template_id="dag-acyclicity",
        version="1.0.0",
        invariant_class="dag_acyclicity",
        canonical_statement=(
            "The directed dependency graph contains no cycle after the proposed edge "
            "or node mutation."
        ),
        python_reference_predicate=dag_is_acyclic,
        supported_backends=_BACKENDS,
        assumptions=(
            "The adjacency mapping is a complete finite graph snapshot.",
            "Node equality and hashing are stable during evaluation.",
        ),
        mutation_cases=(
            _case(
                "acyclic-chain",
                "A finite chain is accepted.",
                True,
                adjacency={"a": ["b"], "b": ["c"], "c": []},
            ),
            _case(
                "back-edge",
                "A cycle introduced by a back edge is rejected.",
                False,
                adjacency={"a": ["b"], "b": ["a"]},
            ),
        ),
        fallback_tests=("pytest:dag-cycle-rejected", "pytest:dag-topological-order"),
        supported_code_shapes=(ReviewedCodeShape.DAG_ACYCLICITY.value,),
    ),
    ProofObligationTemplate(
        template_id="merge-idempotence",
        version="1.0.0",
        invariant_class="merge_idempotence",
        canonical_statement=(
            "Reapplying the same merge to its result does not change the canonical "
            "merged value."
        ),
        python_reference_predicate=merge_is_idempotent,
        supported_backends=_BACKENDS,
        assumptions=(
            "Both observations use identical merge inputs and configuration.",
            "Compared values are in the same canonical representation.",
        ),
        mutation_cases=(
            _case(
                "stable-reapplication",
                "An unchanged second merge is accepted.",
                True,
                merge_once={"items": ["a", "b"]},
                merge_twice={"items": ["a", "b"]},
            ),
            _case(
                "duplicate-on-reapplication",
                "A duplicate introduced by reapplication is rejected.",
                False,
                merge_once=["a"],
                merge_twice=["a", "a"],
            ),
        ),
        fallback_tests=(
            "pytest:merge-reapplication",
            "property-test:merge-idempotence",
        ),
        supported_code_shapes=(ReviewedCodeShape.MERGE_IDEMPOTENCE.value,),
    ),
    ProofObligationTemplate(
        template_id="cache-key-completeness",
        version="1.0.0",
        invariant_class="cache_key_completeness",
        canonical_statement=(
            "Every semantic input that can affect the cached result participates "
            "exactly in the cache-key identity."
        ),
        python_reference_predicate=cache_key_is_complete,
        supported_backends=_BACKENDS,
        assumptions=(
            "The semantic input inventory is complete.",
            "Input values are compared in their canonical representation.",
        ),
        mutation_cases=(
            _case(
                "all-inputs-bound",
                "A key binding every semantic input is accepted.",
                True,
                semantic_inputs={"source": "h1", "template": "h2"},
                cache_key_inputs={"source": "h1", "template": "h2"},
            ),
            _case(
                "template-version-omitted",
                "A key omitting a semantic template input is rejected.",
                False,
                semantic_inputs={"source": "h1", "template": "h2"},
                cache_key_inputs={"source": "h1"},
            ),
        ),
        fallback_tests=(
            "pytest:cache-key-dimension-mutations",
            "property-test:cache-input-injectivity",
        ),
        supported_code_shapes=(ReviewedCodeShape.CACHE_KEY_COMPLETENESS.value,),
    ),
    ProofObligationTemplate(
        template_id="evidence-freshness",
        version="1.0.0",
        invariant_class="evidence_freshness",
        canonical_statement=(
            "Accepted evidence binds the exact current semantic inputs and its "
            "verification time lies within the declared validity interval."
        ),
        python_reference_predicate=evidence_is_fresh,
        supported_backends=_BACKENDS,
        assumptions=(
            "Binding values and timestamps come from authenticated evidence metadata.",
            "Timestamps share one integer time scale.",
        ),
        mutation_cases=(
            _case(
                "current-binding",
                "Exact bindings inside the validity interval are accepted.",
                True,
                expected_bindings={"tree": "t1", "scope": "s1"},
                evidence_bindings={"tree": "t1", "scope": "s1"},
                verified_at=20,
                not_before=10,
                expires_at=30,
            ),
            _case(
                "stale-tree",
                "Evidence for an earlier tree is rejected.",
                False,
                expected_bindings={"tree": "t2"},
                evidence_bindings={"tree": "t1"},
                verified_at=20,
                not_before=10,
                expires_at=30,
            ),
            _case(
                "expired-evidence",
                "Evidence outside its validity interval is rejected.",
                False,
                expected_bindings={"tree": "t1"},
                evidence_bindings={"tree": "t1"},
                verified_at=31,
                not_before=10,
                expires_at=30,
            ),
        ),
        fallback_tests=(
            "pytest:evidence-binding-mismatch",
            "pytest:evidence-expiration-boundaries",
        ),
        supported_code_shapes=(ReviewedCodeShape.EVIDENCE_FRESHNESS.value,),
    ),
    ProofObligationTemplate(
        template_id="projection-equivalence",
        version="1.0.0",
        invariant_class="projection_equivalence",
        canonical_statement=(
            "The materialized projection contains exactly the declared fields and "
            "each projected value equals its source value."
        ),
        python_reference_predicate=projection_is_equivalent,
        supported_backends=_BACKENDS,
        assumptions=(
            "The source snapshot and projection belong to the same revision.",
            "Projected field paths have unambiguous equality semantics.",
        ),
        mutation_cases=(
            _case(
                "exact-projection",
                "An exact declared projection is accepted.",
                True,
                source={"id": "a", "state": "ready", "private": 1},
                projection={"id": "a", "state": "ready"},
                projected_fields=["id", "state"],
            ),
            _case(
                "projection-drift",
                "A changed projected value is rejected.",
                False,
                source={"id": "a", "state": "ready"},
                projection={"id": "a", "state": "done"},
                projected_fields=["id", "state"],
            ),
        ),
        fallback_tests=(
            "pytest:projection-round-trip",
            "property-test:projection-field-equivalence",
        ),
        supported_code_shapes=(ReviewedCodeShape.PROJECTION_EQUIVALENCE.value,),
    ),
    ProofObligationTemplate(
        template_id="unsupported-proof-fail-closed",
        version="1.0.0",
        invariant_class="unsupported_proof",
        canonical_statement=(
            "A proof gate is satisfied only when exactly one reviewed template "
            "explicitly supports the code shape; unsupported or ambiguous selections "
            "remain unsatisfied."
        ),
        python_reference_predicate=unsupported_proof_fails_closed,
        supported_backends=_BACKENDS,
        assumptions=(
            "Support status is produced by exact registry matching.",
            "Gate satisfaction is authoritative and defaults to false.",
        ),
        mutation_cases=(
            _case(
                "one-supported-template",
                "One supported reviewed template may satisfy the gate.",
                True,
                support_status="supported",
                matching_template_ids=["template@1"],
                gate_satisfied=True,
            ),
            _case(
                "unknown-shape-blocked",
                "An unsupported shape cannot satisfy the gate.",
                True,
                support_status="unsupported",
                matching_template_ids=[],
                gate_satisfied=False,
            ),
            _case(
                "ambiguous-shape-not-accepted",
                "An ambiguous selection claiming satisfaction is rejected.",
                False,
                support_status="ambiguous",
                matching_template_ids=["a@1", "b@1"],
                gate_satisfied=True,
            ),
        ),
        fallback_tests=(
            "pytest:unknown-code-shape-is-unsupported",
            "pytest:ambiguous-code-shape-is-unsupported",
        ),
        supported_code_shapes=(
            ReviewedCodeShape.UNSUPPORTED_PROOF_FAIL_CLOSED.value,
        ),
    ),
)

DEFAULT_TEMPLATE_REGISTRY = ProofObligationTemplateRegistry(
    DEFAULT_PROOF_OBLIGATION_TEMPLATES
)
DEFAULT_PROOF_OBLIGATION_TEMPLATE_REGISTRY = DEFAULT_TEMPLATE_REGISTRY


def get_proof_obligation_template(
    template_id: str,
    version: str | None = None,
    *,
    registry: ProofObligationTemplateRegistry = DEFAULT_TEMPLATE_REGISTRY,
) -> ProofObligationTemplate | None:
    return registry.get(template_id, version)


def require_proof_obligation_template(
    template_id: str,
    version: str | None = None,
    *,
    registry: ProofObligationTemplateRegistry = DEFAULT_TEMPLATE_REGISTRY,
) -> ProofObligationTemplate:
    return registry.require(template_id, version)


def select_proof_obligation_template(
    code_shape: str | ReviewedCodeShape,
    *,
    backend_id: str | None = None,
    registry: ProofObligationTemplateRegistry = DEFAULT_TEMPLATE_REGISTRY,
) -> TemplateSelection:
    return registry.select_for_code_shape(code_shape, backend_id=backend_id)


__all__ = [
    "AmbiguousProofTemplateError",
    "CodeProofObligationTemplate",
    "DEFAULT_PROOF_OBLIGATION_TEMPLATES",
    "DEFAULT_PROOF_OBLIGATION_TEMPLATE_REGISTRY",
    "DEFAULT_TEMPLATE_REGISTRY",
    "MutationCase",
    "ObligationTemplate",
    "PROOF_OBLIGATION_TEMPLATE_REGISTRY_SCHEMA",
    "PROOF_OBLIGATION_TEMPLATE_SCHEMA",
    "ProofObligationTemplate",
    "ProofObligationTemplateRegistry",
    "REFERENCE_PREDICATES",
    "ReferencePredicate",
    "ReviewedCodeShape",
    "TEMPLATE_REGISTRY_VERSION",
    "TemplateMutationCase",
    "TemplateRegistry",
    "TemplateSelection",
    "TemplateSelectionStatus",
    "TemplateValidationError",
    "UnsupportedProofTemplateError",
    "cache_key_is_complete",
    "dag_is_acyclic",
    "evidence_is_fresh",
    "get_proof_obligation_template",
    "lease_uniqueness_and_fencing",
    "legal_state_transition",
    "merge_is_idempotent",
    "projection_is_equivalent",
    "require_proof_obligation_template",
    "select_proof_obligation_template",
    "unsupported_proof_fails_closed",
]
