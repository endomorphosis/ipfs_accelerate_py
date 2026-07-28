"""Provider-aware, token-budgeted supervisor context compilation.

The compiler in this module deliberately separates the non-truncatable
context core (goal, authority, scope, and acceptance) from ranked evidence.
It negotiates the usable input limit with the effective provider, emits a
decision for every evidence item, and represents omitted material with
bounded expansion references instead of copying source bodies into receipts.

Retry contexts are parent-bound deltas.  A delta may contain only changed or
explicitly requested evidence; :func:`reconstruct_context` deterministically
rebuilds the effective context and verifies that the immutable core, tree,
policy, and required evidence coverage were not weakened.

Prefix-stable contexts retain that base contract while rendering an ordered
policy/objective prefix, task core, and evidence delta.  Their cache identities
bind provider/model, authority, target, and every semantic prefix dependency;
warm receipts distinguish provider-native reuse from conservative estimates.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, ClassVar, Final

from .context_contracts import (
    ContextBudget,
    ContextBudgetResolution,
    ContextCapsule,
    ContextContractError,
    ContextDeltaCapsule,
    ContextReference,
    ContextTier,
    canonical_context_json_bytes,
)
from .formal_verification_contracts import CanonicalContract


REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID: Final = (
    "208290439421789408250562066350459701853"
)
REQUIRED_CONTEXT_OBJECTIVE_ID: Final = "ASI-G091"
REQUIRED_CONTEXT_ACCEPTANCE_CRITERIA: Final[tuple[str, ...]] = (
    (
        "The compiler derives the effective input limit from the supervisor "
        "ceiling, provider input/window limits, and reserved output/tool tokens"
    ),
    (
        "counts the complete canonical provider input and never trusts a "
        "caller-declared reference token hint below the canonical descriptor cost"
    ),
    (
        "preserves the complete invariant core and every required reference "
        "or rejects compilation"
    ),
    "refuses to defer required evidence as an expansion handle",
    (
        "orders optional material deterministically with explicit "
        "inclusion/omission reasons and bounded expansion handles"
    ),
    (
        "and emits the exact requirement ID only in a witness whose repository "
        "tree, objective, policy, effective budget, required and selected "
        "fields/references, capsule identity, result, and content digest are "
        "revalidated against the capsule. End-to-end promotion remains "
        "ineligible unless capsule-verified compiler results cover the complete "
        "same terminal accepted task population, exactly reconcile charged "
        "candidate input tokens, and retain the authoritative required-coverage "
        "set while the paired 35 percent gate passes."
    ),
)
DELTA_RETRY_EVIDENCE_ID: Final = (
    "306437607356117177048620815571362227127"
)
PREFIX_REUSE_REQUIREMENT_ID: Final = (
    "267664298677617945522201159534035798321"
)
VALUE_OF_INFORMATION_REQUIREMENT_ID: Final = (
    "224169380537603827401044344943410282193"
)
VALUE_OF_INFORMATION_OBJECTIVE_ID: Final = "ASI-G210"
VALUE_OF_INFORMATION_ACCEPTANCE_CRITERIA: Final[tuple[str, ...]] = (
    (
        "Optional evidence is ranked by expected decision change and "
        "uncertainty reduction divided by measured token, latency, "
        "invalidation, and expansion costs"
    ),
    (
        "required context is never auctioned away; exclusions, residual "
        "uncertainty, and diversity penalties remain explicit"
    ),
    (
        "on-demand expansion names an unresolved question and resolves an "
        "exact content-addressed parent handle"
    ),
    (
        "paired fixtures preserve accepted-criterion denominators, required "
        "coverage, and safety while reducing median input tokens per accepted "
        "criterion by at least 40 percent and retry-input tokens by at least "
        "60 percent"
    ),
)
PREFIX_REUSE_OBJECTIVE_ID: Final = "ASI-G210"
PREFIX_REUSE_ACCEPTANCE_CRITERIA: Final[tuple[str, ...]] = (
    (
        "Every provider input is arranged as a canonical stable "
        "policy/objective prefix, stable task core, and volatile evidence delta"
    ),
    (
        "the stable prefix preserves goal, authority, scope, and acceptance and "
        "its cache identity changes exactly with a semantic prefix dependency"
    ),
    (
        "provider prompt-cache or KV-cache identities and actual reused tokens "
        "are bound when available; otherwise warm reuse is conservatively estimated"
    ),
    (
        "reuse is prohibited across provider, model, authority, repository, "
        "tree, stage, or target-scope boundaries"
    ),
    (
        "and the exact requirement ID is emitted only for a warm, evidence-bound "
        "receipt that reuses at least 70 percent of eligible stable-prefix tokens"
    ),
)
DELTA_RETRY_OBJECTIVE_ID: Final = "ASI-G092"
DELTA_RETRY_ACCEPTANCE_CRITERIA: Final[tuple[str, ...]] = (
    (
        "A compact retry references its exact parent capsule without replaying "
        "the invariant core and transmits only deterministic changed or newly "
        "requested evidence"
    ),
    (
        "applying it reconstructs goal, authority, scope, acceptance, deferred "
        "expansion handles, omission diagnostics, and all required evidence "
        "without loss or requiredness downgrade. Reconstructed input accounting "
        "includes the inherited core and every retained or replaced reference "
        "and fails closed above the effective budget"
    ),
    "changed and requested-but-unchanged references remain distinct",
    "stale parents and forged counts or digests fail closed",
    (
        "and canonical provider-tokenized delta input is smaller than canonical "
        "full replay. The exact requirement ID is emitted only in a witness "
        "binding the repository tree, policy, parent, delta, and reconstructed "
        "identities, changed/requested/retained references, required fields and "
        "coverage, token counts, result, and content digest. A "
        "population-complete same-task promotion report must consume "
        "compiler-backed `ContextDeltaResult` values rather than receipt-only "
        "claims, rerun canonical provider-token measurement, retain full required "
        "coverage, exactly reconcile every charged lifecycle input token without "
        "an unattributed remainder, and meet the 35 percent median per-task "
        "input-token reduction gate before promotion."
    ),
)
CONTEXT_EVIDENCE_PRODUCERS: Final = {
    REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID: "context_compiler",
    DELTA_RETRY_EVIDENCE_ID: "context_delta_compiler",
    PREFIX_REUSE_REQUIREMENT_ID: "prefix_context_compiler",
    VALUE_OF_INFORMATION_REQUIREMENT_ID: "value_of_information_compiler",
}

CONTEXT_COMPILATION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/context-compilation-receipt@2"
)
CONTEXT_DELTA_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/context-delta-receipt@1"
)
RETRY_CONTEXT_CAPSULE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/retry-context-capsule@1"
)
REQUIRED_CONTEXT_BUDGET_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "required-context-budget-evidence@2"
)
DELTA_RETRY_CONTEXT_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/delta-retry-context-evidence@1"
)
PREFIX_CACHE_IDENTITY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prefix-cache-identity@1"
)
PREFIX_STABLE_CONTEXT_CAPSULE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prefix-stable-context-capsule@1"
)
PREFIX_REUSE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prefix-reuse-receipt@1"
)
VALUE_OF_INFORMATION_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/value-of-information-evidence@1"
)
EVIDENCE_VALUE_FIXTURE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/evidence-value-paired-fixture@1"
)
CONTEXT_COMPILER_VERSION = 1
MAX_DECISIONS = 4_096
MAX_CALIBRATION_SAMPLES = 128
MAX_ERROR_BPS = 1_000_000
MIN_WARM_PREFIX_REUSE_BPS = 7_000
CONSERVATIVE_PREFIX_REUSE_BPS = 8_000
MIN_INPUT_TOKEN_REDUCTION_BPS = 4_000
MIN_RETRY_INPUT_TOKEN_REDUCTION_BPS = 6_000
DEFAULT_DIVERSITY_PENALTY_BPS = 5_000
MAX_VALUE_BPS = 10_000


class ContextCompilationError(ContextContractError):
    """Base error raised when a safe context cannot be compiled."""


class RequiredContextOverflowError(ContextCompilationError):
    """The invariant core or explicitly required evidence does not fit."""


class ContextDeltaError(ContextCompilationError):
    """A retry delta is stale, lossy, unchanged, or not token efficient."""


class PrefixContextError(ContextCompilationError):
    """A prefix-stable provider input or cache claim is malformed."""


class PrefixCacheBoundaryError(PrefixContextError):
    """Provider cache reuse crossed a semantic or provider boundary."""


class ContextExpansionError(ContextDeltaError):
    """An on-demand context expansion could not be verified."""


class MissingContextReferenceError(ContextExpansionError):
    """A requested handle or its content-addressed object is unavailable."""


class ChangedTreeContextError(ContextExpansionError):
    """A parent capsule belongs to a repository tree that is no longer current."""


class ContextExpansionCancelled(ContextExpansionError):
    """Expansion was cancelled before a complete verified result was built."""


class InclusionReason(str, Enum):
    """Why an evidence reference was included."""

    REQUIRED = "required"
    RANKED_FIT = "ranked_fit"
    CHANGED = "changed"
    REQUESTED = "requested"


class ExclusionReason(str, Enum):
    """Why an evidence reference was not transmitted."""

    TOKEN_BUDGET = "token_budget"
    ITEM_LIMIT = "item_limit"
    UNCHANGED = "unchanged"
    NOT_REQUESTED = "not_requested"
    LOW_VALUE = "low_value"


class PrefixReuseSource(str, Enum):
    """How stable-prefix reuse was measured."""

    COLD = "cold"
    PROVIDER_PROMPT_CACHE = "provider_prompt_cache"
    PROVIDER_KV_CACHE = "provider_kv_cache"
    CONSERVATIVE_ESTIMATE = "conservative_estimate"


class PrefixCacheDecision(str, Enum):
    """Outcome of comparing a stage input with its warm predecessor."""

    COLD = "cold"
    HIT = "hit"
    MISS = "miss"
    INVALIDATED = "invalidated"


class PrefixCacheKind(str, Enum):
    """Provider cache mechanism, or the deterministic local cache key."""

    DERIVED = "derived"
    PROMPT_CACHE = "prompt_cache"
    KV_CACHE = "kv_cache"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ContextCompilationError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise ContextCompilationError(f"{name} must not be empty")
    if "\x00" in result or len(result.encode("utf-8")) > 8_192:
        raise ContextCompilationError(f"{name} is not bounded text")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContextCompilationError(
            f"{name} must be an integer of at least {minimum}"
        )
    return value


def _basis_points(value: Any, name: str, *, default: int = 0) -> int:
    """Normalize a probability/fraction or explicit basis-point integer."""

    if value in (None, ""):
        return default
    if isinstance(value, bool):
        raise ContextCompilationError(f"{name} must be numeric")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        if value < 0.0 or value > 1.0:
            raise ContextCompilationError(
                f"{name} fractions must be between zero and one"
            )
        result = round(value * MAX_VALUE_BPS)
    else:
        raise ContextCompilationError(f"{name} must be numeric")
    if result < 0 or result > MAX_VALUE_BPS:
        raise ContextCompilationError(
            f"{name} must be between zero and {MAX_VALUE_BPS} basis points"
        )
    return result


def _metadata_integer(
    metadata: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    default: int = 0,
) -> int:
    for name in names:
        if name not in metadata:
            continue
        value = metadata[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContextCompilationError(
                f"evidence metadata {name} must be a non-negative integer"
            )
        return value
    return default


def _metadata_basis_points(
    metadata: Mapping[str, Any],
    bps_name: str,
    fraction_name: str,
    *,
    default: int = 0,
) -> int:
    if bps_name in metadata:
        return _basis_points(metadata[bps_name], bps_name)
    if fraction_name in metadata:
        return _basis_points(metadata[fraction_name], fraction_name)
    return default


def _strings(
    value: Iterable[Any],
    name: str,
    *,
    maximum: int = MAX_DECISIONS,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ContextCompilationError(f"{name} must be a sequence")
    result: set[str] = set()
    for index, item in enumerate(value):
        if index >= maximum:
            raise ContextCompilationError(f"{name} exceeds its item limit")
        result.add(_text(item, name))
    return tuple(sorted(result))


def _digest(value: Any, name: str) -> str:
    result = _text(value, name)
    raw = result.removeprefix("sha256:")
    if len(raw) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in raw):
        raise ContextCompilationError(f"{name} must be a SHA-256 digest")
    return "sha256:" + raw.lower()


def _canonical_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_context_json_bytes(value)).hexdigest()


def _reject_unknown(
    payload: Mapping[str, Any], allowed: set[str], noun: str
) -> None:
    if not isinstance(payload, Mapping):
        raise ContextCompilationError(f"{noun} must be an object")
    if set(payload).difference(allowed):
        raise ContextCompilationError(
            f"{noun} contains unsupported fields; rebuild its canonical payload"
        )


def _schema(payload: Mapping[str, Any], expected: str, noun: str) -> None:
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise ContextCompilationError(
            f"unsupported {noun} schema; rebuild the canonical payload"
        )
    version = payload.get("contract_version")
    if version not in (None, CONTEXT_COMPILER_VERSION):
        raise ContextCompilationError(
            f"unsupported {noun} contract version"
        )


def _check_identity(
    payload: Mapping[str, Any], actual: str, noun: str
) -> None:
    claimed = payload.get("content_id") or payload.get("receipt_id")
    if claimed not in (None, "", actual):
        raise ContextCompilationError(f"{noun} identity does not match payload")


def _coerce_references(
    value: Iterable[ContextReference | Mapping[str, Any]],
) -> tuple[ContextReference, ...]:
    result: dict[str, ContextReference] = {}
    for index, raw in enumerate(value):
        if index >= MAX_DECISIONS:
            raise ContextCompilationError(
                "evidence exceeds its reference-count limit"
            )
        item = (
            raw
            if isinstance(raw, ContextReference)
            else ContextReference.from_dict(raw)
            if isinstance(raw, Mapping)
            else None
        )
        if item is None:
            raise ContextCompilationError("evidence contains an invalid reference")
        if item.tier is ContextTier.EXPANSION:
            raise ContextCompilationError(
                "candidate evidence cannot use the expansion tier"
            )
        previous = result.get(item.reference_id)
        if previous is not None and previous != item:
            raise ContextCompilationError(
                "evidence contains conflicting duplicate reference IDs"
            )
        result[item.reference_id] = item
    return tuple(result[key] for key in sorted(result))


def _as_expansion(
    reference: ContextReference,
    *,
    exclusion_reason: ExclusionReason | str | None = None,
) -> ContextReference:
    metadata = dict(reference.metadata)
    if any(
        name in metadata
        for name in (
            "expected_decision_change_bps",
            "expected_decision_change",
            "uncertainty_reduction_bps",
            "uncertainty_reduction",
        )
    ):
        metadata["question_bound_expansion"] = True
    if exclusion_reason is not None:
        metadata["selection_exclusion_reason"] = str(
            getattr(exclusion_reason, "value", exclusion_reason)
        )
    return ContextReference(
        reference_id=reference.reference_id,
        kind=reference.kind,
        tier=ContextTier.EXPANSION,
        referenced_content_id=reference.referenced_content_id,
        repository_id=reference.repository_id,
        tree_id=reference.tree_id,
        path=reference.path,
        summary=reference.summary,
        byte_count=reference.byte_count,
        token_count=reference.token_count,
        metadata=metadata,
    )


def _capsule_omission_reason(reason: ExclusionReason) -> ExclusionReason:
    """Project rich selection reasons into the legacy capsule vocabulary.

    The receipt and expansion handle retain the exact reason.  A base
    ``ContextCapsule`` currently supports resource-limit omission codes, so a
    low-value exclusion uses ``token_budget`` only in that compatibility field.
    """

    if reason is ExclusionReason.LOW_VALUE:
        return ExclusionReason.TOKEN_BUDGET
    return reason


def _cancelled(value: Any) -> bool:
    """Evaluate a bool, event, or zero-argument cancellation predicate."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if callable(value):
        return bool(value())
    checker = getattr(value, "is_set", None)
    if callable(checker):
        return bool(checker())
    raise ContextExpansionError(
        "cancelled must be a boolean, predicate, event, or None"
    )


def _utf8_chunks(value: str, *, max_bytes: int) -> tuple[str, ...]:
    """Split text at semantic boundaries without dropping or corrupting bytes.

    The boundary is a resource limit rather than a prompt slice: every chunk
    is content addressed and can independently become selected evidence or an
    on-demand expansion handle.  Newlines are preferred, then other
    whitespace, with a Unicode code-point boundary as the final fallback.
    """

    if not isinstance(value, str):
        raise ContextCompilationError("text must be a string")
    if "\x00" in value:
        raise ContextCompilationError("text must not contain NUL")
    text = value.strip()
    limit = _integer(max_bytes, "max_bytes", minimum=1)
    if not text:
        return ()
    chunks: list[str] = []
    remaining = text
    while len(remaining.encode("utf-8")) > limit:
        if len(remaining[0].encode("utf-8")) > limit:
            raise ContextCompilationError(
                "max_bytes is smaller than one UTF-8 code point"
            )
        low = 1
        high = len(remaining)
        while low < high:
            midpoint = (low + high + 1) // 2
            if len(remaining[:midpoint].encode("utf-8")) <= limit:
                low = midpoint
            else:
                high = midpoint - 1
        boundary = low
        prefix = remaining[:boundary]
        semantic = prefix.rfind("\n")
        if semantic < 1:
            semantic = max(prefix.rfind(" "), prefix.rfind("\t"))
        if semantic >= 1:
            boundary = semantic + 1
        chunk = remaining[:boundary].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[boundary:].lstrip()
    if remaining.strip():
        chunks.append(remaining.strip())
    return tuple(chunks)


def build_text_context_references(
    text: str,
    *,
    reference_prefix: str,
    kind: str,
    path: str = "",
    repository_id: str = "",
    tree_id: str = "",
    tier: ContextTier | str = ContextTier.EVIDENCE,
    priority: int = 0,
    required: bool = False,
    chunk_bytes: int = 6_144,
    coverage_ids: Iterable[str] = (),
) -> tuple[ContextReference, ...]:
    """Build deterministic, independently expandable references for text.

    Bodies are carried only in bounded reference summaries sent to a provider.
    Receipts retain reference identities and selection decisions, never these
    summaries.  The digest of the complete artifact binds every chunk without
    copying the full artifact or a recursive structure into receipt metadata.
    """

    prefix = _text(reference_prefix, "reference_prefix")
    reference_kind = _text(kind, "kind")
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise ContextCompilationError("priority must be an integer")
    try:
        selected_tier = (
            tier if isinstance(tier, ContextTier) else ContextTier(str(tier))
        )
    except ValueError as exc:
        raise ContextCompilationError("tier is not a supported context tier") from exc
    if required:
        selected_tier = ContextTier.INVARIANT
    chunks = _utf8_chunks(text, max_bytes=chunk_bytes)
    if len(chunks) > MAX_DECISIONS:
        raise ContextCompilationError(
            "text artifact exceeds its context reference-count limit"
        )
    artifact_content_id = _canonical_digest({"text": text})
    normalized_coverage = _strings(
        coverage_ids, "coverage_ids", maximum=MAX_DECISIONS
    )
    result: list[ContextReference] = []
    for index, chunk in enumerate(chunks):
        chunk_content_id = "sha256:" + hashlib.sha256(
            chunk.encode("utf-8")
        ).hexdigest()
        result.append(
            ContextReference(
                reference_id=f"{prefix}:{index + 1:04d}",
                kind=reference_kind,
                tier=selected_tier,
                referenced_content_id=chunk_content_id,
                repository_id=repository_id,
                tree_id=tree_id,
                path=path,
                summary=chunk,
                byte_count=len(chunk.encode("utf-8")),
                metadata={
                    "required": bool(required),
                    "priority": priority,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "artifact_content_id": artifact_content_id,
                    "coverage_ids": normalized_coverage,
                },
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class EvidenceExpansionRequest:
    """A bounded progressive-disclosure request tied to one open question."""

    unresolved_question: str
    reference_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "unresolved_question",
            _text(self.unresolved_question, "unresolved_question"),
        )
        references = _strings(self.reference_ids, "reference_ids")
        if not references:
            raise MissingContextReferenceError(
                "an evidence expansion request requires a reference ID"
            )
        object.__setattr__(self, "reference_ids", references)


class ContentAddressedContextStore:
    """Bounded in-memory artifact store for verified progressive disclosure.

    The store is deliberately a resolver, not a cache of prompt envelopes.
    Handles remain in :class:`ContextCapsule`; artifact bytes are fetched only
    when their reference IDs are explicitly requested.  Every fetch verifies
    the SHA-256 target in the handle and its repository/tree binding before a
    descriptor can enter a retry delta.
    """

    def __init__(
        self,
        *,
        max_artifact_bytes: int = 1_048_576,
        max_artifacts: int = MAX_DECISIONS,
    ) -> None:
        self.max_artifact_bytes = _integer(
            max_artifact_bytes, "max_artifact_bytes", minimum=1
        )
        self.max_artifacts = _integer(
            max_artifacts, "max_artifacts", minimum=1
        )
        self._objects: dict[str, bytes] = {}

    @staticmethod
    def content_id(content: str | bytes) -> str:
        if isinstance(content, str):
            raw = content.encode("utf-8")
        elif isinstance(content, bytes):
            raw = content
        else:
            raise ContextExpansionError("context artifact must be text or bytes")
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    def put(self, content: str | bytes) -> str:
        """Store an immutable object and return its content identity."""

        raw = content.encode("utf-8") if isinstance(content, str) else content
        if not isinstance(raw, bytes):
            raise ContextExpansionError("context artifact must be text or bytes")
        if len(raw) > self.max_artifact_bytes:
            raise ContextExpansionError("context artifact exceeds its byte limit")
        content_id = self.content_id(raw)
        if content_id not in self._objects and len(self._objects) >= self.max_artifacts:
            raise ContextExpansionError("context artifact store is full")
        self._objects.setdefault(content_id, raw)
        return content_id

    def get(self, content_id: str) -> bytes:
        """Return an object by exact identity or fail with a typed miss."""

        identity = _digest(content_id, "content_id")
        try:
            raw = self._objects[identity]
        except KeyError as exc:
            raise MissingContextReferenceError(
                f"context artifact {identity!r} is unavailable"
            ) from exc
        if self.content_id(raw) != identity:
            # Defensive even for an in-memory store: callers may provide a
            # custom mutable mapping in tests or a future persistent backend.
            raise ContextExpansionError(
                "stored context artifact does not match its content identity"
            )
        return raw

    def make_reference(
        self,
        content: str,
        *,
        reference_id: str,
        kind: str,
        repository_id: str,
        tree_id: str,
        path: str = "",
        priority: int = 0,
        coverage_ids: Iterable[str] = (),
        unresolved_questions: Iterable[str] = (),
    ) -> ContextReference:
        """Store text and return a compact expansion handle for it."""

        if not isinstance(content, str):
            raise ContextExpansionError("expandable context must be UTF-8 text")
        target = self.put(content)
        questions = _strings(
            unresolved_questions,
            "unresolved_questions",
            maximum=64,
        )
        return ContextReference(
            reference_id=reference_id,
            kind=kind,
            tier=ContextTier.EXPANSION,
            referenced_content_id=target,
            repository_id=repository_id,
            tree_id=tree_id,
            path=path,
            byte_count=len(content.encode("utf-8")),
            metadata={
                "priority": priority,
                "coverage_ids": _strings(
                    coverage_ids, "coverage_ids", maximum=MAX_DECISIONS
                ),
                "unresolved_questions": questions,
                "question_bound_expansion": bool(questions),
            },
        )

    def resolve(
        self,
        handle: ContextReference,
        *,
        unresolved_question: str = "",
        cancelled: Any = None,
    ) -> ContextReference:
        """Resolve one expansion handle and verify its complete binding."""

        if _cancelled(cancelled):
            raise ContextExpansionCancelled("context expansion was cancelled")
        if not isinstance(handle, ContextReference):
            raise ContextExpansionError("expansion handle must be a ContextReference")
        if handle.tier is not ContextTier.EXPANSION:
            raise ContextExpansionError("only expansion-tier handles can be resolved")
        allowed_questions = handle.metadata.get("unresolved_questions", ())
        if isinstance(allowed_questions, str):
            allowed_questions = (allowed_questions,)
        if not isinstance(allowed_questions, (tuple, list)):
            raise ContextExpansionError(
                "expansion handle unresolved questions are malformed"
            )
        normalized_question = _text(
            unresolved_question,
            "unresolved_question",
            required=False,
        )
        if handle.metadata.get("question_bound_expansion", False):
            if not normalized_question:
                raise ContextExpansionError(
                    "question-bound expansion requires a named unresolved question"
                )
            if allowed_questions and normalized_question not in allowed_questions:
                raise ContextExpansionError(
                    "unresolved question is not bound to the expansion handle"
                )
        raw = self.get(handle.referenced_content_id)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContextExpansionError(
                "model-facing context artifact is not UTF-8 text"
            ) from exc
        if handle.byte_count not in (0, len(raw)):
            raise ContextExpansionError(
                "expansion handle byte count does not match its artifact"
            )
        if _cancelled(cancelled):
            raise ContextExpansionCancelled("context expansion was cancelled")
        metadata = dict(handle.metadata)
        if normalized_question:
            metadata["expansion_question"] = normalized_question
        return ContextReference(
            reference_id=handle.reference_id,
            kind=handle.kind,
            tier=ContextTier.EVIDENCE,
            referenced_content_id=handle.referenced_content_id,
            repository_id=handle.repository_id,
            tree_id=handle.tree_id,
            path=handle.path,
            summary=text,
            byte_count=len(raw),
            # The handle's token count can describe a larger source artifact.
            # The compiler independently tokenizes this exact resolved
            # descriptor, so carrying that coarse hint would make a bounded
            # on-demand fragment impossible to select.
            token_count=0,
            metadata=metadata,
        )

    def expand(
        self,
        compiler: "ContextCompiler",
        parent: ContextCapsule,
        reference_ids: Iterable[str],
        *,
        repository_id: str | None = None,
        tree_id: str | None = None,
        unresolved_question: str = "",
        cancelled: Any = None,
    ) -> "ContextDeltaResult":
        """Resolve named parent handles and compile one lossless retry delta."""

        if not isinstance(compiler, ContextCompiler):
            raise ContextExpansionError("compiler must be a ContextCompiler")
        if not isinstance(parent, ContextCapsule):
            raise ContextExpansionError("parent must be a ContextCapsule")
        if _cancelled(cancelled):
            raise ContextExpansionCancelled("context expansion was cancelled")
        current_repository = _text(
            repository_id or parent.repository_id, "repository_id"
        )
        current_tree = _text(tree_id or parent.tree_id, "tree_id")
        if current_repository != parent.repository_id:
            raise ChangedTreeContextError(
                "parent repository identity is no longer current"
            )
        if current_tree != parent.tree_id:
            raise ChangedTreeContextError(
                "parent repository tree changed; compile a new base context"
            )
        requested = _strings(reference_ids, "reference_ids")
        if not requested:
            raise MissingContextReferenceError(
                "at least one expansion reference ID is required"
            )
        handles = {
            item.reference_id: item for item in parent.expansion_references
        }
        missing = set(requested).difference(handles)
        if missing:
            raise MissingContextReferenceError(
                "requested context handle is not present in the parent capsule: "
                + ", ".join(sorted(missing))
            )
        resolved: list[ContextReference] = []
        for reference_id in requested:
            if _cancelled(cancelled):
                raise ContextExpansionCancelled("context expansion was cancelled")
            item = self.resolve(
                handles[reference_id],
                unresolved_question=unresolved_question,
                cancelled=cancelled,
            )
            if (
                item.referenced_content_id
                != handles[reference_id].referenced_content_id
            ):
                raise ContextExpansionError(
                    "resolved context does not match its expansion handle"
                )
            resolved.append(item)
        if _cancelled(cancelled):
            raise ContextExpansionCancelled("context expansion was cancelled")
        return compiler.compile_delta(
            parent,
            evidence=(*parent.evidence, *resolved),
            requested_reference_ids=requested,
        )


@dataclass(frozen=True)
class RetryContextCapsule(CanonicalContract):
    """Model-facing semantic envelope for one implementation repair.

    It authenticates the existing low-level context delta while making the
    retry semantics explicit.  The original goal, policy prose, authority,
    scope, and acceptance bodies are absent; they are inherited through the
    exact parent context identity.
    """

    SCHEMA: ClassVar[str] = RETRY_CONTEXT_CAPSULE_SCHEMA

    prior_decision_id: str
    diagnostic_receipt_id: str
    repository_id: str
    tree_id: str
    delta_capsule: ContextDeltaCapsule
    failure_evidence_ids: tuple[str, ...]
    counterexample_evidence_ids: tuple[str, ...] = ()
    changed_files: tuple[str, ...] = ()
    changed_symbols: tuple[str, ...] = ()
    unresolved_requirement_ids: tuple[str, ...] = ()
    repair_round: int = 1
    max_repair_rounds: int = 3

    def __post_init__(self) -> None:
        for name in (
            "prior_decision_id",
            "diagnostic_receipt_id",
            "repository_id",
            "tree_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        delta = self.delta_capsule
        if isinstance(delta, Mapping):
            delta = ContextDeltaCapsule.from_dict(delta)
        if not isinstance(delta, ContextDeltaCapsule):
            raise ContextDeltaError(
                "retry delta_capsule must be a ContextDeltaCapsule"
            )
        object.__setattr__(self, "delta_capsule", delta)
        for name in (
            "failure_evidence_ids",
            "counterexample_evidence_ids",
            "changed_files",
            "changed_symbols",
            "unresolved_requirement_ids",
        ):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, maximum=MAX_DECISIONS),
            )
        if not self.failure_evidence_ids and not self.counterexample_evidence_ids:
            raise ContextDeltaError(
                "retry context requires new failure or counterexample evidence"
            )
        for path in self.changed_files:
            if path.startswith("/") or ".." in path.split("/"):
                raise ContextDeltaError(
                    "changed_files must contain repository-relative paths"
                )
        repair_round = _integer(self.repair_round, "repair_round", minimum=1)
        maximum = _integer(
            self.max_repair_rounds, "max_repair_rounds", minimum=1
        )
        if repair_round > maximum:
            raise ContextDeltaError("retry repair round exceeds its bound")
        object.__setattr__(self, "repair_round", repair_round)
        object.__setattr__(self, "max_repair_rounds", maximum)

    @property
    def parent_capsule_id(self) -> str:
        return self.delta_capsule.parent_capsule_id

    @property
    def capsule_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_COMPILER_VERSION,
            "prior_decision_id": self.prior_decision_id,
            "diagnostic_receipt_id": self.diagnostic_receipt_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "delta_capsule": self.delta_capsule.to_record(),
            "failure_evidence_ids": self.failure_evidence_ids,
            "counterexample_evidence_ids": self.counterexample_evidence_ids,
            "changed_files": self.changed_files,
            "changed_symbols": self.changed_symbols,
            "unresolved_requirement_ids": self.unresolved_requirement_ids,
            "repair_round": self.repair_round,
            "max_repair_rounds": self.max_repair_rounds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RetryContextCapsule":
        _schema(payload, cls.SCHEMA, "retry context capsule")
        _reject_unknown(
            payload,
            {
                "schema",
                "content_id",
                "contract_version",
                "prior_decision_id",
                "diagnostic_receipt_id",
                "repository_id",
                "tree_id",
                "delta_capsule",
                "failure_evidence_ids",
                "counterexample_evidence_ids",
                "changed_files",
                "changed_symbols",
                "unresolved_requirement_ids",
                "repair_round",
                "max_repair_rounds",
            },
            "retry context capsule",
        )
        delta = payload.get("delta_capsule")
        result = cls(
            prior_decision_id=payload.get("prior_decision_id", ""),
            diagnostic_receipt_id=payload.get("diagnostic_receipt_id", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            delta_capsule=(
                ContextDeltaCapsule.from_dict(delta)
                if isinstance(delta, Mapping)
                else delta
            ),
            failure_evidence_ids=tuple(payload.get("failure_evidence_ids", ())),
            counterexample_evidence_ids=tuple(
                payload.get("counterexample_evidence_ids", ())
            ),
            changed_files=tuple(payload.get("changed_files", ())),
            changed_symbols=tuple(payload.get("changed_symbols", ())),
            unresolved_requirement_ids=tuple(
                payload.get("unresolved_requirement_ids", ())
            ),
            repair_round=payload.get("repair_round", 1),
            max_repair_rounds=payload.get("max_repair_rounds", 3),
        )
        _check_identity(payload, result.content_id, "retry context capsule")
        return result


@dataclass(frozen=True)
class RetryContextResult:
    """Verified semantic retry capsule plus its exact reconstruction."""

    capsule: RetryContextCapsule
    delta_result: "ContextDeltaResult"

    def __post_init__(self) -> None:
        if not isinstance(self.capsule, RetryContextCapsule):
            raise ContextDeltaError("retry capsule has an invalid type")
        if not isinstance(self.delta_result, ContextDeltaResult):
            raise ContextDeltaError("retry delta result has an invalid type")
        if self.capsule.delta_capsule != self.delta_result.delta_capsule:
            raise ContextDeltaError("retry capsule is not bound to its delta result")
        parent = self.delta_result.parent_capsule
        if (
            self.capsule.repository_id != parent.repository_id
            or self.capsule.tree_id != parent.tree_id
        ):
            raise ChangedTreeContextError(
                "retry capsule repository tree does not match its parent"
            )

    @property
    def reconstructed_capsule(self) -> ContextCapsule:
        return self.delta_result.reconstructed_capsule

    @property
    def receipt(self) -> "ContextDeltaReceipt":
        return self.delta_result.receipt


def render_context_capsule(capsule: ContextCapsule) -> str:
    """Render exactly the canonical provider input measured by the compiler.

    Supervisor-only receipts, omission diagnostics, and deferred bodies are
    intentionally absent.  This keeps the dispatched prompt deterministic and
    makes its token count independently reproducible from the capsule.
    """

    if not isinstance(capsule, ContextCapsule):
        raise ContextCompilationError("capsule must be a ContextCapsule")
    return canonical_context_json_bytes(
        context_provider_input_payload(
            repository_id=capsule.repository_id,
            tree_id=capsule.tree_id,
            objective_id=capsule.objective_id,
            objective_revision=capsule.objective_revision,
            policy_id=capsule.policy_id,
            policy_revision=capsule.policy_revision,
            caller=capsule.caller,
            stage=capsule.stage,
            goal=capsule.goal,
            authority=capsule.authority,
            scope=capsule.scope,
            acceptance=capsule.acceptance,
            evidence=capsule.evidence,
        )
    ).decode("utf-8")


def render_retry_context(capsule: RetryContextCapsule) -> str:
    """Render only the canonical semantic delta sent for a repair round."""

    if not isinstance(capsule, RetryContextCapsule):
        raise ContextDeltaError("capsule must be a RetryContextCapsule")
    return canonical_context_json_bytes(capsule.to_record()).decode("utf-8")


@dataclass(frozen=True)
class EvidenceValueEstimate:
    """Deterministic marginal value and cost projection for one reference."""

    expected_decision_change_bps: int
    uncertainty_bps: int
    uncertainty_reduction_bps: int
    token_cost: int
    latency_cost: int
    invalidation_cost: int
    expansion_cost: int
    diversity_key: str
    diversity_penalty_bps: int
    raw_value_score: int
    value_score: int
    explicit: bool

    @property
    def total_cost(self) -> int:
        return max(
            1,
            self.token_cost
            + self.latency_cost
            + self.invalidation_cost
            + self.expansion_cost,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_decision_change_bps": self.expected_decision_change_bps,
            "uncertainty_bps": self.uncertainty_bps,
            "uncertainty_reduction_bps": self.uncertainty_reduction_bps,
            "token_cost": self.token_cost,
            "latency_cost": self.latency_cost,
            "invalidation_cost": self.invalidation_cost,
            "expansion_cost": self.expansion_cost,
            "total_cost": self.total_cost,
            "diversity_key": self.diversity_key,
            "diversity_penalty_bps": self.diversity_penalty_bps,
            "raw_value_score": self.raw_value_score,
            "value_score": self.value_score,
            "explicit": self.explicit,
        }


@dataclass(frozen=True)
class EvidenceValuePolicy:
    """Fixed-point value-of-information policy for optional evidence."""

    minimum_value_score: int = 1
    diversity_penalty_bps: int = DEFAULT_DIVERSITY_PENALTY_BPS
    max_optional_items: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_value_score",
            _integer(self.minimum_value_score, "minimum_value_score"),
        )
        object.__setattr__(
            self,
            "diversity_penalty_bps",
            _basis_points(
                self.diversity_penalty_bps,
                "diversity_penalty_bps",
            ),
        )
        if self.max_optional_items is not None:
            object.__setattr__(
                self,
                "max_optional_items",
                _integer(
                    self.max_optional_items,
                    "max_optional_items",
                    minimum=1,
                ),
            )

    @staticmethod
    def _diversity_key(reference: ContextReference) -> str:
        metadata = reference.metadata
        raw = metadata.get(
            "diversity_key",
            metadata.get("evidence_family", metadata.get("redundancy_group", "")),
        )
        if raw not in (None, ""):
            return _text(str(raw), "diversity_key")
        if reference.coverage_ids:
            return "coverage:" + "|".join(reference.coverage_ids)
        return ""

    def estimate(
        self,
        reference: ContextReference,
        *,
        token_cost: int,
        selected_diversity_count: int = 0,
    ) -> EvidenceValueEstimate:
        if not isinstance(reference, ContextReference):
            raise ContextCompilationError(
                "value estimation requires a ContextReference"
            )
        tokens = _integer(token_cost, "token_cost", minimum=1)
        count = _integer(
            selected_diversity_count,
            "selected_diversity_count",
        )
        metadata = reference.metadata
        explicit = any(
            name in metadata
            for name in (
                "expected_decision_change_bps",
                "expected_decision_change",
                "uncertainty_reduction_bps",
                "uncertainty_reduction",
            )
        )
        # Existing priority becomes a deterministic prior for legacy callers,
        # but still pays the complete measured and declared cost denominator.
        default_change = (
            0
            if explicit
            else max(1, min(MAX_VALUE_BPS, 5_000 + reference.priority * 100))
        )
        expected_change = _metadata_basis_points(
            metadata,
            "expected_decision_change_bps",
            "expected_decision_change",
            default=default_change,
        )
        uncertainty = _metadata_basis_points(
            metadata,
            "uncertainty_bps",
            "uncertainty",
        )
        uncertainty_reduction = _metadata_basis_points(
            metadata,
            "uncertainty_reduction_bps",
            "uncertainty_reduction",
        )
        if uncertainty and uncertainty_reduction > uncertainty:
            raise ContextCompilationError(
                "uncertainty reduction cannot exceed current uncertainty"
            )
        latency = _metadata_integer(
            metadata,
            ("latency_cost", "latency_cost_units"),
        )
        invalidation = _metadata_integer(
            metadata,
            ("invalidation_cost", "invalidation_cost_units"),
        )
        expansion = _metadata_integer(
            metadata,
            ("expansion_cost", "expansion_cost_units"),
        )
        total_cost = max(1, tokens + latency + invalidation + expansion)
        raw_score = (
            (expected_change + uncertainty_reduction) * 1_000_000
        ) // total_cost
        diversity_key = self._diversity_key(reference)
        penalty = (
            min(MAX_VALUE_BPS, self.diversity_penalty_bps * count)
            if diversity_key
            else 0
        )
        adjusted_score = (
            raw_score * MAX_VALUE_BPS // (MAX_VALUE_BPS + penalty)
        )
        return EvidenceValueEstimate(
            expected_decision_change_bps=expected_change,
            uncertainty_bps=uncertainty,
            uncertainty_reduction_bps=uncertainty_reduction,
            token_cost=tokens,
            latency_cost=latency,
            invalidation_cost=invalidation,
            expansion_cost=expansion,
            diversity_key=diversity_key,
            diversity_penalty_bps=penalty,
            raw_value_score=raw_score,
            value_score=adjusted_score,
            explicit=explicit,
        )


@dataclass(frozen=True)
class EvidenceSelectionDecision:
    """Deterministic selection audit entry for one evidence reference."""

    reference_id: str
    included: bool
    reason: InclusionReason | ExclusionReason | str
    token_count: int
    priority: int = 0
    expected_decision_change_bps: int = 0
    uncertainty_bps: int = 0
    uncertainty_reduction_bps: int = 0
    latency_cost: int = 0
    invalidation_cost: int = 0
    expansion_cost: int = 0
    value_score: int = 0
    diversity_key: str = ""
    diversity_penalty_bps: int = 0
    unresolved_question: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reference_id", _text(self.reference_id, "reference_id")
        )
        if not isinstance(self.included, bool):
            raise ContextCompilationError("included must be a boolean")
        enum_type = InclusionReason if self.included else ExclusionReason
        try:
            reason = (
                self.reason
                if isinstance(self.reason, enum_type)
                else enum_type(str(getattr(self.reason, "value", self.reason)))
            )
        except ValueError as exc:
            raise ContextCompilationError(
                "selection reason does not match inclusion state"
            ) from exc
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self, "token_count", _integer(self.token_count, "token_count")
        )
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ContextCompilationError("priority must be an integer")
        for name in (
            "expected_decision_change_bps",
            "uncertainty_bps",
            "uncertainty_reduction_bps",
            "diversity_penalty_bps",
        ):
            object.__setattr__(
                self,
                name,
                _basis_points(getattr(self, name), name),
            )
        if self.uncertainty_bps and (
            self.uncertainty_reduction_bps > self.uncertainty_bps
        ):
            raise ContextCompilationError(
                "uncertainty reduction cannot exceed current uncertainty"
            )
        for name in (
            "latency_cost",
            "invalidation_cost",
            "expansion_cost",
            "value_score",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "diversity_key",
            _text(self.diversity_key, "diversity_key", required=False),
        )
        object.__setattr__(
            self,
            "unresolved_question",
            _text(
                self.unresolved_question,
                "unresolved_question",
                required=False,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "included": self.included,
            "reason": self.reason.value,
            "token_count": self.token_count,
            "priority": self.priority,
            "expected_decision_change_bps": self.expected_decision_change_bps,
            "uncertainty_bps": self.uncertainty_bps,
            "uncertainty_reduction_bps": self.uncertainty_reduction_bps,
            "latency_cost": self.latency_cost,
            "invalidation_cost": self.invalidation_cost,
            "expansion_cost": self.expansion_cost,
            "value_score": self.value_score,
            "diversity_key": self.diversity_key,
            "diversity_penalty_bps": self.diversity_penalty_bps,
            "unresolved_question": self.unresolved_question,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EvidenceSelectionDecision":
        fields = {
            "reference_id",
            "included",
            "reason",
            "token_count",
            "priority",
            "expected_decision_change_bps",
            "uncertainty_bps",
            "uncertainty_reduction_bps",
            "latency_cost",
            "invalidation_cost",
            "expansion_cost",
            "value_score",
            "diversity_key",
            "diversity_penalty_bps",
            "unresolved_question",
        }
        _reject_unknown(payload, fields, "selection decision")
        return cls(
            reference_id=payload.get("reference_id", ""),
            included=payload.get("included", False),
            reason=payload.get("reason", ""),
            token_count=payload.get("token_count", 0),
            priority=payload.get("priority", 0),
            expected_decision_change_bps=payload.get(
                "expected_decision_change_bps", 0
            ),
            uncertainty_bps=payload.get("uncertainty_bps", 0),
            uncertainty_reduction_bps=payload.get(
                "uncertainty_reduction_bps", 0
            ),
            latency_cost=payload.get("latency_cost", 0),
            invalidation_cost=payload.get("invalidation_cost", 0),
            expansion_cost=payload.get("expansion_cost", 0),
            value_score=payload.get("value_score", 0),
            diversity_key=payload.get("diversity_key", ""),
            diversity_penalty_bps=payload.get("diversity_penalty_bps", 0),
            unresolved_question=payload.get("unresolved_question", ""),
        )


class CalibratedTokenEstimator:
    """Use a provider tokenizer when available, otherwise a calibrated fallback."""

    def __init__(
        self,
        tokenizer: Callable[[str], Any] | Any | None = None,
        *,
        chars_per_token: int = 4,
    ) -> None:
        self._tokenizer = tokenizer
        self._chars_per_token = _integer(
            chars_per_token, "chars_per_token", minimum=1
        )
        self._samples: list[tuple[int, int]] = []

    @property
    def provider_aware(self) -> bool:
        return self._tokenizer is not None

    @property
    def name(self) -> str:
        return "provider_tokenizer" if self.provider_aware else "calibrated_utf8"

    @property
    def calibration_samples(self) -> int:
        return len(self._samples)

    @property
    def error_bps(self) -> int:
        if self.provider_aware:
            return 0
        if not self._samples:
            return 10_000
        total_actual = sum(actual for _, actual in self._samples)
        if total_actual == 0:
            return 0
        absolute_error = sum(
            abs(estimated - actual) for estimated, actual in self._samples
        )
        return min(MAX_ERROR_BPS, absolute_error * 10_000 // total_actual)

    def _provider_count(self, text: str) -> int:
        tokenizer = self._tokenizer
        if callable(tokenizer):
            result = tokenizer(text)
        elif hasattr(tokenizer, "encode"):
            result = tokenizer.encode(text)
        else:
            raise ContextCompilationError(
                "tokenizer must be callable or expose encode()"
            )
        if isinstance(result, bool):
            raise ContextCompilationError("provider tokenizer returned a boolean")
        if isinstance(result, int):
            return _integer(result, "provider token count")
        try:
            return len(result)
        except TypeError as exc:
            raise ContextCompilationError(
                "provider tokenizer returned an uncountable value"
            ) from exc

    def estimate(self, value: str | bytes | Any) -> int:
        if isinstance(value, bytes):
            text = value.decode("utf-8")
        elif isinstance(value, str):
            text = value
        else:
            text = canonical_context_json_bytes(value).decode("utf-8")
        if self.provider_aware:
            return self._provider_count(text)
        byte_count = len(text.encode("utf-8"))
        raw = max(1, (byte_count + self._chars_per_token - 1) // self._chars_per_token)
        if not self._samples:
            return raw
        estimated_total = sum(estimated for estimated, _ in self._samples)
        actual_total = sum(actual for _, actual in self._samples)
        if estimated_total == 0:
            return raw
        return max(1, (raw * actual_total + estimated_total - 1) // estimated_total)

    count = estimate

    def calibrate(self, value: str | bytes | Any, actual_tokens: int) -> None:
        actual = _integer(actual_tokens, "actual_tokens")
        if self.provider_aware:
            return
        if isinstance(value, bytes):
            byte_count = len(value)
        elif isinstance(value, str):
            byte_count = len(value.encode("utf-8"))
        else:
            byte_count = len(canonical_context_json_bytes(value))
        estimated = max(
            1, (byte_count + self._chars_per_token - 1) // self._chars_per_token
        )
        self._samples.append((estimated, actual))
        del self._samples[:-MAX_CALIBRATION_SAMPLES]


def _reference_tokens(
    estimator: CalibratedTokenEstimator, reference: ContextReference
) -> int:
    # A producer-supplied token count is a useful conservative hint, never an
    # authority boundary.  Always tokenize the canonical descriptor as well
    # so a large reference cannot claim ``token_count=1`` and escape the
    # effective provider budget.
    return max(
        reference.token_count,
        estimator.estimate(reference.to_record()),
    )


def _core_payload(
    *,
    goal: Any,
    authority: Any,
    scope: Any,
    acceptance: Any,
) -> dict[str, Any]:
    return {
        "goal": goal,
        "authority": authority,
        "scope": scope,
        "acceptance": acceptance,
    }


def context_provider_input_payload(
    *,
    repository_id: str,
    tree_id: str,
    objective_id: str,
    objective_revision: str,
    policy_id: str,
    policy_revision: str,
    caller: str,
    stage: str,
    goal: Any,
    authority: Any,
    scope: Any,
    acceptance: Any,
    evidence: Iterable[ContextReference] = (),
) -> dict[str, Any]:
    """Return the canonical authority-bearing provider input.

    This is deliberately separate from :class:`ContextCapsule`'s supervisor
    accounting envelope.  It includes every binding and selected reference
    sent to the provider while excluding deferred expansion handles and
    supervisor-only omission diagnostics.
    """

    return {
        "contract_version": CONTEXT_COMPILER_VERSION,
        "repository_id": repository_id,
        "tree_id": tree_id,
        "objective_id": objective_id,
        "objective_revision": objective_revision,
        "policy_id": policy_id,
        "policy_revision": policy_revision,
        "caller": caller,
        "stage": stage,
        **_core_payload(
            goal=goal,
            authority=authority,
            scope=scope,
            acceptance=acceptance,
        ),
        "evidence": tuple(item.to_record() for item in evidence),
    }


def _stable_policy_objective_prefix(
    capsule: ContextCapsule,
) -> dict[str, Any]:
    return {
        "contract_version": CONTEXT_COMPILER_VERSION,
        "repository_id": capsule.repository_id,
        "objective_id": capsule.objective_id,
        "objective_revision": capsule.objective_revision,
        "policy_id": capsule.policy_id,
        "policy_revision": capsule.policy_revision,
        "goal": capsule.goal,
        "authority": capsule.authority,
        "acceptance": capsule.acceptance,
    }


def _stable_task_core(capsule: ContextCapsule) -> dict[str, Any]:
    return {
        "tree_id": capsule.tree_id,
        "caller": capsule.caller,
        "stage": capsule.stage,
        "scope": capsule.scope,
    }


def _prefix_segment_bytes(label: str, value: Any) -> bytes:
    return canonical_context_json_bytes({label: value}) + b"\n"


def _stable_prefix_bytes(capsule: ContextCapsule) -> bytes:
    return _prefix_segment_bytes(
        "stable_policy_objective_prefix",
        _stable_policy_objective_prefix(capsule),
    ) + _prefix_segment_bytes(
        "stable_task_core", _stable_task_core(capsule)
    )


def _volatile_evidence_bytes(capsule: ContextCapsule) -> bytes:
    return canonical_context_json_bytes(
        {
            "volatile_evidence_delta": tuple(
                item.to_record() for item in capsule.evidence
            )
        }
    )


def _prefix_provider_input_tokens(
    estimator: "CalibratedTokenEstimator",
    capsule: ContextCapsule,
) -> int:
    stable = _stable_prefix_bytes(capsule)
    canonical_count = estimator.estimate(
        stable + _volatile_evidence_bytes(capsule)
    )
    empty_delta_count = estimator.estimate(
        stable
        + canonical_context_json_bytes({"volatile_evidence_delta": ()})
    )
    component_count = empty_delta_count + sum(
        _reference_tokens(estimator, item) for item in capsule.evidence
    )
    return max(canonical_count, component_count)


def _prefix_dependency_values(
    capsule: "PrefixStableContextCapsule",
) -> dict[str, Any]:
    base = capsule.context_capsule
    return {
        "provider_id": capsule.provider_id,
        "model_id": capsule.model_id,
        "repository_id": base.repository_id,
        "tree_id": base.tree_id,
        "objective_id": base.objective_id,
        "objective_revision": base.objective_revision,
        "policy_id": base.policy_id,
        "policy_revision": base.policy_revision,
        "caller": base.caller,
        "stage": base.stage,
        "goal": base.goal,
        "authority": base.authority,
        "scope": base.scope,
        "acceptance": base.acceptance,
    }


@dataclass(frozen=True)
class PrefixCacheIdentity(CanonicalContract):
    """Provider/model-bound identity for one exact reusable prefix."""

    SCHEMA: ClassVar[str] = PREFIX_CACHE_IDENTITY_SCHEMA

    provider_id: str
    model_id: str
    cache_kind: PrefixCacheKind | str
    semantic_prefix_id: str
    authority_boundary_id: str
    target_boundary_id: str
    provider_cache_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "provider_id",
            "model_id",
            "semantic_prefix_id",
            "authority_boundary_id",
            "target_boundary_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        try:
            kind = (
                self.cache_kind
                if isinstance(self.cache_kind, PrefixCacheKind)
                else PrefixCacheKind(str(self.cache_kind))
            )
        except ValueError as exc:
            raise PrefixContextError("cache_kind is not supported") from exc
        provider_cache_id = _text(
            self.provider_cache_id,
            "provider_cache_id",
            required=False,
        )
        if kind is PrefixCacheKind.DERIVED and provider_cache_id:
            raise PrefixContextError(
                "a derived prefix key cannot claim a provider cache identity"
            )
        if kind is not PrefixCacheKind.DERIVED and not provider_cache_id:
            raise PrefixContextError(
                "provider cache identities require provider_cache_id"
            )
        object.__setattr__(self, "cache_kind", kind)
        object.__setattr__(self, "provider_cache_id", provider_cache_id)

    @property
    def cache_identity_id(self) -> str:
        return self.content_id

    @property
    def cache_key(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_COMPILER_VERSION,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "cache_kind": self.cache_kind.value,
            "semantic_prefix_id": self.semantic_prefix_id,
            "authority_boundary_id": self.authority_boundary_id,
            "target_boundary_id": self.target_boundary_id,
            "provider_cache_id": self.provider_cache_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrefixCacheIdentity":
        _schema(payload, cls.SCHEMA, "prefix cache identity")
        _reject_unknown(
            payload,
            {
                "schema",
                "content_id",
                "cache_identity_id",
                "cache_key",
                "contract_version",
                "provider_id",
                "model_id",
                "cache_kind",
                "semantic_prefix_id",
                "authority_boundary_id",
                "target_boundary_id",
                "provider_cache_id",
            },
            "prefix cache identity",
        )
        result = cls(
            provider_id=payload.get("provider_id", ""),
            model_id=payload.get("model_id", ""),
            cache_kind=payload.get("cache_kind", ""),
            semantic_prefix_id=payload.get("semantic_prefix_id", ""),
            authority_boundary_id=payload.get("authority_boundary_id", ""),
            target_boundary_id=payload.get("target_boundary_id", ""),
            provider_cache_id=payload.get("provider_cache_id", ""),
        )
        _check_identity(payload, result.content_id, "prefix cache identity")
        for name in ("cache_identity_id", "cache_key"):
            claimed = payload.get(name)
            if claimed not in (None, "", result.content_id):
                raise PrefixContextError(
                    "prefix cache identity does not match its canonical key"
                )
        return result


@dataclass(frozen=True)
class PrefixStableContextCapsule(CanonicalContract):
    """One stage input split into two stable segments and an evidence delta."""

    SCHEMA: ClassVar[str] = PREFIX_STABLE_CONTEXT_CAPSULE_SCHEMA

    context_capsule: ContextCapsule
    provider_id: str
    model_id: str
    stable_prefix_tokens: int
    provider_input_tokens: int
    effective_input_limit: int

    def __post_init__(self) -> None:
        capsule = self.context_capsule
        if isinstance(capsule, Mapping):
            capsule = ContextCapsule.from_dict(capsule)
        if not isinstance(capsule, ContextCapsule):
            raise PrefixContextError(
                "context_capsule must be a ContextCapsule"
            )
        object.__setattr__(self, "context_capsule", capsule)
        object.__setattr__(
            self, "provider_id", _text(self.provider_id, "provider_id")
        )
        object.__setattr__(
            self, "model_id", _text(self.model_id, "model_id")
        )
        stable_tokens = _integer(
            self.stable_prefix_tokens,
            "stable_prefix_tokens",
            minimum=1,
        )
        input_tokens = _integer(
            self.provider_input_tokens,
            "provider_input_tokens",
            minimum=1,
        )
        effective_limit = _integer(
            self.effective_input_limit,
            "effective_input_limit",
            minimum=1,
        )
        if stable_tokens > input_tokens:
            raise PrefixContextError(
                "stable prefix tokens cannot exceed provider input tokens"
            )
        if effective_limit < capsule.budget.max_input_tokens:
            raise PrefixContextError(
                "prefix effective limit cannot weaken its base capsule limit"
            )
        if input_tokens > effective_limit:
            raise PrefixContextError(
                "prefix-stable input exceeds the effective provider budget"
            )
        object.__setattr__(self, "stable_prefix_tokens", stable_tokens)
        object.__setattr__(self, "provider_input_tokens", input_tokens)
        object.__setattr__(self, "effective_input_limit", effective_limit)

    @property
    def stable_policy_objective_prefix(self) -> Mapping[str, Any]:
        return _stable_policy_objective_prefix(self.context_capsule)

    @property
    def policy_objective_prefix(self) -> Mapping[str, Any]:
        return self.stable_policy_objective_prefix

    @property
    def stable_task_core(self) -> Mapping[str, Any]:
        return _stable_task_core(self.context_capsule)

    @property
    def volatile_evidence_delta(self) -> tuple[ContextReference, ...]:
        return self.context_capsule.evidence

    @property
    def evidence_delta(self) -> tuple[ContextReference, ...]:
        return self.volatile_evidence_delta

    @property
    def stable_policy_objective_bytes(self) -> bytes:
        return _prefix_segment_bytes(
            "stable_policy_objective_prefix",
            self.stable_policy_objective_prefix,
        )

    @property
    def stable_task_core_bytes(self) -> bytes:
        return _prefix_segment_bytes(
            "stable_task_core", self.stable_task_core
        )

    @property
    def stable_prefix_bytes(self) -> bytes:
        return _stable_prefix_bytes(self.context_capsule)

    @property
    def volatile_evidence_bytes(self) -> bytes:
        return _volatile_evidence_bytes(self.context_capsule)

    @property
    def provider_input_bytes(self) -> bytes:
        return self.stable_prefix_bytes + self.volatile_evidence_bytes

    @property
    def provider_input(self) -> str:
        return self.provider_input_bytes.decode("utf-8")

    @property
    def capsule_id(self) -> str:
        return self.content_id

    @property
    def input_tokens(self) -> int:
        return self.provider_input_tokens

    @property
    def semantic_prefix_id(self) -> str:
        return _canonical_digest(
            {
                "stable_policy_objective_prefix": (
                    self.stable_policy_objective_prefix
                ),
                "stable_task_core": self.stable_task_core,
            }
        )

    @property
    def prefix_dependency_id(self) -> str:
        return self.semantic_prefix_id

    @property
    def authority_boundary_id(self) -> str:
        capsule = self.context_capsule
        return _canonical_digest(
            {
                "repository_id": capsule.repository_id,
                "policy_id": capsule.policy_id,
                "policy_revision": capsule.policy_revision,
                "caller": capsule.caller,
                "authority": capsule.authority,
            }
        )

    @property
    def target_boundary_id(self) -> str:
        capsule = self.context_capsule
        return _canonical_digest(
            {
                "repository_id": capsule.repository_id,
                "tree_id": capsule.tree_id,
                "objective_id": capsule.objective_id,
                "stage": capsule.stage,
                "scope": capsule.scope,
            }
        )

    @property
    def evidence_digest(self) -> str:
        return _canonical_digest(
            tuple(item.to_record() for item in self.volatile_evidence_delta)
        )

    @property
    def required_field_names(self) -> tuple[str, ...]:
        return self.context_capsule.required_field_names

    def cache_identity(
        self,
        *,
        cache_kind: PrefixCacheKind | str = PrefixCacheKind.DERIVED,
        provider_cache_id: str = "",
    ) -> PrefixCacheIdentity:
        return PrefixCacheIdentity(
            provider_id=self.provider_id,
            model_id=self.model_id,
            cache_kind=cache_kind,
            semantic_prefix_id=self.semantic_prefix_id,
            authority_boundary_id=self.authority_boundary_id,
            target_boundary_id=self.target_boundary_id,
            provider_cache_id=provider_cache_id,
        )

    @property
    def prompt_cache_key(self) -> str:
        return self.cache_identity().cache_key

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_COMPILER_VERSION,
            "context_capsule": self.context_capsule.to_record(),
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "stable_policy_objective_prefix": (
                self.stable_policy_objective_prefix
            ),
            "stable_task_core": self.stable_task_core,
            "volatile_evidence_delta": tuple(
                item.to_record() for item in self.volatile_evidence_delta
            ),
            "semantic_prefix_id": self.semantic_prefix_id,
            "authority_boundary_id": self.authority_boundary_id,
            "target_boundary_id": self.target_boundary_id,
            "evidence_digest": self.evidence_digest,
            "stable_prefix_tokens": self.stable_prefix_tokens,
            "provider_input_tokens": self.provider_input_tokens,
            "effective_input_limit": self.effective_input_limit,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PrefixStableContextCapsule":
        _schema(payload, cls.SCHEMA, "prefix-stable context capsule")
        _reject_unknown(
            payload,
            {
                "schema",
                "content_id",
                "capsule_id",
                "contract_version",
                "context_capsule",
                "provider_id",
                "model_id",
                "stable_policy_objective_prefix",
                "stable_task_core",
                "volatile_evidence_delta",
                "semantic_prefix_id",
                "prefix_dependency_id",
                "authority_boundary_id",
                "target_boundary_id",
                "evidence_digest",
                "stable_prefix_tokens",
                "provider_input_tokens",
                "input_tokens",
                "effective_input_limit",
            },
            "prefix-stable context capsule",
        )
        raw_capsule = payload.get("context_capsule")
        if not isinstance(raw_capsule, (ContextCapsule, Mapping)):
            raise PrefixContextError("context_capsule is required")
        result = cls(
            context_capsule=(
                raw_capsule
                if isinstance(raw_capsule, ContextCapsule)
                else ContextCapsule.from_dict(raw_capsule)
            ),
            provider_id=payload.get("provider_id", ""),
            model_id=payload.get("model_id", ""),
            stable_prefix_tokens=payload.get("stable_prefix_tokens", 0),
            provider_input_tokens=payload.get(
                "provider_input_tokens", payload.get("input_tokens", 0)
            ),
            effective_input_limit=payload.get("effective_input_limit", 0),
        )
        projections = {
            "stable_policy_objective_prefix": (
                result.stable_policy_objective_prefix
            ),
            "stable_task_core": result.stable_task_core,
            "volatile_evidence_delta": tuple(
                item.to_record() for item in result.volatile_evidence_delta
            ),
            "semantic_prefix_id": result.semantic_prefix_id,
            "prefix_dependency_id": result.prefix_dependency_id,
            "authority_boundary_id": result.authority_boundary_id,
            "target_boundary_id": result.target_boundary_id,
            "evidence_digest": result.evidence_digest,
        }
        for name, actual in projections.items():
            claimed = payload.get(name)
            if (
                claimed not in (None, "")
                and canonical_context_json_bytes(claimed)
                != canonical_context_json_bytes(actual)
            ):
                raise PrefixContextError(
                    "prefix-stable projection does not match its context capsule"
                )
        _check_identity(
            payload, result.content_id, "prefix-stable context capsule"
        )
        claimed = payload.get("capsule_id")
        if claimed not in (None, "", result.content_id):
            raise PrefixContextError(
                "prefix-stable capsule identity does not match payload"
            )
        return result


@dataclass(frozen=True)
class PrefixReuseReceipt(CanonicalContract):
    """Auditable reuse decision for one prefix-stable stage input."""

    SCHEMA: ClassVar[str] = PREFIX_REUSE_RECEIPT_SCHEMA

    capsule_id: str
    context_capsule_id: str
    repository_id: str
    tree_id: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    caller: str
    stage: str
    cache_identity: PrefixCacheIdentity
    previous_capsule_id: str
    previous_semantic_prefix_id: str
    reuse_source: PrefixReuseSource | str
    cache_decision: PrefixCacheDecision | str
    eligible_stable_prefix_tokens: int
    reused_prefix_tokens: int
    provider_input_tokens: int
    provider_reused_tokens: int | None
    invalidated_dependencies: tuple[str, ...]
    evidence_reference_ids: tuple[str, ...]
    evidence_digest: str

    def __post_init__(self) -> None:
        for name in (
            "capsule_id",
            "context_capsule_id",
            "repository_id",
            "tree_id",
            "objective_id",
            "objective_revision",
            "policy_id",
            "policy_revision",
            "caller",
            "stage",
            "evidence_digest",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "previous_capsule_id",
            "previous_semantic_prefix_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, required=False),
            )
        identity = self.cache_identity
        if isinstance(identity, Mapping):
            identity = PrefixCacheIdentity.from_dict(identity)
        if not isinstance(identity, PrefixCacheIdentity):
            raise PrefixContextError(
                "cache_identity must be a PrefixCacheIdentity"
            )
        object.__setattr__(self, "cache_identity", identity)
        try:
            source = (
                self.reuse_source
                if isinstance(self.reuse_source, PrefixReuseSource)
                else PrefixReuseSource(str(self.reuse_source))
            )
            decision = (
                self.cache_decision
                if isinstance(self.cache_decision, PrefixCacheDecision)
                else PrefixCacheDecision(str(self.cache_decision))
            )
        except ValueError as exc:
            raise PrefixContextError(
                "prefix reuse source or cache decision is unsupported"
            ) from exc
        object.__setattr__(self, "reuse_source", source)
        object.__setattr__(self, "cache_decision", decision)
        for name in (
            "eligible_stable_prefix_tokens",
            "reused_prefix_tokens",
            "provider_input_tokens",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name)
            )
        if self.eligible_stable_prefix_tokens < 1:
            raise PrefixContextError(
                "eligible stable prefix tokens must be positive"
            )
        if self.reused_prefix_tokens > self.eligible_stable_prefix_tokens:
            raise PrefixContextError(
                "reused prefix tokens exceed the eligible stable prefix"
            )
        if self.eligible_stable_prefix_tokens > self.provider_input_tokens:
            raise PrefixContextError(
                "eligible prefix tokens exceed provider input tokens"
            )
        provider_reused = self.provider_reused_tokens
        if provider_reused is not None:
            provider_reused = _integer(
                provider_reused, "provider_reused_tokens"
            )
            if provider_reused > self.provider_input_tokens:
                raise PrefixContextError(
                    "provider reused tokens exceed provider input tokens"
                )
        object.__setattr__(
            self, "provider_reused_tokens", provider_reused
        )
        invalidated = _strings(
            self.invalidated_dependencies, "invalidated_dependencies"
        )
        evidence_ids = _strings(
            self.evidence_reference_ids, "evidence_reference_ids"
        )
        object.__setattr__(
            self, "invalidated_dependencies", invalidated
        )
        object.__setattr__(self, "evidence_reference_ids", evidence_ids)
        if bool(self.previous_capsule_id) != bool(
            self.previous_semantic_prefix_id
        ):
            raise PrefixContextError(
                "warm predecessor identities must be supplied together"
            )
        native_source = source in {
            PrefixReuseSource.PROVIDER_PROMPT_CACHE,
            PrefixReuseSource.PROVIDER_KV_CACHE,
        }
        if native_source != (provider_reused is not None):
            raise PrefixContextError(
                "provider-native reuse requires an actual reused-token count"
            )
        expected_kind = {
            PrefixReuseSource.PROVIDER_PROMPT_CACHE: (
                PrefixCacheKind.PROMPT_CACHE
            ),
            PrefixReuseSource.PROVIDER_KV_CACHE: PrefixCacheKind.KV_CACHE,
        }.get(source)
        if (
            expected_kind is not None
            and identity.cache_kind is not expected_kind
        ):
            raise PrefixContextError(
                "reuse source does not match the cache identity kind"
            )
        if decision is PrefixCacheDecision.COLD:
            if self.previous_capsule_id or self.reused_prefix_tokens:
                raise PrefixContextError(
                    "cold cache decisions cannot claim warm reuse"
                )
        elif not self.previous_capsule_id:
            raise PrefixContextError(
                "non-cold cache decisions require a warm predecessor"
            )
        if decision is PrefixCacheDecision.INVALIDATED:
            if not invalidated or self.reused_prefix_tokens:
                raise PrefixContextError(
                    "invalidated prefixes must name dependencies and reuse zero"
                )
        elif invalidated:
            raise PrefixContextError(
                "only invalidated cache decisions may name changed dependencies"
            )
        if decision is PrefixCacheDecision.HIT and not self.reused_prefix_tokens:
            raise PrefixContextError("cache hits must reuse prefix tokens")
        if decision is PrefixCacheDecision.MISS and self.reused_prefix_tokens:
            raise PrefixContextError("cache misses cannot reuse prefix tokens")
        if source is PrefixReuseSource.COLD and decision not in {
            PrefixCacheDecision.COLD,
            PrefixCacheDecision.INVALIDATED,
        }:
            raise PrefixContextError(
                "cold reuse source is limited to cold or invalidated decisions"
            )

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def reuse_bps(self) -> int:
        return (
            self.reused_prefix_tokens
            * 10_000
            // self.eligible_stable_prefix_tokens
        )

    @property
    def reuse_ratio_bps(self) -> int:
        return self.reuse_bps

    @property
    def qualifies(self) -> bool:
        return bool(
            self.cache_decision is PrefixCacheDecision.HIT
            and self.previous_capsule_id
            and not self.invalidated_dependencies
            and self.reuse_bps >= MIN_WARM_PREFIX_REUSE_BPS
            and self.evidence_digest
        )

    @property
    def evidence_claim_references(self) -> tuple[str, ...]:
        return (PREFIX_REUSE_REQUIREMENT_ID,) if self.qualifies else ()

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_COMPILER_VERSION,
            "capsule_id": self.capsule_id,
            "context_capsule_id": self.context_capsule_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "caller": self.caller,
            "stage": self.stage,
            "cache_identity": self.cache_identity.to_record(),
            "previous_capsule_id": self.previous_capsule_id,
            "previous_semantic_prefix_id": (
                self.previous_semantic_prefix_id
            ),
            "reuse_source": self.reuse_source.value,
            "cache_decision": self.cache_decision.value,
            "eligible_stable_prefix_tokens": (
                self.eligible_stable_prefix_tokens
            ),
            "reused_prefix_tokens": self.reused_prefix_tokens,
            "provider_input_tokens": self.provider_input_tokens,
            "provider_reused_tokens": self.provider_reused_tokens,
            "reuse_bps": self.reuse_bps,
            "invalidated_dependencies": self.invalidated_dependencies,
            "evidence_reference_ids": self.evidence_reference_ids,
            "evidence_digest": self.evidence_digest,
            "evidence_claim_references": self.evidence_claim_references,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrefixReuseReceipt":
        _schema(payload, cls.SCHEMA, "prefix reuse receipt")
        _reject_unknown(
            payload,
            {
                "schema",
                "content_id",
                "receipt_id",
                "contract_version",
                "capsule_id",
                "context_capsule_id",
                "repository_id",
                "tree_id",
                "objective_id",
                "objective_revision",
                "policy_id",
                "policy_revision",
                "caller",
                "stage",
                "cache_identity",
                "previous_capsule_id",
                "previous_semantic_prefix_id",
                "reuse_source",
                "cache_decision",
                "eligible_stable_prefix_tokens",
                "reused_prefix_tokens",
                "provider_input_tokens",
                "provider_reused_tokens",
                "reuse_bps",
                "invalidated_dependencies",
                "evidence_reference_ids",
                "evidence_digest",
                "evidence_claim_references",
            },
            "prefix reuse receipt",
        )
        identity = payload.get("cache_identity")
        if not isinstance(identity, (PrefixCacheIdentity, Mapping)):
            raise PrefixContextError("cache_identity is required")
        result = cls(
            capsule_id=payload.get("capsule_id", ""),
            context_capsule_id=payload.get("context_capsule_id", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            objective_id=payload.get("objective_id", ""),
            objective_revision=payload.get("objective_revision", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            caller=payload.get("caller", ""),
            stage=payload.get("stage", ""),
            cache_identity=(
                identity
                if isinstance(identity, PrefixCacheIdentity)
                else PrefixCacheIdentity.from_dict(identity)
            ),
            previous_capsule_id=payload.get("previous_capsule_id", ""),
            previous_semantic_prefix_id=payload.get(
                "previous_semantic_prefix_id", ""
            ),
            reuse_source=payload.get("reuse_source", ""),
            cache_decision=payload.get("cache_decision", ""),
            eligible_stable_prefix_tokens=payload.get(
                "eligible_stable_prefix_tokens", 0
            ),
            reused_prefix_tokens=payload.get("reused_prefix_tokens", 0),
            provider_input_tokens=payload.get("provider_input_tokens", 0),
            provider_reused_tokens=payload.get("provider_reused_tokens"),
            invalidated_dependencies=tuple(
                payload.get("invalidated_dependencies", ())
            ),
            evidence_reference_ids=tuple(
                payload.get("evidence_reference_ids", ())
            ),
            evidence_digest=payload.get("evidence_digest", ""),
        )
        claimed_bps = payload.get("reuse_bps")
        if claimed_bps not in (None, result.reuse_bps):
            raise PrefixContextError("prefix reuse ratio is forged")
        claims = payload.get("evidence_claim_references")
        if claims is not None and _strings(
            claims, "evidence_claim_references"
        ) != result.evidence_claim_references:
            raise PrefixContextError("prefix reuse evidence claim is forged")
        _check_identity(payload, result.content_id, "prefix reuse receipt")
        claimed_id = payload.get("receipt_id")
        if claimed_id not in (None, "", result.content_id):
            raise PrefixContextError(
                "prefix reuse receipt identity does not match payload"
            )
        return result


@dataclass(frozen=True)
class PrefixContextResult:
    """Compiler result joining the ordinary capsule and its prefix proof."""

    context_result: "ContextCompileResult"
    capsule: PrefixStableContextCapsule
    receipt: PrefixReuseReceipt
    verifier: Any = field(default=None, repr=False, compare=False)

    @property
    def prefix_capsule(self) -> PrefixStableContextCapsule:
        return self.capsule

    @property
    def base_capsule(self) -> ContextCapsule:
        return self.context_result.capsule

    @property
    def provider_input(self) -> str:
        return self.capsule.provider_input

    def __post_init__(self) -> None:
        if not isinstance(self.context_result, ContextCompileResult):
            raise PrefixContextError(
                "context_result must be a ContextCompileResult"
            )
        if not isinstance(self.capsule, PrefixStableContextCapsule):
            raise PrefixContextError(
                "capsule must be a PrefixStableContextCapsule"
            )
        if not isinstance(self.receipt, PrefixReuseReceipt):
            raise PrefixContextError(
                "receipt must be a PrefixReuseReceipt"
            )
        base = self.context_result.capsule
        if self.capsule.context_capsule != base:
            raise PrefixContextError(
                "prefix capsule is detached from its compiled context"
            )
        expected = {
            "capsule_id": self.capsule.capsule_id,
            "context_capsule_id": base.capsule_id,
            "repository_id": base.repository_id,
            "tree_id": base.tree_id,
            "objective_id": base.objective_id,
            "objective_revision": base.objective_revision,
            "policy_id": base.policy_id,
            "policy_revision": base.policy_revision,
            "caller": base.caller,
            "stage": base.stage,
            "eligible_stable_prefix_tokens": (
                self.capsule.stable_prefix_tokens
            ),
            "provider_input_tokens": self.capsule.provider_input_tokens,
            "evidence_digest": self.capsule.evidence_digest,
        }
        if any(
            getattr(self.receipt, name) != value
            for name, value in expected.items()
        ):
            raise PrefixContextError(
                "prefix reuse receipt is detached from its capsule"
            )
        if self.receipt.cache_identity != self.capsule.cache_identity(
            cache_kind=self.receipt.cache_identity.cache_kind,
            provider_cache_id=(
                self.receipt.cache_identity.provider_cache_id
            ),
        ):
            raise PrefixContextError(
                "cache identity is not bound to the exact stable prefix"
            )
        reference_ids = tuple(
            sorted(item.reference_id for item in base.evidence)
        )
        if self.receipt.evidence_reference_ids != reference_ids:
            raise PrefixContextError(
                "prefix receipt does not bind current evidence"
            )
        if self.verifier is not None:
            if not isinstance(self.verifier, ContextCompiler):
                raise PrefixContextError(
                    "prefix result verifier must be its ContextCompiler"
                )
            self.verifier.verify_prefix_result(self)


def render_prefix_context(capsule: PrefixStableContextCapsule) -> str:
    """Render stable policy/task segments before the volatile evidence delta."""

    if not isinstance(capsule, PrefixStableContextCapsule):
        raise PrefixContextError(
            "capsule must be a PrefixStableContextCapsule"
        )
    return capsule.provider_input


@dataclass(frozen=True)
class RequiredContextBudgetEvidence(CanonicalContract):
    """Qualifying witness that required context survived the effective budget."""

    SCHEMA: ClassVar[str] = REQUIRED_CONTEXT_BUDGET_EVIDENCE_SCHEMA

    repository_id: str
    tree_id: str
    policy_id: str
    policy_revision: str
    capsule_id: str
    budget_resolution: ContextBudgetResolution
    effective_input_limit: int
    input_tokens: int
    required_fields: tuple[str, ...]
    required_reference_ids: tuple[str, ...]
    selected_reference_ids: tuple[str, ...]
    artifact_digest: str
    requirement_id: str = REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID
    result: str = "passed"

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "policy_id",
            "policy_revision",
            "capsule_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.requirement_id != REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID:
            raise ContextCompilationError("unexpected required-context requirement ID")
        if self.result != "passed":
            raise ContextCompilationError("required-context evidence must pass")
        object.__setattr__(
            self,
            "effective_input_limit",
            _integer(
                self.effective_input_limit,
                "effective_input_limit",
                minimum=1,
            ),
        )
        resolution = self.budget_resolution
        if isinstance(resolution, Mapping):
            resolution = ContextBudgetResolution.from_dict(resolution)
        if not isinstance(resolution, ContextBudgetResolution):
            raise ContextCompilationError(
                "budget_resolution must be a ContextBudgetResolution"
            )
        if resolution.effective_input_limit != self.effective_input_limit:
            raise ContextCompilationError(
                "budget resolution does not derive the effective input limit"
            )
        object.__setattr__(self, "budget_resolution", resolution)
        object.__setattr__(
            self, "input_tokens", _integer(self.input_tokens, "input_tokens")
        )
        if self.input_tokens > self.effective_input_limit:
            raise ContextCompilationError("evidence exceeds effective input limit")
        fields = _strings(self.required_fields, "required_fields")
        if fields != ("acceptance", "authority", "goal", "scope"):
            raise ContextCompilationError(
                "required fields must bind goal, authority, scope, and acceptance"
            )
        object.__setattr__(self, "required_fields", fields)
        required = _strings(
            self.required_reference_ids, "required_reference_ids"
        )
        selected = _strings(
            self.selected_reference_ids, "selected_reference_ids"
        )
        if not set(required).issubset(selected):
            raise ContextCompilationError(
                "required references must be selected by qualifying evidence"
            )
        object.__setattr__(self, "required_reference_ids", required)
        object.__setattr__(self, "selected_reference_ids", selected)
        object.__setattr__(
            self, "artifact_digest", _digest(self.artifact_digest, "artifact_digest")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_COMPILER_VERSION,
            "requirement_id": self.requirement_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "capsule_id": self.capsule_id,
            "budget_resolution": self.budget_resolution,
            "effective_input_limit": self.effective_input_limit,
            "input_tokens": self.input_tokens,
            "required_fields": self.required_fields,
            "required_reference_ids": self.required_reference_ids,
            "selected_reference_ids": self.selected_reference_ids,
            "artifact_digest": self.artifact_digest,
            "result": self.result,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "RequiredContextBudgetEvidence":
        _schema(payload, cls.SCHEMA, "required-context evidence")
        _reject_unknown(
            payload,
            {
                "schema",
                "content_id",
                "contract_version",
                "requirement_id",
                "repository_id",
                "tree_id",
                "policy_id",
                "policy_revision",
                "capsule_id",
                "budget_resolution",
                "effective_input_limit",
                "input_tokens",
                "required_fields",
                "required_reference_ids",
                "selected_reference_ids",
                "artifact_digest",
                "result",
            },
            "required-context evidence",
        )
        result = cls(
            requirement_id=payload.get("requirement_id", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            capsule_id=payload.get("capsule_id", ""),
            budget_resolution=payload.get("budget_resolution"),
            effective_input_limit=payload.get("effective_input_limit", 0),
            input_tokens=payload.get("input_tokens", 0),
            required_fields=tuple(payload.get("required_fields", ())),
            required_reference_ids=tuple(
                payload.get("required_reference_ids", ())
            ),
            selected_reference_ids=tuple(
                payload.get("selected_reference_ids", ())
            ),
            artifact_digest=payload.get("artifact_digest", ""),
            result=payload.get("result", ""),
        )
        _check_identity(payload, result.content_id, "required-context evidence")
        return result


@dataclass(frozen=True)
class EvidenceValuePairedFixture(CanonicalContract):
    """One denominator-stable baseline/VOI comparison."""

    SCHEMA: ClassVar[str] = EVIDENCE_VALUE_FIXTURE_SCHEMA

    fixture_id: str
    accepted_criterion_ids: tuple[str, ...]
    baseline_input_tokens: int
    selected_input_tokens: int
    baseline_retry_input_tokens: int
    selected_retry_input_tokens: int
    baseline_required_coverage_ids: tuple[str, ...]
    selected_required_coverage_ids: tuple[str, ...]
    baseline_safety_passed: bool = True
    selected_safety_passed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fixture_id", _text(self.fixture_id, "fixture_id")
        )
        criteria = _strings(
            self.accepted_criterion_ids,
            "accepted_criterion_ids",
        )
        if not criteria:
            raise ContextCompilationError(
                "paired fixture requires accepted criterion IDs"
            )
        object.__setattr__(self, "accepted_criterion_ids", criteria)
        for name, minimum in (
            ("baseline_input_tokens", 1),
            ("selected_input_tokens", 0),
            ("baseline_retry_input_tokens", 1),
            ("selected_retry_input_tokens", 0),
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, minimum=minimum),
            )
        baseline_coverage = _strings(
            self.baseline_required_coverage_ids,
            "baseline_required_coverage_ids",
        )
        selected_coverage = _strings(
            self.selected_required_coverage_ids,
            "selected_required_coverage_ids",
        )
        if baseline_coverage != selected_coverage:
            raise ContextCompilationError(
                "paired evidence selection changes required coverage"
            )
        object.__setattr__(
            self, "baseline_required_coverage_ids", baseline_coverage
        )
        object.__setattr__(
            self, "selected_required_coverage_ids", selected_coverage
        )
        if (
            not isinstance(self.baseline_safety_passed, bool)
            or not isinstance(self.selected_safety_passed, bool)
            or not self.baseline_safety_passed
            or not self.selected_safety_passed
        ):
            raise ContextCompilationError(
                "paired evidence selection must preserve passing safety"
            )

    @property
    def accepted_criterion_count(self) -> int:
        return len(self.accepted_criterion_ids)

    @property
    def baseline_tokens_per_criterion(self) -> Fraction:
        return Fraction(
            self.baseline_input_tokens,
            self.accepted_criterion_count,
        )

    @property
    def selected_tokens_per_criterion(self) -> Fraction:
        return Fraction(
            self.selected_input_tokens,
            self.accepted_criterion_count,
        )

    @property
    def baseline_retry_tokens_per_criterion(self) -> Fraction:
        return Fraction(
            self.baseline_retry_input_tokens,
            self.accepted_criterion_count,
        )

    @property
    def selected_retry_tokens_per_criterion(self) -> Fraction:
        return Fraction(
            self.selected_retry_input_tokens,
            self.accepted_criterion_count,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_COMPILER_VERSION,
            "fixture_id": self.fixture_id,
            "accepted_criterion_ids": self.accepted_criterion_ids,
            "baseline_input_tokens": self.baseline_input_tokens,
            "selected_input_tokens": self.selected_input_tokens,
            "baseline_retry_input_tokens": self.baseline_retry_input_tokens,
            "selected_retry_input_tokens": self.selected_retry_input_tokens,
            "baseline_required_coverage_ids": (
                self.baseline_required_coverage_ids
            ),
            "selected_required_coverage_ids": (
                self.selected_required_coverage_ids
            ),
            "baseline_safety_passed": self.baseline_safety_passed,
            "selected_safety_passed": self.selected_safety_passed,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EvidenceValuePairedFixture":
        _schema(payload, cls.SCHEMA, "evidence-value paired fixture")
        allowed = {
            "schema",
            "content_id",
            "contract_version",
            "fixture_id",
            "accepted_criterion_ids",
            "baseline_input_tokens",
            "selected_input_tokens",
            "baseline_retry_input_tokens",
            "selected_retry_input_tokens",
            "baseline_required_coverage_ids",
            "selected_required_coverage_ids",
            "baseline_safety_passed",
            "selected_safety_passed",
        }
        _reject_unknown(payload, allowed, "evidence-value paired fixture")
        result = cls(
            fixture_id=payload.get("fixture_id", ""),
            accepted_criterion_ids=tuple(
                payload.get("accepted_criterion_ids", ())
            ),
            baseline_input_tokens=payload.get("baseline_input_tokens", 0),
            selected_input_tokens=payload.get("selected_input_tokens", 0),
            baseline_retry_input_tokens=payload.get(
                "baseline_retry_input_tokens", 0
            ),
            selected_retry_input_tokens=payload.get(
                "selected_retry_input_tokens", 0
            ),
            baseline_required_coverage_ids=tuple(
                payload.get("baseline_required_coverage_ids", ())
            ),
            selected_required_coverage_ids=tuple(
                payload.get("selected_required_coverage_ids", ())
            ),
            baseline_safety_passed=payload.get(
                "baseline_safety_passed", False
            ),
            selected_safety_passed=payload.get(
                "selected_safety_passed", False
            ),
        )
        _check_identity(payload, result.content_id, "evidence-value paired fixture")
        return result


def _median_fraction(values: Iterable[Fraction]) -> Fraction:
    ordered = sorted(values)
    if not ordered:
        raise ContextCompilationError(
            "value-of-information evidence requires paired fixtures"
        )
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _reduction_bps(baseline: Fraction, selected: Fraction) -> int:
    if baseline <= 0:
        raise ContextCompilationError(
            "paired baseline token denominator must be positive"
        )
    reduction = (baseline - selected) * MAX_VALUE_BPS / baseline
    return reduction.numerator // reduction.denominator


@dataclass(frozen=True)
class ValueOfInformationEvidence(CanonicalContract):
    """Qualifying population-complete evidence for the 40/60 percent gates."""

    SCHEMA: ClassVar[str] = VALUE_OF_INFORMATION_EVIDENCE_SCHEMA

    repository_id: str
    tree_id: str
    policy_id: str
    policy_revision: str
    provider_id: str
    model_id: str
    fixtures: tuple[EvidenceValuePairedFixture, ...]
    requirement_id: str = VALUE_OF_INFORMATION_REQUIREMENT_ID
    result: str = "passed"

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "policy_id",
            "policy_revision",
            "provider_id",
            "model_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        if self.requirement_id != VALUE_OF_INFORMATION_REQUIREMENT_ID:
            raise ContextCompilationError(
                "unexpected value-of-information requirement ID"
            )
        if self.result != "passed":
            raise ContextCompilationError(
                "value-of-information evidence must pass"
            )
        fixtures: list[EvidenceValuePairedFixture] = []
        for raw in self.fixtures:
            fixtures.append(
                raw
                if isinstance(raw, EvidenceValuePairedFixture)
                else EvidenceValuePairedFixture.from_dict(raw)
            )
        fixtures.sort(key=lambda item: item.fixture_id)
        if not fixtures or len(fixtures) > MAX_DECISIONS:
            raise ContextCompilationError(
                "value-of-information fixtures must be non-empty and bounded"
            )
        if len({item.fixture_id for item in fixtures}) != len(fixtures):
            raise ContextCompilationError(
                "value-of-information fixture IDs must be unique"
            )
        object.__setattr__(self, "fixtures", tuple(fixtures))
        if self.input_token_reduction_bps < MIN_INPUT_TOKEN_REDUCTION_BPS:
            raise ContextCompilationError(
                "median input tokens per accepted criterion improve by less "
                "than 40 percent"
            )
        if (
            self.retry_input_token_reduction_bps
            < MIN_RETRY_INPUT_TOKEN_REDUCTION_BPS
        ):
            raise ContextCompilationError(
                "median retry-input tokens improve by less than 60 percent"
            )

    @property
    def input_token_reduction_bps(self) -> int:
        return _reduction_bps(
            _median_fraction(
                item.baseline_tokens_per_criterion for item in self.fixtures
            ),
            _median_fraction(
                item.selected_tokens_per_criterion for item in self.fixtures
            ),
        )

    @property
    def retry_input_token_reduction_bps(self) -> int:
        return _reduction_bps(
            _median_fraction(
                item.baseline_retry_tokens_per_criterion
                for item in self.fixtures
            ),
            _median_fraction(
                item.selected_retry_tokens_per_criterion
                for item in self.fixtures
            ),
        )

    @property
    def accepted_criterion_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    criterion
                    for item in self.fixtures
                    for criterion in item.accepted_criterion_ids
                }
            )
        )

    @property
    def required_coverage_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    coverage
                    for item in self.fixtures
                    for coverage in item.selected_required_coverage_ids
                }
            )
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_COMPILER_VERSION,
            "requirement_id": self.requirement_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "fixtures": tuple(item.to_record() for item in self.fixtures),
            "accepted_criterion_ids": self.accepted_criterion_ids,
            "required_coverage_ids": self.required_coverage_ids,
            "input_token_reduction_bps": self.input_token_reduction_bps,
            "retry_input_token_reduction_bps": (
                self.retry_input_token_reduction_bps
            ),
            "required_coverage_preserved": True,
            "safety_preserved": True,
            "result": self.result,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ValueOfInformationEvidence":
        _schema(payload, cls.SCHEMA, "value-of-information evidence")
        allowed = {
            "schema",
            "content_id",
            "contract_version",
            "requirement_id",
            "repository_id",
            "tree_id",
            "policy_id",
            "policy_revision",
            "provider_id",
            "model_id",
            "fixtures",
            "accepted_criterion_ids",
            "required_coverage_ids",
            "input_token_reduction_bps",
            "retry_input_token_reduction_bps",
            "required_coverage_preserved",
            "safety_preserved",
            "result",
        }
        _reject_unknown(payload, allowed, "value-of-information evidence")
        result = cls(
            requirement_id=payload.get("requirement_id", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            provider_id=payload.get("provider_id", ""),
            model_id=payload.get("model_id", ""),
            fixtures=tuple(
                EvidenceValuePairedFixture.from_dict(item)
                for item in payload.get("fixtures", ())
            ),
            result=payload.get("result", ""),
        )
        expected = result._payload()
        for name in ("accepted_criterion_ids", "required_coverage_ids"):
            claimed = payload.get(name)
            canonical = expected[name]
            if claimed is not None and _strings(claimed, name) != canonical:
                raise ContextCompilationError(
                    f"value-of-information {name} is forged"
                )
        for name in (
            "input_token_reduction_bps",
            "retry_input_token_reduction_bps",
            "required_coverage_preserved",
            "safety_preserved",
        ):
            claimed = payload.get(name)
            canonical = expected[name]
            if claimed not in (None, canonical):
                raise ContextCompilationError(
                    f"value-of-information {name} is forged"
                )
        _check_identity(payload, result.content_id, "value-of-information evidence")
        return result


def evaluate_evidence_value_fixtures(
    *,
    repository_id: str,
    tree_id: str,
    policy_id: str,
    policy_revision: str,
    provider_id: str,
    model_id: str,
    fixtures: Iterable[EvidenceValuePairedFixture | Mapping[str, Any]],
) -> ValueOfInformationEvidence:
    """Evaluate and bind one complete paired evidence-selection population."""

    return ValueOfInformationEvidence(
        repository_id=repository_id,
        tree_id=tree_id,
        policy_id=policy_id,
        policy_revision=policy_revision,
        provider_id=provider_id,
        model_id=model_id,
        fixtures=tuple(
            item
            if isinstance(item, EvidenceValuePairedFixture)
            else EvidenceValuePairedFixture.from_dict(item)
            for item in fixtures
        ),
    )


@dataclass(frozen=True)
class ContextCompilationReceipt(CanonicalContract):
    """Bounded audit receipt for one base-context compilation."""

    SCHEMA: ClassVar[str] = CONTEXT_COMPILATION_RECEIPT_SCHEMA

    repository_id: str
    tree_id: str
    objective_id: str
    policy_id: str
    policy_revision: str
    stage: str
    capsule_id: str
    budget_resolution: ContextBudgetResolution
    effective_input_limit: int
    input_tokens: int
    estimator_name: str
    estimator_error_bps: int
    decisions: tuple[EvidenceSelectionDecision, ...] = ()
    evidence: RequiredContextBudgetEvidence | None = None

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "objective_id",
            "policy_id",
            "policy_revision",
            "stage",
            "capsule_id",
            "estimator_name",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("effective_input_limit", "input_tokens", "estimator_error_bps"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.effective_input_limit < 1:
            raise ContextCompilationError("effective_input_limit must be positive")
        if self.input_tokens > self.effective_input_limit:
            raise ContextCompilationError("receipt input exceeds effective limit")
        if self.estimator_error_bps > MAX_ERROR_BPS:
            raise ContextCompilationError("estimator error exceeds its bound")
        resolution = self.budget_resolution
        if isinstance(resolution, Mapping):
            resolution = ContextBudgetResolution.from_dict(resolution)
        if not isinstance(resolution, ContextBudgetResolution):
            raise ContextCompilationError(
                "budget_resolution must be a ContextBudgetResolution"
            )
        if resolution.effective_input_limit != self.effective_input_limit:
            raise ContextCompilationError(
                "receipt budget resolution does not derive its effective limit"
            )
        object.__setattr__(self, "budget_resolution", resolution)
        decisions: list[EvidenceSelectionDecision] = []
        for raw in self.decisions:
            decisions.append(
                raw
                if isinstance(raw, EvidenceSelectionDecision)
                else EvidenceSelectionDecision.from_dict(raw)
            )
        decisions.sort(key=lambda item: item.reference_id)
        if len(decisions) > MAX_DECISIONS or len(
            {item.reference_id for item in decisions}
        ) != len(decisions):
            raise ContextCompilationError(
                "selection decisions must be bounded and unique"
            )
        object.__setattr__(self, "decisions", tuple(decisions))
        if self.evidence is not None:
            evidence = (
                self.evidence
                if isinstance(self.evidence, RequiredContextBudgetEvidence)
                else RequiredContextBudgetEvidence.from_dict(self.evidence)
            )
            if (
                evidence.repository_id != self.repository_id
                or evidence.tree_id != self.tree_id
                or evidence.policy_id != self.policy_id
                or evidence.policy_revision != self.policy_revision
                or evidence.capsule_id != self.capsule_id
                or evidence.budget_resolution != self.budget_resolution
                or evidence.input_tokens != self.input_tokens
                or evidence.effective_input_limit != self.effective_input_limit
            ):
                raise ContextCompilationError(
                    "required-context evidence is not bound to its receipt"
                )
            object.__setattr__(self, "evidence", evidence)

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def evidence_claim_references(self) -> tuple[str, ...]:
        return (
            (REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID,)
            if self.evidence is not None
            else ()
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_COMPILER_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "stage": self.stage,
            "capsule_id": self.capsule_id,
            "budget_resolution": self.budget_resolution,
            "effective_input_limit": self.effective_input_limit,
            "input_tokens": self.input_tokens,
            "estimator_name": self.estimator_name,
            "estimator_error_bps": self.estimator_error_bps,
            "decisions": tuple(item.to_dict() for item in self.decisions),
            "evidence": self.evidence.to_record() if self.evidence else None,
            "evidence_claim_references": self.evidence_claim_references,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextCompilationReceipt":
        _schema(payload, cls.SCHEMA, "context compilation receipt")
        _reject_unknown(
            payload,
            {
                "schema",
                "content_id",
                "receipt_id",
                "contract_version",
                "repository_id",
                "tree_id",
                "objective_id",
                "policy_id",
                "policy_revision",
                "stage",
                "capsule_id",
                "budget_resolution",
                "effective_input_limit",
                "input_tokens",
                "estimator_name",
                "estimator_error_bps",
                "decisions",
                "evidence",
                "evidence_claim_references",
            },
            "context compilation receipt",
        )
        evidence = payload.get("evidence")
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            objective_id=payload.get("objective_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            stage=payload.get("stage", ""),
            capsule_id=payload.get("capsule_id", ""),
            budget_resolution=payload.get("budget_resolution"),
            effective_input_limit=payload.get("effective_input_limit", 0),
            input_tokens=payload.get("input_tokens", 0),
            estimator_name=payload.get("estimator_name", ""),
            estimator_error_bps=payload.get("estimator_error_bps", 0),
            decisions=tuple(
                EvidenceSelectionDecision.from_dict(item)
                for item in payload.get("decisions", ())
            ),
            evidence=(
                RequiredContextBudgetEvidence.from_dict(evidence)
                if isinstance(evidence, Mapping)
                else None
            ),
        )
        claims = payload.get("evidence_claim_references")
        if claims is not None and _strings(
            claims, "evidence_claim_references"
        ) != result.evidence_claim_references:
            raise ContextCompilationError("context evidence claim is forged")
        _check_identity(payload, result.content_id, "context compilation receipt")
        return result


@dataclass(frozen=True)
class DeltaRetryContextEvidence(CanonicalContract):
    """Qualifying witness for a smaller lossless parent-bound retry delta."""

    SCHEMA: ClassVar[str] = DELTA_RETRY_CONTEXT_EVIDENCE_SCHEMA

    repository_id: str
    tree_id: str
    policy_id: str
    policy_revision: str
    parent_capsule_id: str
    delta_capsule_id: str
    reconstructed_capsule_id: str
    full_replay_tokens: int
    delta_tokens: int
    required_coverage_ids: tuple[str, ...]
    reconstructed_coverage_ids: tuple[str, ...]
    changed_reference_ids: tuple[str, ...]
    requested_reference_ids: tuple[str, ...]
    retained_reference_ids: tuple[str, ...]
    required_fields: tuple[str, ...]
    artifact_digest: str
    requirement_id: str = DELTA_RETRY_EVIDENCE_ID
    result: str = "passed"

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "policy_id",
            "policy_revision",
            "parent_capsule_id",
            "delta_capsule_id",
            "reconstructed_capsule_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.requirement_id != DELTA_RETRY_EVIDENCE_ID:
            raise ContextDeltaError("unexpected delta-retry requirement ID")
        if self.result != "passed":
            raise ContextDeltaError("delta-retry evidence must pass")
        for name in ("full_replay_tokens", "delta_tokens"):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=1)
            )
        if self.delta_tokens >= self.full_replay_tokens:
            raise ContextDeltaError("qualifying delta must use fewer tokens")
        required = _strings(
            self.required_coverage_ids, "required_coverage_ids"
        )
        reconstructed = _strings(
            self.reconstructed_coverage_ids, "reconstructed_coverage_ids"
        )
        if not set(required).issubset(reconstructed):
            raise ContextDeltaError("qualifying delta loses required coverage")
        object.__setattr__(self, "required_coverage_ids", required)
        object.__setattr__(self, "reconstructed_coverage_ids", reconstructed)
        changed = _strings(
            self.changed_reference_ids, "changed_reference_ids"
        )
        object.__setattr__(self, "changed_reference_ids", changed)
        requested = _strings(
            self.requested_reference_ids, "requested_reference_ids"
        )
        if set(changed).intersection(requested):
            raise ContextDeltaError(
                "changed and requested-only references must be disjoint"
            )
        if not changed and not requested:
            raise ContextDeltaError(
                "qualifying delta must carry changed or requested evidence"
            )
        object.__setattr__(self, "requested_reference_ids", requested)
        retained = _strings(
            self.retained_reference_ids, "retained_reference_ids"
        )
        if set(changed).union(requested).intersection(retained):
            raise ContextDeltaError(
                "transmitted and retained delta references must be disjoint"
            )
        object.__setattr__(self, "retained_reference_ids", retained)
        fields = _strings(self.required_fields, "required_fields")
        if fields != ("acceptance", "authority", "goal", "scope"):
            raise ContextDeltaError(
                "delta evidence must preserve every invariant context field"
            )
        object.__setattr__(self, "required_fields", fields)
        object.__setattr__(
            self, "artifact_digest", _digest(self.artifact_digest, "artifact_digest")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_COMPILER_VERSION,
            "requirement_id": self.requirement_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "parent_capsule_id": self.parent_capsule_id,
            "delta_capsule_id": self.delta_capsule_id,
            "reconstructed_capsule_id": self.reconstructed_capsule_id,
            "full_replay_tokens": self.full_replay_tokens,
            "delta_tokens": self.delta_tokens,
            "required_coverage_ids": self.required_coverage_ids,
            "reconstructed_coverage_ids": self.reconstructed_coverage_ids,
            "changed_reference_ids": self.changed_reference_ids,
            "requested_reference_ids": self.requested_reference_ids,
            "retained_reference_ids": self.retained_reference_ids,
            "required_fields": self.required_fields,
            "artifact_digest": self.artifact_digest,
            "result": self.result,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DeltaRetryContextEvidence":
        _schema(payload, cls.SCHEMA, "delta-retry evidence")
        _reject_unknown(
            payload,
            {
                "schema",
                "content_id",
                "contract_version",
                "requirement_id",
                "repository_id",
                "tree_id",
                "policy_id",
                "policy_revision",
                "parent_capsule_id",
                "delta_capsule_id",
                "reconstructed_capsule_id",
                "full_replay_tokens",
                "delta_tokens",
                "required_coverage_ids",
                "reconstructed_coverage_ids",
                "changed_reference_ids",
                "requested_reference_ids",
                "retained_reference_ids",
                "required_fields",
                "artifact_digest",
                "result",
            },
            "delta-retry evidence",
        )
        result = cls(
            requirement_id=payload.get("requirement_id", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            parent_capsule_id=payload.get("parent_capsule_id", ""),
            delta_capsule_id=payload.get("delta_capsule_id", ""),
            reconstructed_capsule_id=payload.get(
                "reconstructed_capsule_id", ""
            ),
            full_replay_tokens=payload.get("full_replay_tokens", 0),
            delta_tokens=payload.get("delta_tokens", 0),
            required_coverage_ids=tuple(
                payload.get("required_coverage_ids", ())
            ),
            reconstructed_coverage_ids=tuple(
                payload.get("reconstructed_coverage_ids", ())
            ),
            changed_reference_ids=tuple(
                payload.get("changed_reference_ids", ())
            ),
            requested_reference_ids=tuple(
                payload.get("requested_reference_ids", ())
            ),
            retained_reference_ids=tuple(
                payload.get("retained_reference_ids", ())
            ),
            required_fields=tuple(payload.get("required_fields", ())),
            artifact_digest=payload.get("artifact_digest", ""),
            result=payload.get("result", ""),
        )
        _check_identity(payload, result.content_id, "delta-retry evidence")
        return result


@dataclass(frozen=True)
class ContextDeltaReceipt(CanonicalContract):
    """Content-addressed audit receipt for one retry delta."""

    SCHEMA: ClassVar[str] = CONTEXT_DELTA_RECEIPT_SCHEMA

    repository_id: str
    tree_id: str
    objective_id: str
    policy_id: str
    policy_revision: str
    parent_capsule_id: str
    delta_capsule_id: str
    reconstructed_capsule_id: str
    full_replay_tokens: int
    delta_tokens: int
    decisions: tuple[EvidenceSelectionDecision, ...]
    evidence: DeltaRetryContextEvidence | None = None

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "objective_id",
            "policy_id",
            "policy_revision",
            "parent_capsule_id",
            "delta_capsule_id",
            "reconstructed_capsule_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("full_replay_tokens", "delta_tokens"):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=1)
            )
        decisions = tuple(
            sorted(
                (
                    item
                    if isinstance(item, EvidenceSelectionDecision)
                    else EvidenceSelectionDecision.from_dict(item)
                    for item in self.decisions
                ),
                key=lambda item: item.reference_id,
            )
        )
        if len(decisions) > MAX_DECISIONS or len(
            {item.reference_id for item in decisions}
        ) != len(decisions):
            raise ContextDeltaError("delta decisions must be bounded and unique")
        object.__setattr__(self, "decisions", decisions)
        if self.evidence is not None:
            evidence = (
                self.evidence
                if isinstance(self.evidence, DeltaRetryContextEvidence)
                else DeltaRetryContextEvidence.from_dict(self.evidence)
            )
            if any(
                (
                    evidence.repository_id != self.repository_id,
                    evidence.tree_id != self.tree_id,
                    evidence.policy_id != self.policy_id,
                    evidence.policy_revision != self.policy_revision,
                    evidence.parent_capsule_id != self.parent_capsule_id,
                    evidence.delta_capsule_id != self.delta_capsule_id,
                    evidence.reconstructed_capsule_id
                    != self.reconstructed_capsule_id,
                    evidence.full_replay_tokens != self.full_replay_tokens,
                    evidence.delta_tokens != self.delta_tokens,
                )
            ):
                raise ContextDeltaError("delta evidence is not bound to its receipt")
            included = {
                item.reference_id
                for item in decisions
                if item.included
            }
            witnessed = set(evidence.changed_reference_ids).union(
                evidence.requested_reference_ids
            )
            if included != witnessed:
                raise ContextDeltaError(
                    "delta decisions do not match witnessed transmitted references"
                )
            object.__setattr__(self, "evidence", evidence)

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def evidence_claim_references(self) -> tuple[str, ...]:
        return (DELTA_RETRY_EVIDENCE_ID,) if self.evidence else ()

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_COMPILER_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "parent_capsule_id": self.parent_capsule_id,
            "delta_capsule_id": self.delta_capsule_id,
            "reconstructed_capsule_id": self.reconstructed_capsule_id,
            "full_replay_tokens": self.full_replay_tokens,
            "delta_tokens": self.delta_tokens,
            "decisions": tuple(item.to_dict() for item in self.decisions),
            "evidence": self.evidence.to_record() if self.evidence else None,
            "evidence_claim_references": self.evidence_claim_references,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextDeltaReceipt":
        _schema(payload, cls.SCHEMA, "context delta receipt")
        _reject_unknown(
            payload,
            {
                "schema",
                "content_id",
                "receipt_id",
                "contract_version",
                "repository_id",
                "tree_id",
                "objective_id",
                "policy_id",
                "policy_revision",
                "parent_capsule_id",
                "delta_capsule_id",
                "reconstructed_capsule_id",
                "full_replay_tokens",
                "delta_tokens",
                "decisions",
                "evidence",
                "evidence_claim_references",
            },
            "context delta receipt",
        )
        evidence = payload.get("evidence")
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            objective_id=payload.get("objective_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            parent_capsule_id=payload.get("parent_capsule_id", ""),
            delta_capsule_id=payload.get("delta_capsule_id", ""),
            reconstructed_capsule_id=payload.get(
                "reconstructed_capsule_id", ""
            ),
            full_replay_tokens=payload.get("full_replay_tokens", 0),
            delta_tokens=payload.get("delta_tokens", 0),
            decisions=tuple(
                EvidenceSelectionDecision.from_dict(item)
                for item in payload.get("decisions", ())
            ),
            evidence=(
                DeltaRetryContextEvidence.from_dict(evidence)
                if isinstance(evidence, Mapping)
                else None
            ),
        )
        claims = payload.get("evidence_claim_references")
        if claims is not None and _strings(
            claims, "evidence_claim_references"
        ) != result.evidence_claim_references:
            raise ContextDeltaError("delta evidence claim is forged")
        _check_identity(payload, result.content_id, "context delta receipt")
        return result


@dataclass(frozen=True)
class ContextCompileResult:
    capsule: ContextCapsule
    receipt: ContextCompilationReceipt
    decisions: tuple[EvidenceSelectionDecision, ...]
    verifier: Any = field(default=None, repr=False, compare=False)

    @property
    def required_context_preserved(self) -> bool:
        """Whether the receipt proves the complete invariant context survived."""

        evidence = self.receipt.evidence
        return bool(
            evidence is not None
            and evidence.required_fields
            == tuple(sorted(self.capsule.required_field_names))
            and not set(self.capsule.required_field_names).intersection(
                self.capsule.omitted_reference_ids
            )
        )

    def __post_init__(self) -> None:
        if not isinstance(self.capsule, ContextCapsule):
            raise ContextCompilationError("capsule must be a ContextCapsule")
        if not isinstance(self.receipt, ContextCompilationReceipt):
            raise ContextCompilationError(
                "receipt must be a ContextCompilationReceipt"
            )
        receipt_bindings = {
            "repository_id": self.capsule.repository_id,
            "tree_id": self.capsule.tree_id,
            "objective_id": self.capsule.objective_id,
            "policy_id": self.capsule.policy_id,
            "policy_revision": self.capsule.policy_revision,
            "stage": self.capsule.stage,
            "capsule_id": self.capsule.capsule_id,
            "effective_input_limit": self.capsule.budget.max_input_tokens,
            "input_tokens": self.capsule.input_tokens,
        }
        if any(
            getattr(self.receipt, name) != expected
            for name, expected in receipt_bindings.items()
        ):
            raise ContextCompilationError(
                "receipt does not bind the complete compiled capsule"
            )
        if self.decisions != self.receipt.decisions:
            raise ContextCompilationError("result decisions do not match receipt")
        evidence = self.receipt.evidence
        if evidence is None:
            raise ContextCompilationError(
                "compiled result requires qualifying required-context evidence"
            )
        selected_by_id = {
            item.reference_id: item for item in self.capsule.evidence
        }
        expansion_by_id = {
            item.reference_id: item
            for item in self.capsule.expansion_references
        }
        omission_by_id = {
            omission.rpartition(":")[0]: omission.rpartition(":")[2]
            for omission in self.capsule.omissions
        }
        decision_by_id = {
            item.reference_id: item for item in self.decisions
        }
        if set(decision_by_id) != set(selected_by_id) | set(expansion_by_id):
            raise ContextCompilationError(
                "selection decisions do not cover the complete candidate set"
            )
        required_ids = {
            item.reference_id
            for item in self.capsule.evidence
            if item.required
        }
        if set(evidence.required_reference_ids) != required_ids:
            raise ContextCompilationError(
                "required-context evidence does not bind required references"
            )
        if set(evidence.selected_reference_ids) != set(selected_by_id):
            raise ContextCompilationError(
                "required-context evidence does not bind selected references"
            )
        if evidence.required_fields != tuple(
            sorted(self.capsule.required_field_names)
        ):
            raise ContextCompilationError(
                "required-context evidence does not bind invariant fields"
            )
        if not self.required_context_preserved:
            raise ContextCompilationError(
                "compiled result does not preserve its invariant context"
            )
        for reference_id, reference in selected_by_id.items():
            decision = decision_by_id[reference_id]
            expected_reason = (
                InclusionReason.REQUIRED
                if reference.required
                else InclusionReason.RANKED_FIT
            )
            if (
                not decision.included
                or decision.reason is not expected_reason
                or decision.priority != reference.priority
            ):
                raise ContextCompilationError(
                    "selection decision does not match selected reference"
                )
        for reference_id, reference in expansion_by_id.items():
            decision = decision_by_id[reference_id]
            if (
                decision.included
                or decision.reason
                not in (
                    ExclusionReason.TOKEN_BUDGET,
                    ExclusionReason.ITEM_LIMIT,
                    ExclusionReason.LOW_VALUE,
                )
                or decision.priority != reference.priority
            ):
                raise ContextCompilationError(
                    "selection decision does not match deferred reference"
                )
            recorded_reason = reference.metadata.get(
                "selection_exclusion_reason", ""
            )
            if recorded_reason and recorded_reason != decision.reason.value:
                raise ContextCompilationError(
                    "deferred reference exclusion reason does not match decision"
                )
            if omission_by_id.get(reference_id) != _capsule_omission_reason(
                decision.reason
            ).value:
                raise ContextCompilationError(
                    "capsule omission reason does not match selection decision"
                )
        if evidence.artifact_digest != _canonical_digest(
            self.capsule.to_record()
        ):
            raise ContextCompilationError(
                "required-context evidence artifact digest does not match capsule"
            )
        resolution = self.receipt.budget_resolution
        if (
            self.capsule.budget.reserved_output_tokens
            != resolution.reserved_output_tokens
            or self.capsule.budget.reserved_tool_tokens
            != resolution.reserved_tool_tokens
        ):
            raise ContextCompilationError(
                "capsule reserves do not match its budget resolution"
            )
        if self.verifier is not None:
            if not isinstance(self.verifier, ContextCompiler):
                raise ContextCompilationError(
                    "context result verifier must be its ContextCompiler"
                )
            self.verifier.verify_compile_result(self)


@dataclass(frozen=True)
class ContextDeltaResult:
    parent_capsule: ContextCapsule
    delta_capsule: ContextDeltaCapsule
    reconstructed_capsule: ContextCapsule
    receipt: ContextDeltaReceipt
    decisions: tuple[EvidenceSelectionDecision, ...]
    verifier: Any = field(default=None, repr=False, compare=False)

    @property
    def capsule(self) -> ContextDeltaCapsule:
        return self.delta_capsule

    @property
    def invariant_core_preserved(self) -> bool:
        """Whether retry reconstruction retained the exact parent core."""

        return bool(
            self.parent_capsule.invariant_core_id
            == self.reconstructed_capsule.invariant_core_id
            and self.parent_capsule.invariant_core
            == self.reconstructed_capsule.invariant_core
        )

    def __post_init__(self) -> None:
        if not isinstance(self.parent_capsule, ContextCapsule):
            raise ContextDeltaError("delta result parent must be a ContextCapsule")
        if not isinstance(self.delta_capsule, ContextDeltaCapsule):
            raise ContextDeltaError(
                "delta result capsule must be a ContextDeltaCapsule"
            )
        if not isinstance(self.reconstructed_capsule, ContextCapsule):
            raise ContextDeltaError(
                "delta result reconstruction must be a ContextCapsule"
            )
        if not isinstance(self.receipt, ContextDeltaReceipt):
            raise ContextDeltaError(
                "delta result receipt must be a ContextDeltaReceipt"
            )
        if self.decisions != self.receipt.decisions:
            raise ContextDeltaError("delta result is not receipt-bound")
        reconstructed = reconstruct_context(
            self.parent_capsule, self.delta_capsule
        )
        if reconstructed != self.reconstructed_capsule:
            raise ContextDeltaError(
                "delta result is not an exact reconstruction of its parent"
            )
        if not self.invariant_core_preserved:
            raise ContextDeltaError(
                "delta result changed the non-truncatable invariant context"
            )
        receipt_bindings = {
            "repository_id": self.parent_capsule.repository_id,
            "tree_id": self.parent_capsule.tree_id,
            "objective_id": self.parent_capsule.objective_id,
            "policy_id": self.parent_capsule.policy_id,
            "policy_revision": self.parent_capsule.policy_revision,
            "parent_capsule_id": self.parent_capsule.capsule_id,
            "delta_capsule_id": self.delta_capsule.capsule_id,
            "reconstructed_capsule_id": self.reconstructed_capsule.capsule_id,
            "full_replay_tokens": self.reconstructed_capsule.input_tokens,
        }
        if any(
            getattr(self.receipt, name) != expected
            for name, expected in receipt_bindings.items()
        ):
            raise ContextDeltaError(
                "delta receipt does not bind the complete parent reconstruction"
            )
        if (
            self.delta_capsule.reconstructed_input_tokens
            != self.reconstructed_capsule.input_tokens
        ):
            raise ContextDeltaError(
                "delta reconstruction token accounting is inconsistent"
            )
        evidence = self.receipt.evidence
        if evidence is None:
            raise ContextDeltaError(
                "delta result requires qualifying delta-retry evidence"
            )
        parent_by_id = {
            item.reference_id: item for item in self.parent_capsule.evidence
        }
        transmitted_by_id = {
            item.reference_id: item for item in self.delta_capsule.evidence
        }
        transmitted_ids = set(transmitted_by_id)
        witnessed_ids = set(evidence.changed_reference_ids).union(
            evidence.requested_reference_ids
        )
        if transmitted_ids != witnessed_ids:
            raise ContextDeltaError(
                "delta witness does not describe the transmitted references"
            )
        if (
            self.delta_capsule.requested_reference_ids
            != evidence.requested_reference_ids
        ):
            raise ContextDeltaError(
                "delta witness does not bind explicitly requested references"
            )
        decisions_by_id = {
            item.reference_id: item for item in self.decisions
        }
        included_ids = {
            item.reference_id for item in self.decisions if item.included
        }
        if included_ids != transmitted_ids:
            raise ContextDeltaError(
                "delta decisions do not bind every transmitted reference"
            )
        for reference_id, decision in decisions_by_id.items():
            transmitted = transmitted_by_id.get(reference_id)
            previous = parent_by_id.get(reference_id)
            if transmitted is not None:
                expected_reason = (
                    InclusionReason.REQUESTED
                    if reference_id
                    in self.delta_capsule.requested_reference_ids
                    else InclusionReason.CHANGED
                )
                if (
                    decision.reason is not expected_reason
                    or decision.priority != transmitted.priority
                    or decision.token_count < transmitted.token_count
                    or decision.unresolved_question
                    != str(transmitted.metadata.get("expansion_question", ""))
                ):
                    raise ContextDeltaError(
                        "delta decision does not match transmitted evidence"
                    )
                if (
                    expected_reason is InclusionReason.REQUESTED
                    and previous != transmitted
                ) or (
                    expected_reason is InclusionReason.CHANGED
                    and previous == transmitted
                ):
                    raise ContextDeltaError(
                        "delta decision misclassifies changed or requested evidence"
                    )
            elif (
                decision.included
                or decision.reason is not ExclusionReason.UNCHANGED
                or previous is None
                or decision.priority != previous.priority
                or decision.token_count < previous.token_count
            ):
                raise ContextDeltaError(
                    "delta decision does not match retained evidence"
                )
        actual_changed_ids = {
            reference_id
            for reference_id, item in transmitted_by_id.items()
            if parent_by_id.get(reference_id) != item
        }
        if set(evidence.changed_reference_ids) != actual_changed_ids:
            raise ContextDeltaError(
                "delta witness does not bind the actual changed references"
            )
        retained_ids = {
            item.reference_id for item in self.reconstructed_capsule.evidence
        }.difference(transmitted_ids)
        if set(evidence.retained_reference_ids) != retained_ids:
            raise ContextDeltaError(
                "delta witness does not bind retained references"
            )
        required_coverage = {
            coverage
            for item in self.reconstructed_capsule.evidence
            if item.required
            for coverage in item.coverage_ids
        }
        if set(evidence.required_coverage_ids) != required_coverage:
            raise ContextDeltaError(
                "delta witness does not bind reconstructed required coverage"
            )
        if (
            evidence.reconstructed_coverage_ids
            != self.reconstructed_capsule.evidence_coverage_ids
            or evidence.required_fields
            != tuple(sorted(self.reconstructed_capsule.required_field_names))
        ):
            raise ContextDeltaError(
                "delta witness does not bind reconstructed context coverage"
            )
        expected_digest = _canonical_digest(
            {
                "parent_capsule_id": self.delta_capsule.parent_capsule_id,
                "delta": self.delta_capsule.to_record(),
                "reconstructed": self.reconstructed_capsule.to_record(),
            }
        )
        if evidence.artifact_digest != expected_digest:
            raise ContextDeltaError(
                "delta witness artifact digest does not match its capsules"
            )
        if self.verifier is not None:
            if not isinstance(self.verifier, ContextCompiler):
                raise ContextDeltaError(
                    "delta result verifier must be its ContextCompiler"
                )
            self.verifier.verify_delta_result(self)


class ContextCompiler:
    """Compile base and retry contexts under one provider-aware budget."""

    def __init__(
        self,
        budget: ContextBudget,
        *,
        tokenizer: Callable[[str], Any] | Any | None = None,
        estimator: CalibratedTokenEstimator | None = None,
        provider_context_window: int | None = None,
        provider_max_input_tokens: int | None = None,
        reserved_output_tokens: int | None = None,
        reserved_tool_tokens: int | None = None,
        value_policy: EvidenceValuePolicy | Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(budget, ContextBudget):
            if not isinstance(budget, Mapping):
                raise ContextCompilationError("budget must be a ContextBudget")
            budget = ContextBudget.from_dict(budget)
        if estimator is not None and tokenizer is not None:
            raise ContextCompilationError(
                "provide tokenizer or estimator, not both"
            )
        self.budget = budget
        self.estimator = estimator or CalibratedTokenEstimator(tokenizer)
        if value_policy is None:
            selected_value_policy = EvidenceValuePolicy()
        elif isinstance(value_policy, EvidenceValuePolicy):
            selected_value_policy = value_policy
        elif isinstance(value_policy, Mapping):
            try:
                selected_value_policy = EvidenceValuePolicy(**dict(value_policy))
            except TypeError as exc:
                raise ContextCompilationError(
                    "value_policy contains unsupported fields"
                ) from exc
        else:
            raise ContextCompilationError(
                "value_policy must be an EvidenceValuePolicy or mapping"
            )
        self.value_policy = selected_value_policy
        self.budget_resolution = budget.resolve_input_limit(
            provider_context_window=provider_context_window,
            provider_max_input_tokens=provider_max_input_tokens,
            reserved_output_tokens=reserved_output_tokens,
            reserved_tool_tokens=reserved_tool_tokens,
        )
        self.effective_input_limit = (
            self.budget_resolution.effective_input_limit
        )
        if self.effective_input_limit < 1:
            raise RequiredContextOverflowError(
                "provider reserves leave no usable input budget"
            )
        self.effective_budget = budget.for_effective_input_limit(
            self.effective_input_limit,
            reserved_output_tokens=(
                self.budget_resolution.reserved_output_tokens
            ),
            reserved_tool_tokens=self.budget_resolution.reserved_tool_tokens,
        )

    def _provider_input_tokens(
        self,
        *,
        repository_id: str,
        tree_id: str,
        objective_id: str,
        objective_revision: str,
        policy_id: str,
        policy_revision: str,
        caller: str,
        stage: str,
        goal: Any,
        authority: Any,
        scope: Any,
        acceptance: Any,
        evidence: Iterable[ContextReference] = (),
    ) -> int:
        selected = tuple(evidence)
        canonical_count = self.estimator.estimate(
            context_provider_input_payload(
                repository_id=repository_id,
                tree_id=tree_id,
                objective_id=objective_id,
                objective_revision=objective_revision,
                policy_id=policy_id,
                policy_revision=policy_revision,
                caller=caller,
                stage=stage,
                goal=goal,
                authority=authority,
                scope=scope,
                acceptance=acceptance,
                evidence=selected,
            )
        )
        # Component accounting is conservative for tokenizers whose result is
        # not additive across JSON boundaries and for references carrying a
        # larger producer-observed count than the local tokenizer.
        component_count = self.estimator.estimate(
            context_provider_input_payload(
                repository_id=repository_id,
                tree_id=tree_id,
                objective_id=objective_id,
                objective_revision=objective_revision,
                policy_id=policy_id,
                policy_revision=policy_revision,
                caller=caller,
                stage=stage,
                goal=goal,
                authority=authority,
                scope=scope,
                acceptance=acceptance,
                evidence=(),
            )
        ) + sum(_reference_tokens(self.estimator, item) for item in selected)
        return max(canonical_count, component_count)

    def estimate_capsule_input(self, capsule: ContextCapsule) -> int:
        """Independently recompute conservative provider input accounting."""

        if not isinstance(capsule, ContextCapsule):
            raise ContextCompilationError("capsule must be a ContextCapsule")
        return self._provider_input_tokens(
            repository_id=capsule.repository_id,
            tree_id=capsule.tree_id,
            objective_id=capsule.objective_id,
            objective_revision=capsule.objective_revision,
            policy_id=capsule.policy_id,
            policy_revision=capsule.policy_revision,
            caller=capsule.caller,
            stage=capsule.stage,
            goal=capsule.goal,
            authority=capsule.authority,
            scope=capsule.scope,
            acceptance=capsule.acceptance,
            evidence=capsule.evidence,
        )

    def verify_compile_result(
        self,
        result: ContextCompileResult,
    ) -> ContextCompileResult:
        """Remeasure one base context against its original provider policy."""

        if not isinstance(result, ContextCompileResult):
            raise ContextCompilationError(
                "result must be a ContextCompileResult"
            )
        if result.receipt.budget_resolution != self.budget_resolution:
            raise ContextCompilationError(
                "context result budget resolution does not match its verifier"
            )
        actual = self.estimate_capsule_input(result.capsule)
        if actual != result.capsule.input_tokens:
            raise ContextCompilationError(
                "context provider token accounting is not reproducible"
            )
        if actual > self.effective_input_limit:
            raise ContextCompilationError(
                "context result exceeds its verified effective input limit"
            )
        references = {
            item.reference_id: item
            for item in (
                *result.capsule.evidence,
                *result.capsule.expansion_references,
            )
        }
        for decision in result.decisions:
            reference = references[decision.reference_id]
            tokens = (
                _reference_tokens(self.estimator, reference)
                if decision.included
                else decision.token_count
            )
            if tokens < reference.token_count:
                raise ContextCompilationError(
                    "selection decision understates reference tokens"
                )
            unpenalized = self.value_policy.estimate(
                reference,
                token_cost=tokens,
            )
            expected_score = (
                unpenalized.raw_value_score
                * MAX_VALUE_BPS
                // (MAX_VALUE_BPS + decision.diversity_penalty_bps)
            )
            expected_fields = {
                "token_count": tokens,
                "expected_decision_change_bps": (
                    unpenalized.expected_decision_change_bps
                ),
                "uncertainty_bps": unpenalized.uncertainty_bps,
                "uncertainty_reduction_bps": (
                    unpenalized.uncertainty_reduction_bps
                ),
                "latency_cost": unpenalized.latency_cost,
                "invalidation_cost": unpenalized.invalidation_cost,
                "expansion_cost": unpenalized.expansion_cost,
                "diversity_key": unpenalized.diversity_key,
                "value_score": expected_score,
            }
            if any(
                getattr(decision, name) != value
                for name, value in expected_fields.items()
            ):
                raise ContextCompilationError(
                    "value-of-information decision is not reproducible"
                )
            if decision.diversity_penalty_bps and (
                not decision.diversity_key
                or not self.value_policy.diversity_penalty_bps
                or decision.diversity_penalty_bps
                % self.value_policy.diversity_penalty_bps
                or decision.diversity_penalty_bps > MAX_VALUE_BPS
            ):
                raise ContextCompilationError(
                    "diversity penalty is not reproducible"
                )
            if decision.reason is ExclusionReason.LOW_VALUE and (
                decision.value_score >= self.value_policy.minimum_value_score
            ):
                raise ContextCompilationError(
                    "low-value exclusion is not reproducible"
                )
            if (
                decision.included
                and decision.reason is not InclusionReason.REQUIRED
                and decision.value_score < self.value_policy.minimum_value_score
            ):
                raise ContextCompilationError(
                    "selected optional evidence is below the value threshold"
                )
        return result

    def verify_delta_result(
        self, result: ContextDeltaResult
    ) -> ContextDeltaResult:
        """Remeasure and return one exact-parent delta proof.

        ``ContextDeltaResult`` independently verifies all structural,
        identity, coverage, and reconstruction bindings.  Provider token
        accounting additionally requires the same effective tokenizer used
        by this compiler, so this final producer gate recomputes both sides
        instead of trusting counts carried by the receipt.
        """

        if not isinstance(result, ContextDeltaResult):
            raise ContextDeltaError("result must be a ContextDeltaResult")
        reconstructed_tokens = self.estimate_capsule_input(
            result.reconstructed_capsule
        )
        if (
            reconstructed_tokens
            != result.reconstructed_capsule.input_tokens
            or result.receipt.full_replay_tokens != reconstructed_tokens
        ):
            raise ContextDeltaError(
                "delta full-replay token accounting is not reproducible"
            )
        delta_tokens = self.estimator.estimate(
            result.delta_capsule.to_record()
        )
        if delta_tokens != result.receipt.delta_tokens:
            raise ContextDeltaError(
                "delta transmitted token accounting is not reproducible"
            )
        if (
            delta_tokens >= reconstructed_tokens
            or delta_tokens > self.effective_input_limit
        ):
            raise ContextDeltaError(
                "delta no longer qualifies against the effective token budget"
            )
        return result

    def verify_prefix_result(
        self, result: PrefixContextResult
    ) -> PrefixContextResult:
        """Remeasure the segmented input and its current evidence binding."""

        if not isinstance(result, PrefixContextResult):
            raise PrefixContextError(
                "result must be a PrefixContextResult"
            )
        base_verifier = result.context_result.verifier
        if isinstance(base_verifier, ContextCompiler):
            base_verifier.verify_compile_result(result.context_result)
        else:
            self.verify_compile_result(result.context_result)
        stable_tokens = self.estimator.estimate(
            result.capsule.stable_prefix_bytes
        )
        input_tokens = _prefix_provider_input_tokens(
            self.estimator,
            result.capsule.context_capsule,
        )
        if (
            stable_tokens != result.capsule.stable_prefix_tokens
            or input_tokens != result.capsule.provider_input_tokens
        ):
            raise PrefixContextError(
                "prefix-stable provider token accounting is not reproducible"
            )
        if (
            result.capsule.effective_input_limit
            != self.effective_input_limit
        ):
            raise PrefixContextError(
                "prefix result effective limit does not match its verifier"
            )
        if input_tokens > self.effective_input_limit:
            raise PrefixContextError(
                "prefix-stable provider input exceeds its verified budget"
            )
        if result.receipt.reused_prefix_tokens > stable_tokens:
            raise PrefixContextError(
                "prefix reuse exceeds the remeasured stable prefix"
            )
        return result

    def compile_prefix_context(
        self,
        *,
        provider_id: str,
        model_id: str,
        previous: (
            PrefixContextResult | PrefixStableContextCapsule | None
        ) = None,
        provider_cache_id: str = "",
        provider_cache_kind: PrefixCacheKind | str | None = None,
        provider_reused_tokens: int | None = None,
        **kwargs: Any,
    ) -> PrefixContextResult:
        """Compile an ordered prefix/core/evidence stage input.

        Evidence is deliberately excluded from the prefix identity.  A warm
        predecessor therefore remains eligible when evidence changes, while
        every policy, objective, authority, target, caller, stage, provider,
        or model dependency invalidates reuse.
        """

        provider_id = _text(provider_id, "provider_id")
        model_id = _text(model_id, "model_id")
        if provider_reused_tokens is not None:
            provider_reused_tokens = _integer(
                provider_reused_tokens, "provider_reused_tokens"
            )
        cache_id = _text(
            provider_cache_id, "provider_cache_id", required=False
        )
        raw_kind = provider_cache_kind
        if raw_kind is None:
            cache_kind = (
                PrefixCacheKind.PROMPT_CACHE
                if cache_id
                else PrefixCacheKind.DERIVED
            )
        else:
            kind_value = str(getattr(raw_kind, "value", raw_kind))
            aliases = {
                PrefixReuseSource.PROVIDER_PROMPT_CACHE.value: (
                    PrefixCacheKind.PROMPT_CACHE.value
                ),
                PrefixReuseSource.PROVIDER_KV_CACHE.value: (
                    PrefixCacheKind.KV_CACHE.value
                ),
            }
            try:
                cache_kind = PrefixCacheKind(
                    aliases.get(kind_value, kind_value)
                )
            except ValueError as exc:
                raise PrefixContextError(
                    "provider_cache_kind is not supported"
                ) from exc
        if cache_kind is PrefixCacheKind.DERIVED:
            if cache_id:
                raise PrefixContextError(
                    "provider_cache_id requires a provider cache kind"
                )
            if provider_reused_tokens is not None:
                raise PrefixContextError(
                    "provider reused tokens require a provider cache identity"
                )
        elif not cache_id:
            raise PrefixContextError(
                "provider cache reuse requires provider_cache_id"
            )

        context_result = self.compile(**kwargs)
        prefix_input_tokens = _prefix_provider_input_tokens(
            self.estimator,
            context_result.capsule,
        )
        while prefix_input_tokens > self.effective_input_limit:
            overflow = prefix_input_tokens - self.effective_input_limit
            reduced_limit = (
                context_result.capsule.budget.max_input_tokens
                - overflow
                - 1
            )
            if reduced_limit < 1:
                raise RequiredContextOverflowError(
                    "required prefix-stable policy/objective and task core "
                    "exceed the effective provider input budget"
                )
            reduced_budget = self.effective_budget.for_effective_input_limit(
                reduced_limit,
                reserved_output_tokens=(
                    self.budget_resolution.reserved_output_tokens
                ),
                reserved_tool_tokens=(
                    self.budget_resolution.reserved_tool_tokens
                ),
            )
            selection_compiler = ContextCompiler(
                reduced_budget,
                estimator=self.estimator,
            )
            context_result = selection_compiler.compile(**kwargs)
            prefix_input_tokens = _prefix_provider_input_tokens(
                self.estimator,
                context_result.capsule,
            )
        capsule = PrefixStableContextCapsule(
            context_capsule=context_result.capsule,
            provider_id=provider_id,
            model_id=model_id,
            stable_prefix_tokens=self.estimator.estimate(
                _stable_prefix_bytes(context_result.capsule)
            ),
            provider_input_tokens=prefix_input_tokens,
            effective_input_limit=self.effective_input_limit,
        )
        previous_capsule: PrefixStableContextCapsule | None
        if previous is None:
            previous_capsule = None
        elif isinstance(previous, PrefixContextResult):
            previous_capsule = previous.capsule
        elif isinstance(previous, PrefixStableContextCapsule):
            previous_capsule = previous
        else:
            raise PrefixContextError(
                "previous must be a PrefixContextResult or "
                "PrefixStableContextCapsule"
            )

        invalidated: tuple[str, ...] = ()
        previous_capsule_id = ""
        previous_prefix_id = ""
        if previous_capsule is not None:
            previous_capsule_id = previous_capsule.capsule_id
            previous_prefix_id = previous_capsule.semantic_prefix_id
            current_values = _prefix_dependency_values(capsule)
            previous_values = _prefix_dependency_values(previous_capsule)
            invalidated = tuple(
                sorted(
                    name
                    for name, value in current_values.items()
                    if canonical_context_json_bytes(value)
                    != canonical_context_json_bytes(
                        previous_values.get(name)
                    )
                )
            )
            semantic_change = (
                previous_capsule.semantic_prefix_id
                != capsule.semantic_prefix_id
            )
            provider_change = any(
                name in invalidated
                for name in ("provider_id", "model_id")
            )
            if semantic_change != bool(
                set(invalidated).difference(
                    {"provider_id", "model_id"}
                )
            ):
                raise PrefixContextError(
                    "semantic prefix invalidation is inconsistent"
                )
            if provider_change and not invalidated:
                raise PrefixContextError(
                    "provider boundary invalidation is inconsistent"
                )

        if previous_capsule is None:
            if provider_reused_tokens:
                raise PrefixCacheBoundaryError(
                    "provider cache reuse requires a verified warm "
                    "predecessor with matching authority and target"
                )
            decision = PrefixCacheDecision.COLD
            reused_tokens = 0
            source = (
                PrefixReuseSource.COLD
                if provider_reused_tokens is None
                else PrefixReuseSource.PROVIDER_PROMPT_CACHE
                if cache_kind is PrefixCacheKind.PROMPT_CACHE
                else PrefixReuseSource.PROVIDER_KV_CACHE
                if cache_kind is PrefixCacheKind.KV_CACHE
                else PrefixReuseSource.COLD
            )
        elif invalidated:
            if provider_reused_tokens:
                raise PrefixCacheBoundaryError(
                    "provider reported cache reuse after a semantic, "
                    "authority, or target dependency changed"
                )
            decision = PrefixCacheDecision.INVALIDATED
            reused_tokens = 0
            source = (
                PrefixReuseSource.COLD
                if provider_reused_tokens is None
                else PrefixReuseSource.PROVIDER_PROMPT_CACHE
                if cache_kind is PrefixCacheKind.PROMPT_CACHE
                else PrefixReuseSource.PROVIDER_KV_CACHE
                if cache_kind is PrefixCacheKind.KV_CACHE
                else PrefixReuseSource.COLD
            )
        elif provider_reused_tokens is not None:
            reused_tokens = min(
                provider_reused_tokens, capsule.stable_prefix_tokens
            )
            decision = (
                PrefixCacheDecision.HIT
                if reused_tokens
                else PrefixCacheDecision.MISS
            )
            source = (
                PrefixReuseSource.PROVIDER_PROMPT_CACHE
                if cache_kind is PrefixCacheKind.PROMPT_CACHE
                else PrefixReuseSource.PROVIDER_KV_CACHE
            )
        else:
            reused_tokens = max(
                1,
                capsule.stable_prefix_tokens
                * CONSERVATIVE_PREFIX_REUSE_BPS
                // 10_000,
            )
            reused_tokens = min(
                reused_tokens, capsule.stable_prefix_tokens
            )
            decision = PrefixCacheDecision.HIT
            source = PrefixReuseSource.CONSERVATIVE_ESTIMATE

        identity = capsule.cache_identity(
            cache_kind=cache_kind,
            provider_cache_id=cache_id,
        )
        base = context_result.capsule
        receipt = PrefixReuseReceipt(
            capsule_id=capsule.capsule_id,
            context_capsule_id=base.capsule_id,
            repository_id=base.repository_id,
            tree_id=base.tree_id,
            objective_id=base.objective_id,
            objective_revision=base.objective_revision,
            policy_id=base.policy_id,
            policy_revision=base.policy_revision,
            caller=base.caller,
            stage=base.stage,
            cache_identity=identity,
            previous_capsule_id=previous_capsule_id,
            previous_semantic_prefix_id=previous_prefix_id,
            reuse_source=source,
            cache_decision=decision,
            eligible_stable_prefix_tokens=capsule.stable_prefix_tokens,
            reused_prefix_tokens=reused_tokens,
            provider_input_tokens=capsule.provider_input_tokens,
            provider_reused_tokens=provider_reused_tokens,
            invalidated_dependencies=invalidated,
            evidence_reference_ids=tuple(
                item.reference_id for item in base.evidence
            ),
            evidence_digest=capsule.evidence_digest,
        )
        return PrefixContextResult(
            context_result=context_result,
            capsule=capsule,
            receipt=receipt,
            verifier=self,
        )

    compile_prefix = compile_prefix_context

    def compile(
        self,
        *,
        repository_id: str,
        tree_id: str,
        objective_id: str,
        objective_revision: str,
        policy_id: str,
        policy_revision: str,
        caller: str,
        stage: str,
        goal: Any,
        authority: Any,
        scope: Any,
        acceptance: Any,
        evidence: Iterable[ContextReference | Mapping[str, Any]] = (),
    ) -> ContextCompileResult:
        references = _coerce_references(evidence)
        required = tuple(item for item in references if item.required)
        optional = tuple(item for item in references if not item.required)
        normalized = ContextCapsule(
            repository_id=repository_id,
            tree_id=tree_id,
            objective_id=objective_id,
            objective_revision=objective_revision,
            policy_id=policy_id,
            policy_revision=policy_revision,
            caller=caller,
            stage=stage,
            budget=self.effective_budget,
            goal=goal,
            authority=authority,
            scope=scope,
            acceptance=acceptance,
        )
        input_arguments = {
            "repository_id": normalized.repository_id,
            "tree_id": normalized.tree_id,
            "objective_id": normalized.objective_id,
            "objective_revision": normalized.objective_revision,
            "policy_id": normalized.policy_id,
            "policy_revision": normalized.policy_revision,
            "caller": normalized.caller,
            "stage": normalized.stage,
            "goal": normalized.goal,
            "authority": normalized.authority,
            "scope": normalized.scope,
            "acceptance": normalized.acceptance,
        }
        base_tokens = self._provider_input_tokens(
            **input_arguments,
            evidence=(),
        )
        selected: list[ContextReference] = []
        decisions: dict[str, EvidenceSelectionDecision] = {}
        diversity_counts: dict[str, int] = {}
        used = base_tokens
        if used > self.effective_input_limit:
            raise RequiredContextOverflowError(
                "invariant goal/authority/scope/acceptance exceeds "
                "the effective provider input budget"
            )
        for item in sorted(required, key=lambda member: member.reference_id):
            tokens = _reference_tokens(self.estimator, item)
            estimate = self.value_policy.estimate(item, token_cost=tokens)
            proposed = self._provider_input_tokens(
                **input_arguments,
                evidence=(*selected, item),
            )
            if (
                len(selected) >= self.effective_budget.max_items
                or proposed > self.effective_input_limit
            ):
                raise RequiredContextOverflowError(
                    f"required evidence {item.reference_id!r} does not fit "
                    "the effective provider input budget"
                )
            selected.append(item)
            used = proposed
            if estimate.diversity_key:
                diversity_counts[estimate.diversity_key] = (
                    diversity_counts.get(estimate.diversity_key, 0) + 1
                )
            decisions[item.reference_id] = EvidenceSelectionDecision(
                item.reference_id,
                True,
                InclusionReason.REQUIRED,
                tokens,
                item.priority,
                expected_decision_change_bps=(
                    estimate.expected_decision_change_bps
                ),
                uncertainty_bps=estimate.uncertainty_bps,
                uncertainty_reduction_bps=(
                    estimate.uncertainty_reduction_bps
                ),
                latency_cost=estimate.latency_cost,
                invalidation_cost=estimate.invalidation_cost,
                expansion_cost=estimate.expansion_cost,
                value_score=estimate.value_score,
                diversity_key=estimate.diversity_key,
                diversity_penalty_bps=estimate.diversity_penalty_bps,
            )
        omitted: list[ContextReference] = []
        optional_selected = 0
        remaining = {item.reference_id: item for item in optional}
        while remaining:
            ranked: list[
                tuple[ContextReference, int, EvidenceValueEstimate]
            ] = []
            for item in remaining.values():
                tokens = _reference_tokens(self.estimator, item)
                diversity_key = self.value_policy._diversity_key(item)
                estimate = self.value_policy.estimate(
                    item,
                    token_cost=tokens,
                    selected_diversity_count=diversity_counts.get(
                        diversity_key, 0
                    ),
                )
                ranked.append((item, tokens, estimate))
            item, tokens, estimate = min(
                ranked,
                key=lambda member: (
                    -member[2].value_score,
                    -member[2].uncertainty_reduction_bps,
                    -member[2].expected_decision_change_bps,
                    -member[0].priority,
                    member[0].tier.value,
                    member[0].reference_id,
                    member[0].reference_content_id,
                ),
            )
            remaining.pop(item.reference_id)
            proposed = self._provider_input_tokens(
                **input_arguments,
                evidence=(*selected, item),
            )
            if estimate.value_score < self.value_policy.minimum_value_score:
                reason = ExclusionReason.LOW_VALUE
            elif (
                self.value_policy.max_optional_items is not None
                and optional_selected >= self.value_policy.max_optional_items
            ):
                reason = ExclusionReason.ITEM_LIMIT
            elif len(selected) >= self.effective_budget.max_items:
                reason = ExclusionReason.ITEM_LIMIT
            elif proposed > self.effective_input_limit:
                reason = ExclusionReason.TOKEN_BUDGET
            else:
                selected.append(item)
                optional_selected += 1
                used = proposed
                if estimate.diversity_key:
                    diversity_counts[estimate.diversity_key] = (
                        diversity_counts.get(estimate.diversity_key, 0) + 1
                    )
                decisions[item.reference_id] = EvidenceSelectionDecision(
                    item.reference_id,
                    True,
                    InclusionReason.RANKED_FIT,
                    tokens,
                    item.priority,
                    expected_decision_change_bps=(
                        estimate.expected_decision_change_bps
                    ),
                    uncertainty_bps=estimate.uncertainty_bps,
                    uncertainty_reduction_bps=(
                        estimate.uncertainty_reduction_bps
                    ),
                    latency_cost=estimate.latency_cost,
                    invalidation_cost=estimate.invalidation_cost,
                    expansion_cost=estimate.expansion_cost,
                    value_score=estimate.value_score,
                    diversity_key=estimate.diversity_key,
                    diversity_penalty_bps=estimate.diversity_penalty_bps,
                )
                continue
            omitted.append(item)
            decisions[item.reference_id] = EvidenceSelectionDecision(
                item.reference_id,
                False,
                reason,
                tokens,
                item.priority,
                expected_decision_change_bps=(
                    estimate.expected_decision_change_bps
                ),
                uncertainty_bps=estimate.uncertainty_bps,
                uncertainty_reduction_bps=(
                    estimate.uncertainty_reduction_bps
                ),
                latency_cost=estimate.latency_cost,
                invalidation_cost=estimate.invalidation_cost,
                expansion_cost=estimate.expansion_cost,
                value_score=estimate.value_score,
                diversity_key=estimate.diversity_key,
                diversity_penalty_bps=estimate.diversity_penalty_bps,
            )
        ordered_decisions = tuple(
            decisions[key] for key in sorted(decisions)
        )
        capsule = ContextCapsule(
            repository_id=repository_id,
            tree_id=tree_id,
            objective_id=objective_id,
            objective_revision=objective_revision,
            policy_id=policy_id,
            policy_revision=policy_revision,
            caller=caller,
            stage=stage,
            budget=self.effective_budget,
            goal=goal,
            authority=authority,
            scope=scope,
            acceptance=acceptance,
            evidence=tuple(selected),
            expansion_references=tuple(
                _as_expansion(
                    item,
                    exclusion_reason=decisions[item.reference_id].reason,
                )
                for item in omitted
            ),
            input_tokens=used,
            truncated=bool(omitted),
            omissions=tuple(
                (
                    f"{item.reference_id}:"
                    f"{_capsule_omission_reason(decisions[item.reference_id].reason).value}"
                )
                for item in omitted
            ),
        )
        selected_ids = tuple(item.reference_id for item in capsule.evidence)
        required_ids = tuple(item.reference_id for item in required)
        witness = RequiredContextBudgetEvidence(
            repository_id=capsule.repository_id,
            tree_id=capsule.tree_id,
            policy_id=capsule.policy_id,
            policy_revision=capsule.policy_revision,
            capsule_id=capsule.capsule_id,
            budget_resolution=self.budget_resolution,
            effective_input_limit=self.effective_input_limit,
            input_tokens=capsule.input_tokens,
            required_fields=capsule.required_field_names,
            required_reference_ids=required_ids,
            selected_reference_ids=selected_ids,
            artifact_digest=_canonical_digest(capsule.to_record()),
        )
        receipt = ContextCompilationReceipt(
            repository_id=capsule.repository_id,
            tree_id=capsule.tree_id,
            objective_id=capsule.objective_id,
            policy_id=capsule.policy_id,
            policy_revision=capsule.policy_revision,
            stage=capsule.stage,
            capsule_id=capsule.capsule_id,
            budget_resolution=self.budget_resolution,
            effective_input_limit=self.effective_input_limit,
            input_tokens=capsule.input_tokens,
            estimator_name=self.estimator.name,
            estimator_error_bps=self.estimator.error_bps,
            decisions=ordered_decisions,
            evidence=witness,
        )
        return ContextCompileResult(
            capsule,
            receipt,
            ordered_decisions,
            self,
        )

    compile_context = compile

    def compile_delta(
        self,
        parent: ContextCapsule,
        *,
        evidence: Iterable[ContextReference | Mapping[str, Any]],
        requested_reference_ids: Iterable[str] = (),
        stage: str | None = None,
    ) -> ContextDeltaResult:
        if not isinstance(parent, ContextCapsule):
            raise ContextDeltaError("parent must be a ContextCapsule")
        if parent.is_delta:
            raise ContextDeltaError(
                "delta chaining requires reconstruction of the prior delta"
            )
        candidates = _coerce_references(evidence)
        candidate_by_id = {item.reference_id: item for item in candidates}
        parent_by_id = {item.reference_id: item for item in parent.evidence}
        requested = set(
            _strings(requested_reference_ids, "requested_reference_ids")
        )
        unknown_requests = requested.difference(candidate_by_id)
        if unknown_requests:
            raise ContextDeltaError(
                "requested retry evidence is not present in the candidate set"
            )
        required_ids = {
            item.reference_id
            for item in parent.evidence
            if item.required
        }
        missing_required = required_ids.difference(candidate_by_id)
        if missing_required:
            raise ContextDeltaError(
                "retry candidate drops required evidence references"
            )
        downgraded_required = {
            reference_id
            for reference_id in required_ids
            if not candidate_by_id[reference_id].required
        }
        if downgraded_required:
            raise ContextDeltaError(
                "retry candidate downgrades parent-required evidence"
            )
        transmitted: list[ContextReference] = []
        genuinely_changed: list[str] = []
        requested_only: list[str] = []
        decisions: list[EvidenceSelectionDecision] = []
        for item in candidates:
            previous = parent_by_id.get(item.reference_id)
            is_changed = (
                previous is None
                or previous.reference_content_id != item.reference_content_id
                or previous.to_dict() != item.to_dict()
            )
            tokens = _reference_tokens(self.estimator, item)
            if is_changed or item.reference_id in requested:
                transmitted.append(item)
                if is_changed:
                    genuinely_changed.append(item.reference_id)
                else:
                    requested_only.append(item.reference_id)
                decisions.append(
                    EvidenceSelectionDecision(
                        item.reference_id,
                        True,
                        (
                            InclusionReason.CHANGED
                            if is_changed
                            else InclusionReason.REQUESTED
                        ),
                        tokens,
                        item.priority,
                        unresolved_question=str(
                            item.metadata.get("expansion_question", "")
                        ),
                    )
                )
            else:
                decisions.append(
                    EvidenceSelectionDecision(
                        item.reference_id,
                        False,
                        ExclusionReason.UNCHANGED,
                        tokens,
                        item.priority,
                    )
                )
        if not transmitted:
            raise ContextDeltaError(
                "retry delta must contain changed or explicitly requested evidence"
            )
        combined_by_id = dict(parent_by_id)
        combined_by_id.update(candidate_by_id)
        combined = tuple(combined_by_id[key] for key in sorted(combined_by_id))
        reconstructed_input_tokens = self._provider_input_tokens(
            repository_id=parent.repository_id,
            tree_id=parent.tree_id,
            objective_id=parent.objective_id,
            objective_revision=parent.objective_revision,
            policy_id=parent.policy_id,
            policy_revision=parent.policy_revision,
            caller=parent.caller,
            stage=stage or parent.stage,
            goal=parent.goal,
            authority=parent.authority,
            scope=parent.scope,
            acceptance=parent.acceptance,
            evidence=combined,
        )
        reconstructed_limit = min(
            parent.budget.max_input_tokens, self.effective_input_limit
        )
        if reconstructed_input_tokens > reconstructed_limit:
            raise ContextDeltaError(
                "reconstructed full context exceeds the effective input budget"
            )
        delta_capsule = ContextDeltaCapsule(
            parent_capsule_id=parent.capsule_id,
            stage=stage or parent.stage,
            evidence=tuple(transmitted),
            reconstructed_input_tokens=reconstructed_input_tokens,
            requested_reference_ids=tuple(requested_only),
        )
        if len(delta_capsule.canonical_bytes()) > (
            self.effective_budget.max_serialized_bytes
        ):
            raise ContextDeltaError(
                "retry delta exceeds the serialized-byte budget"
            )
        reconstructed = reconstruct_context(parent, delta_capsule)
        # The delta is its compact provider wire record.  Full replay is the
        # canonical provider input, conservatively floored by the same
        # component accounting used for the effective-budget check; it does
        # not include supervisor-only budget or omission metadata.
        delta_tokens = self.estimator.estimate(delta_capsule.to_record())
        full_replay_tokens = max(
            reconstructed.input_tokens,
            self.estimator.estimate(reconstructed.provider_input_payload),
        )
        if delta_tokens >= full_replay_tokens:
            raise ContextDeltaError(
                "retry delta does not use fewer tokens than full replay"
            )
        if delta_tokens > self.effective_input_limit:
            raise ContextDeltaError("retry delta exceeds effective input budget")
        parent_required_coverage = {
            coverage
            for item in parent.evidence
            if item.required
            for coverage in item.coverage_ids
        }
        required_coverage = {
            coverage
            for item in reconstructed.evidence
            if item.required
            for coverage in item.coverage_ids
        }
        reconstructed_coverage = set(reconstructed.evidence_coverage_ids)
        if (
            not parent_required_coverage.issubset(required_coverage)
            or not required_coverage.issubset(reconstructed_coverage)
        ):
            raise ContextDeltaError("retry reconstruction loses required coverage")
        ordered_decisions = tuple(
            sorted(decisions, key=lambda item: item.reference_id)
        )
        witness = DeltaRetryContextEvidence(
            repository_id=parent.repository_id,
            tree_id=parent.tree_id,
            policy_id=parent.policy_id,
            policy_revision=parent.policy_revision,
            parent_capsule_id=parent.capsule_id,
            delta_capsule_id=delta_capsule.capsule_id,
            reconstructed_capsule_id=reconstructed.capsule_id,
            full_replay_tokens=full_replay_tokens,
            delta_tokens=delta_tokens,
            required_coverage_ids=tuple(required_coverage),
            reconstructed_coverage_ids=reconstructed.evidence_coverage_ids,
            changed_reference_ids=tuple(genuinely_changed),
            requested_reference_ids=tuple(requested_only),
            retained_reference_ids=tuple(
                item.reference_id
                for item in reconstructed.evidence
                if item.reference_id
                not in {member.reference_id for member in transmitted}
            ),
            required_fields=reconstructed.required_field_names,
            artifact_digest=_canonical_digest(
                {
                    "parent_capsule_id": parent.capsule_id,
                    "delta": delta_capsule.to_record(),
                    "reconstructed": reconstructed.to_record(),
                }
            ),
        )
        receipt = ContextDeltaReceipt(
            repository_id=parent.repository_id,
            tree_id=parent.tree_id,
            objective_id=parent.objective_id,
            policy_id=parent.policy_id,
            policy_revision=parent.policy_revision,
            parent_capsule_id=parent.capsule_id,
            delta_capsule_id=delta_capsule.capsule_id,
            reconstructed_capsule_id=reconstructed.capsule_id,
            full_replay_tokens=full_replay_tokens,
            delta_tokens=delta_tokens,
            decisions=ordered_decisions,
            evidence=witness,
        )
        result = ContextDeltaResult(
            parent,
            delta_capsule,
            reconstructed,
            receipt,
            ordered_decisions,
            self,
        )
        return result

    compile_retry = compile_delta

    def compile_decision_context(
        self,
        request: Any,
        graph: Any,
        retrieval_receipt: Any,
        **kwargs: Any,
    ) -> Any:
        """Compile a generation-3 mandatory decision context.

        The local import keeps the shared generation-1/2 compiler usable
        without importing semantic-graph contracts during module import.
        """

        return DecisionContextCompiler.from_context_compiler(self).compile(
            request,
            graph,
            retrieval_receipt,
            **kwargs,
        )


def reconstruct_context(
    parent: ContextCapsule, delta: ContextDeltaCapsule
) -> ContextCapsule:
    """Apply a parent-bound delta and return the deterministic full context."""

    if not isinstance(parent, ContextCapsule) or not isinstance(
        delta, ContextDeltaCapsule
    ):
        raise ContextDeltaError(
            "parent must be a ContextCapsule and delta a ContextDeltaCapsule"
        )
    if delta.parent_capsule_id != parent.capsule_id:
        raise ContextDeltaError("delta is not bound to the supplied parent")
    parent_by_id = {
        item.reference_id: item for item in parent.evidence
    }
    requested_ids = set(delta.requested_reference_ids)
    for item in delta.evidence:
        if item.repository_id and item.repository_id != parent.repository_id:
            raise ContextDeltaError(
                "delta evidence changes immutable repository identity"
            )
        if item.tree_id and item.tree_id != parent.tree_id:
            raise ContextDeltaError(
                "delta evidence changes immutable tree identity"
            )
        if (
            parent_by_id.get(item.reference_id) == item
            and item.reference_id not in requested_ids
        ):
            raise ContextDeltaError(
                "delta replays unchanged evidence without an explicit request"
            )
    combined = dict(parent_by_id)
    combined.update({item.reference_id: item for item in delta.evidence})
    required_ids = {
        item.reference_id for item in parent.evidence if item.required
    }
    if not required_ids.issubset(combined) or any(
        not combined[reference_id].required for reference_id in required_ids
    ):
        raise ContextDeltaError(
            "reconstructed context loses or downgrades required evidence"
        )
    evidence = tuple(combined[key] for key in sorted(combined))
    reconstructed_tokens = delta.reconstructed_input_tokens
    if reconstructed_tokens > parent.budget.max_input_tokens:
        raise ContextDeltaError(
            "reconstructed context exceeds the parent input budget"
        )
    parent_declared_reference_tokens = sum(
        item.token_count for item in parent.evidence
    )
    inherited_core_floor = max(
        0, parent.input_tokens - parent_declared_reference_tokens
    )
    reconstructed_floor = inherited_core_floor + sum(
        item.token_count for item in evidence
    )
    if reconstructed_floor > reconstructed_tokens:
        raise ContextDeltaError(
            "reconstructed token count omits inherited core or evidence tokens"
        )
    selected_ids = set(combined)
    expansions = {
        item.reference_id: item
        for item in parent.expansion_references
        if item.reference_id not in selected_ids
    }
    retained_omissions = {
        omission
        for omission in parent.omissions
        if omission.rpartition(":")[0] in expansions
    }
    recorded_expansion_ids = {
        omission.rpartition(":")[0] for omission in retained_omissions
    }
    retained_omissions.update(
        f"{reference_id}:{ExclusionReason.TOKEN_BUDGET.value}"
        for reference_id in expansions
        if reference_id not in recorded_expansion_ids
    )
    return ContextCapsule(
        repository_id=parent.repository_id,
        tree_id=parent.tree_id,
        objective_id=parent.objective_id,
        objective_revision=parent.objective_revision,
        policy_id=parent.policy_id,
        policy_revision=parent.policy_revision,
        caller=parent.caller,
        stage=delta.stage,
        budget=parent.budget,
        goal=parent.goal,
        authority=parent.authority,
        scope=parent.scope,
        acceptance=parent.acceptance,
        evidence=evidence,
        expansion_references=tuple(
            expansions[key] for key in sorted(expansions)
        ),
        input_tokens=reconstructed_tokens,
        truncated=bool(expansions),
        omissions=tuple(sorted(retained_omissions)),
    )


def _decision_retry_wire(capsule: Any) -> dict[str, Any]:
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/decision-context-retry-delta@1",
        "contract_version": CONTEXT_COMPILER_VERSION,
        "parent_decision_request_id": capsule.parent_decision_request_id,
        "parent_context_id": capsule.parent_context_id,
        "parent_completeness_witness_id": capsule.parent_completeness_witness_id,
        "parent_stable_core_id": capsule.parent_stable_core_id,
        "parent_closure_id": capsule.parent_closure_id,
        "changed_dependencies": tuple(
            item.to_record() for item in capsule.changed_dependencies
        ),
        "expanded_evidence": tuple(
            item.to_record() for item in capsule.expanded_evidence
        ),
        "omission_reasons": dict(capsule.omission_reasons),
    }


def _unsafe_decision_delta(value: Any) -> bool:
    forbidden = {
        "authority", "authority_id", "authorization",
        "authorization_state", "requested_authority", "principal_id",
        "capabilities", "capability_ids", "lease_id", "fencing_epoch",
        "repository_id", "dirty_worktree_root", "dirty_worktree_root_id",
        "semantic_root", "semantic_roots", "semantic_roots_digest",
        "semantic_graph_root_id", "root_id", "corpus_query",
        "corpus_browse", "retrieval_query",
    }
    if isinstance(value, Mapping):
        return any(
            str(key) in forbidden or _unsafe_decision_delta(member)
            for key, member in value.items()
        )
    if isinstance(value, (tuple, list)):
        return any(_unsafe_decision_delta(item) for item in value)
    return False


def _decision_references(compilation: Any) -> dict[str, Any]:
    return {
        reference.node_id: reference
        for context in compilation.contexts
        for reference in context.references
    }


@dataclass(frozen=True)
class DecisionContextRetryResult:
    parent_compilation: Any
    retry_capsule: Any
    reconstructed_compilation: Any
    verifier: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.retry_capsule.validate_parent(self.parent_compilation)
        parent, rebuilt = self.parent_compilation, self.reconstructed_compilation
        if (
            rebuilt.stable_core_id != parent.stable_core_id
            or rebuilt.required_core != parent.required_core
            or rebuilt.witness.mandatory_node_ids != parent.witness.mandatory_node_ids
            or rebuilt.witness.mandatory_edge_ids != parent.witness.mandatory_edge_ids
            or rebuilt.witness.dependency_paths != parent.witness.dependency_paths
        ):
            from .decision_context import DecisionContextRetryError

            raise DecisionContextRetryError(
                "retry reconstruction changed stable core or mandatory closure"
            )

    @property
    def capsule(self) -> Any:
        return self.retry_capsule

    @property
    def reconstructed(self) -> Any:
        return self.reconstructed_compilation

    @property
    def transmitted_payload(self) -> Mapping[str, Any]:
        return _decision_retry_wire(self.retry_capsule)

    @property
    def delta_input_tokens(self) -> int:
        return self.retry_capsule.delta_input_tokens

    @property
    def full_replay_input_tokens(self) -> int:
        return self.retry_capsule.full_replay_input_tokens


def reconstruct_decision_context(
    parent: Any, capsule: Any, target: Any | None = None
) -> Any:
    from .decision_context import (
        DecisionContextChangeKind,
        DecisionContextCompilation,
        DecisionContextRetryCapsule,
        DecisionContextRetryError,
    )

    if not isinstance(parent, DecisionContextCompilation) or not isinstance(
        capsule, DecisionContextRetryCapsule
    ):
        raise DecisionContextRetryError("retry reconstruction has invalid types")
    capsule.validate_parent(parent)
    structural = any(
        item.kind
        not in {
            DecisionContextChangeKind.DIAGNOSTICS,
            DecisionContextChangeKind.EXPANDED_EVIDENCE,
        }
        for item in capsule.changed_dependencies
    )
    if structural and target is None:
        raise DecisionContextRetryError(
            "structural reconstruction requires the current verified closure"
        )
    if target is not None:
        rebuilt = DecisionContextCompilation.from_dict(target.to_record())
    else:
        rebuilt = DecisionContextCompilation.from_dict(parent.to_record())
    if (
        rebuilt.stable_core_id != parent.stable_core_id
        or rebuilt.witness.semantic_graph_root_id
        != parent.witness.semantic_graph_root_id
        or rebuilt.witness.roots_digest != parent.witness.roots_digest
        or rebuilt.witness.mandatory_node_ids != parent.witness.mandatory_node_ids
        or rebuilt.witness.mandatory_edge_ids != parent.witness.mandatory_edge_ids
        or rebuilt.witness.dependency_paths != parent.witness.dependency_paths
        or rebuilt.complete_input_tokens != capsule.full_replay_input_tokens
    ):
        raise DecisionContextRetryError(
            "retry target does not preserve its parent closure and accounting"
        )
    previous, current = _decision_references(parent), _decision_references(rebuilt)
    actual_changes = {
        node_id
        for node_id in previous
        if previous[node_id].to_record() != current[node_id].to_record()
    }
    declared_changes: set[str] = set()
    for change in capsule.changed_dependencies:
        if change.kind in {
            DecisionContextChangeKind.DIAGNOSTICS,
            DecisionContextChangeKind.EXPANDED_EVIDENCE,
        }:
            continue
        node = change.dependency_id
        if node not in previous:
            node = next(
                (key for key, item in previous.items() if item.reference_id == node),
                "",
            )
        declared_changes.add(node)
        if (
            node not in current
            or previous[node].node_content_id != change.previous_content_id
            or current[node].node_content_id != change.current_content_id
        ):
            raise DecisionContextRetryError(
                "structural dependency identities do not match retry delta"
            )
    if actual_changes != declared_changes:
        raise DecisionContextRetryError(
            "retry target contains undeclared structural changes"
        )
    return rebuilt


class DecisionContextCompiler:
    """Compile the complete authoritative closure for one decision.

    Required dependencies are enumerated solely by the semantic graph's
    mandatory closure and never enter :class:`EvidenceValuePolicy`.  The exact
    canonical provider payload is remeasured after every representation and
    split decision.
    """

    def __init__(
        self,
        budget: ContextBudget,
        *,
        tokenizer: Callable[[str], Any] | Any | None = None,
        estimator: CalibratedTokenEstimator | None = None,
        provider_context_window: int | None = None,
        provider_max_input_tokens: int | None = None,
        reserved_output_tokens: int | None = None,
        reserved_tool_tokens: int | None = None,
        require_provider_tokenizer: bool = True,
        max_inline_node_bytes: int = 4_096,
        max_inline_bytes: int | None = None,
    ) -> None:
        self.context_compiler = ContextCompiler(
            budget,
            tokenizer=tokenizer,
            estimator=estimator,
            provider_context_window=provider_context_window,
            provider_max_input_tokens=provider_max_input_tokens,
            reserved_output_tokens=reserved_output_tokens,
            reserved_tool_tokens=reserved_tool_tokens,
        )
        self.estimator = self.context_compiler.estimator
        self.effective_input_limit = (
            self.context_compiler.effective_input_limit
        )
        self.effective_budget = self.context_compiler.effective_budget
        self.require_provider_tokenizer = bool(require_provider_tokenizer)
        selected_inline_bytes = (
            max_inline_node_bytes
            if max_inline_bytes is None
            else max_inline_bytes
        )
        self.max_inline_node_bytes = _integer(
            selected_inline_bytes,
            "max_inline_node_bytes",
            minimum=128,
        )
        if self.max_inline_node_bytes > self.effective_budget.max_item_bytes:
            raise ContextCompilationError(
                "max_inline_node_bytes exceeds the context item byte limit"
            )
        self._decision_expansion_usage: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_context_compiler(
        cls, compiler: ContextCompiler
    ) -> "DecisionContextCompiler":
        if not isinstance(compiler, ContextCompiler):
            raise ContextCompilationError(
                "compiler must be a ContextCompiler"
            )
        result = object.__new__(cls)
        result.context_compiler = compiler
        result.estimator = compiler.estimator
        result.effective_input_limit = compiler.effective_input_limit
        result.effective_budget = compiler.effective_budget
        result.require_provider_tokenizer = True
        result.max_inline_node_bytes = min(
            4_096, compiler.effective_budget.max_item_bytes
        )
        result._decision_expansion_usage = {}
        return result

    @staticmethod
    def _has_unknown(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(
                "unknown" in str(key).lower()
                or "unresolved" in str(key).lower()
                or DecisionContextCompiler._has_unknown(member)
                for key, member in value.items()
            )
        if isinstance(value, (tuple, list)):
            return any(
                DecisionContextCompiler._has_unknown(member)
                for member in value
            )
        if isinstance(value, str):
            lowered = value.lower()
            return "unknown" in lowered or "unresolved" in lowered
        return False

    @staticmethod
    def _reference_ids(
        nodes: Iterable[Any],
        predicate: Callable[[Any], bool],
    ) -> tuple[str, ...]:
        return tuple(
            f"mandatory:{node.node_id}"
            for node in nodes
            if predicate(node)
        )

    @staticmethod
    def _path_edge_ids(
        graph: Any,
        path: tuple[str, ...],
    ) -> tuple[str, ...]:
        result: list[str] = []
        for source, target in zip(path, path[1:]):
            candidates = tuple(
                sorted(
                    (
                        edge
                        for edge in graph.edges
                        if edge.source == source
                        and edge.target == target
                        and edge.mandatory
                        and edge.authoritative
                    ),
                    key=lambda edge: (edge.kind.value, edge.edge_id),
                )
            )
            if not candidates:
                from .decision_context import DecisionContextBindingError

                raise DecisionContextBindingError(
                    "mandatory dependency path is not present in the graph"
                )
            result.append(candidates[0].edge_id)
        return tuple(result)

    @staticmethod
    def _summary(node: Any, *, byte_count: int) -> Mapping[str, Any]:
        record_keys = tuple(sorted(str(key) for key in node.record)[:32])
        return {
            "node_id": node.node_id,
            "kind": node.kind.value,
            "content_id": node.content_id,
            "root_id": node.root_id,
            "source_root_id": node.source_root_id,
            "provenance": node.provenance.value,
            "provenance_id": node.provenance_id,
            "trust": node.trust.value,
            "authority": node.authority.value,
            "version": node.version,
            "canonical_byte_count": byte_count,
            "record_keys": record_keys,
        }

    def _build_reference(
        self,
        *,
        request: Any,
        graph: Any,
        node: Any,
        artifact_store: ContentAddressedContextStore | None,
    ) -> Any:
        from .decision_context import (
            DecisionContextReference,
            DecisionContextRepresentation,
            MissingDecisionContextExpansionError,
        )

        body = node.to_dict()
        raw = canonical_context_json_bytes(body)
        reference_id = f"mandatory:{node.node_id}"
        summary = self._summary(node, byte_count=len(raw))
        if len(raw) <= self.max_inline_node_bytes:
            return DecisionContextReference(
                reference_id=reference_id,
                node_id=node.node_id,
                node_kind=node.kind.value,
                node_content_id=node.content_id,
                representation=DecisionContextRepresentation.INLINE,
                summary=summary,
                body=body,
            )
        if artifact_store is None:
            raise MissingDecisionContextExpansionError(
                f"mandatory dependency {node.node_id!r} exceeds the inline "
                "bound and requires a resolvable artifact_store"
            )
        text = raw.decode("utf-8")
        target = artifact_store.put(text)
        question = f"expand mandatory dependency {node.node_id}"
        handle = ContextReference(
            reference_id=reference_id,
            kind="mandatory-decision-dependency",
            tier=ContextTier.EXPANSION,
            referenced_content_id=target,
            repository_id=request.repository_id,
            tree_id=request.dirty_worktree_root.cid_v1,
            byte_count=len(raw),
            metadata={
                "mandatory_node_id": node.node_id,
                "node_content_id": node.content_id,
                "semantic_graph_root_id": graph.root_id,
                "unresolved_questions": (question,),
                "question_bound_expansion": True,
            },
        )
        if artifact_store.get(handle.referenced_content_id) != raw:
            raise MissingDecisionContextExpansionError(
                "mandatory expansion handle did not resolve to its canonical body"
            )
        return DecisionContextReference(
            reference_id=reference_id,
            node_id=node.node_id,
            node_kind=node.kind.value,
            node_content_id=node.content_id,
            representation=DecisionContextRepresentation.EXPANSION,
            summary=summary,
            expansion_handle=handle,
        )

    @staticmethod
    def _bounded_index_metadata(
        graph: Any, receipt: Any
    ) -> Mapping[str, Any]:
        maximum = 1_000_000

        def bounded(value: Any) -> int:
            if isinstance(value, bool) or not isinstance(value, int):
                return 0
            return min(maximum, max(0, value))

        truncation = receipt.truncation
        return {
            "graph_node_count": bounded(len(graph.nodes)),
            "graph_edge_count": bounded(len(graph.edges)),
            "candidate_audit_count": bounded(len(receipt.candidates)),
            "optional_included_count": bounded(
                len(receipt.optional_node_ids)
            ),
            "optional_omitted_count": bounded(
                len(receipt.omitted_node_ids)
            ),
            "candidate_truncation_count": bounded(
                truncation.get("candidate_truncation_count", 0)
            ),
            "counts_saturated": any(
                value >= maximum
                for value in (
                    len(graph.nodes),
                    len(graph.edges),
                    len(receipt.candidates),
                    len(receipt.optional_node_ids),
                    len(receipt.omitted_node_ids),
                )
            ),
        }

    def _core(
        self,
        request: Any,
        closure_nodes: tuple[Any, ...],
        graph: Any,
        *,
        acceptance: Any,
        validation: Any,
        failure_behavior: Any,
    ) -> Mapping[str, Any]:
        from .semantic_dependency_graph import (
            SemanticEdgeKind,
            SemanticNodeKind,
        )

        kinds = SemanticNodeKind
        legal = self._reference_ids(
            closure_nodes,
            lambda node: node.kind.value.startswith("legal_"),
        )
        security = self._reference_ids(
            closure_nodes,
            lambda node: node.kind.value.startswith("security_"),
        )
        assumptions = self._reference_ids(
            closure_nodes,
            lambda node: node.kind
            in {
                kinds.ASSUMPTION,
                kinds.PREMISE,
                kinds.INTENT_ASSUMPTION,
                kinds.LEGAL_ASSUMPTION,
                kinds.SECURITY_THREAT_ASSUMPTION,
            },
        )
        obligations = self._reference_ids(
            closure_nodes,
            lambda node: node.kind
            in {
                kinds.OBLIGATION,
                kinds.INTENT_OBLIGATION,
                kinds.LEGAL_OBLIGATION,
                kinds.LEGAL_PROOF_OBLIGATION,
                kinds.SECURITY_OBLIGATION,
            },
        )
        proof_ids = self._reference_ids(
            closure_nodes, lambda node: node.kind is kinds.PROOF
        )
        monitor_ids = self._reference_ids(
            closure_nodes, lambda node: node.kind is kinds.MONITOR
        )
        validation_ids = self._reference_ids(
            closure_nodes,
            lambda node: node.kind
            in {kinds.VALIDATION, kinds.INTENT_VERIFICATION},
        )
        intent_ids = self._reference_ids(
            closure_nodes,
            lambda node: node.kind.value.startswith("intent_")
            or node.kind is kinds.ACTION,
        )
        program_ids = self._reference_ids(
            closure_nodes,
            lambda node: node.kind
            in {
                kinds.WORKTREE,
                kinds.REPOSITORY_TREE,
                kinds.FILE,
                kinds.AST,
                kinds.SYMBOL,
                kinds.INTERFACE,
                kinds.CALL,
                kinds.DATA_FLOW,
                kinds.PROGRAM,
                kinds.ENVIRONMENT,
                kinds.TOOLCHAIN,
                kinds.TOOL,
                kinds.RESOURCE,
            },
        )
        effect_ids = self._reference_ids(
            closure_nodes,
            lambda node: node.kind
            in {kinds.EFFECT, kinds.INTENT_EFFECT, kinds.INTENT_POSTCONDITION},
        )
        authorization_ids = self._reference_ids(
            closure_nodes,
            lambda node: node.kind
            in {
                kinds.AUTHORIZATION,
                kinds.INTENT_RESULT_AUTHORITY,
                kinds.LEGAL_RESULT_AUTHORITY,
                kinds.SECURITY_RESULT_AUTHORITY,
            },
        )
        failure_ids = self._reference_ids(
            closure_nodes,
            lambda node: node.kind
            in {kinds.INTENT_FAILURE, kinds.INTENT_RETRY},
        )
        legal_unknowns = self._reference_ids(
            closure_nodes,
            lambda node: node.kind.value.startswith("legal_")
            and (
                node.kind is kinds.LEGAL_ASSUMPTION
                or self._has_unknown(node.record)
            ),
        )
        security_unknowns = self._reference_ids(
            closure_nodes,
            lambda node: node.kind.value.startswith("security_")
            and (
                node.kind is kinds.SECURITY_THREAT_ASSUMPTION
                or self._has_unknown(node.record)
            ),
        )
        closure_node_ids = {node.node_id for node in closure_nodes}
        denial_edges = tuple(
            edge.edge_id
            for edge in graph.edges
            if edge.kind is SemanticEdgeKind.DENIES
            and edge.mandatory
            and edge.authoritative
            and edge.source in closure_node_ids
            and edge.target in closure_node_ids
        )
        authorization_status = (
            "denied"
            if denial_edges
            else (
                "authorization_bound"
                if request.authority.authorization is not None
                else "authority_declared"
            )
        )
        roots = tuple(root.to_record() for root in request.semantic_roots)
        application_unknowns: list[str] = []
        if not request.jurisdiction:
            application_unknowns.append("jurisdiction")
        if request.effective_at_ms is None:
            application_unknowns.append("effective_at_ms")
        return {
            "decision": request.to_record(),
            "roots": roots,
            "intent_action_contract": {
                "selected_action": request.action.to_record(),
                "capabilities": tuple(
                    item.to_record() for item in request.capabilities
                ),
                "mandatory_reference_ids": intent_ids,
            },
            "legal_constraints": {
                "jurisdiction": request.jurisdiction,
                "effective_at_ms": request.effective_at_ms,
                "applicability_facts": tuple(
                    item.to_record() for item in request.applicability_facts
                ),
                "mandatory_reference_ids": legal,
            },
            "legal_unknowns": {
                "unknown_fields": tuple(application_unknowns),
                "mandatory_reference_ids": legal_unknowns,
            },
            "security_constraints": {
                "principal_id": request.authority.principal_id,
                "requested_authority": (
                    request.authority.requested_authority.value
                ),
                "capability_ids": request.authority.capability_ids,
                "mandatory_reference_ids": security,
            },
            "security_unknowns": {
                "mandatory_reference_ids": security_unknowns,
            },
            "authorization_state": {
                "status": authorization_status,
                "authority": request.authority.to_record(),
                "authorization_reference_ids": authorization_ids,
                "denial_edge_ids": tuple(sorted(denial_edges)),
            },
            "program_scope": {
                "repository_id": request.repository_id,
                "repository_path": request.repository_path,
                "targets": tuple(
                    item.to_record() for item in request.action.targets
                ),
                "mandatory_reference_ids": program_ids,
            },
            "effect_scope": {
                "expected_effects": tuple(
                    item.to_record() for item in request.expected_effects
                ),
                "mandatory_reference_ids": effect_ids,
            },
            "assumptions": {"mandatory_reference_ids": assumptions},
            "obligations": {"mandatory_reference_ids": obligations},
            "proof_state": {
                "mandatory_reference_ids": proof_ids,
                "proof_edge_ids": tuple(
                    sorted(
                        edge.edge_id
                        for edge in graph.edges
                        if edge.kind is SemanticEdgeKind.PROVEN_BY
                        and edge.mandatory
                        and edge.authoritative
                        and edge.source in closure_node_ids
                        and edge.target in closure_node_ids
                    )
                ),
            },
            "monitor_state": {
                "mandatory_reference_ids": monitor_ids,
                "monitor_edge_ids": tuple(
                    sorted(
                        edge.edge_id
                        for edge in graph.edges
                        if edge.kind is SemanticEdgeKind.MONITORED_BY
                        and edge.mandatory
                        and edge.authoritative
                        and edge.source in closure_node_ids
                        and edge.target in closure_node_ids
                    )
                ),
            },
            "validation": {
                "acceptance_id": request.acceptance_id,
                "mandatory_reference_ids": validation_ids,
                "contract": (
                    validation
                    if validation is not None
                    else {
                        "required": True,
                        "failure": "fail_closed",
                    }
                ),
            },
            "acceptance": (
                acceptance
                if acceptance is not None
                else {"acceptance_id": request.acceptance_id}
            ),
            "failure_behavior": {
                "mode": "fail_closed",
                "mandatory_reference_ids": failure_ids,
                "contract": (
                    failure_behavior
                    if failure_behavior is not None
                    else {
                        "on_missing_dependency": "fail_closed",
                        "on_unresolvable_handle": "fail_closed",
                        "on_mandatory_overflow": "split_or_fail_closed",
                        "on_root_mismatch": "fail_closed",
                    }
                ),
            },
        }

    def _measure_payload(
        self,
        *,
        core: Mapping[str, Any],
        references: tuple[Any, ...],
        witness: Any,
        entries: tuple[Any, ...],
        index_metadata: Mapping[str, Any],
        segment_index: int,
        segment_count: int,
        expansion_request: str,
    ) -> int:
        from .decision_context import DECISION_CONTEXT_SCHEMA

        payload = {
            "schema": DECISION_CONTEXT_SCHEMA,
            "contract_version": 1,
            "required_core": core,
            "references": [item.to_record() for item in references],
            "completeness_witness_id": witness.content_id,
            "witness_entries": [item.to_record() for item in entries],
            "index_metadata": index_metadata,
            "segment": {
                "index": segment_index,
                "count": segment_count,
                "expansion_request": expansion_request,
            },
        }
        return self.estimator.estimate(
            canonical_context_json_bytes(payload).decode("utf-8")
        )

    def verify(self, result: Any) -> Any:
        from .decision_context import (
            DecisionContextBindingError,
            DecisionContextCompilation,
            DecisionContextOverflowError,
        )

        if not isinstance(result, DecisionContextCompilation):
            raise DecisionContextBindingError(
                "result must be a DecisionContextCompilation"
            )
        total = 0
        for context in result.contexts:
            measured = self.estimator.estimate(
                canonical_context_json_bytes(
                    context.provider_payload()
                ).decode("utf-8")
            )
            if measured != context.provider_input_tokens:
                raise DecisionContextBindingError(
                    "provider token accounting is not reproducible"
                )
            if measured > context.effective_input_limit:
                raise DecisionContextOverflowError(
                    "remeasured context exceeds its provider input limit"
                )
            total += measured
        if total != result.complete_input_tokens:
            raise DecisionContextBindingError(
                "complete provider input token accounting is forged"
            )
        return result

    verify_compilation = verify

    def _retry_bindings(self, parent: Any, **current: Any) -> dict[str, str]:
        from .decision_context import (
            DecisionContextInvalidatedError,
            decision_context_bindings,
        )

        self.verify(parent)
        bindings = decision_context_bindings(parent)
        for name, value in current.items():
            if value is not None and str(value) != bindings[name]:
                raise DecisionContextInvalidatedError(
                    f"retry parent invalidated by changed {name.replace('_', ' ')}"
                )
        return bindings

    def verify_retry(self, result: DecisionContextRetryResult) -> DecisionContextRetryResult:
        from .decision_context import DecisionContextRetryError

        self.verify(result.parent_compilation)
        self.verify(result.reconstructed_compilation)
        measured = self.estimator.estimate(
            canonical_context_json_bytes(result.transmitted_payload).decode()
        )
        if (
            measured != result.delta_input_tokens
            or result.full_replay_input_tokens
            != result.reconstructed_compilation.complete_input_tokens
        ):
            raise DecisionContextRetryError("retry token accounting is forged")
        return result

    def compile_retry(
        self,
        parent: Any,
        *,
        changed_dependencies: Iterable[Any],
        expanded_evidence: Iterable[Any] = (),
        omission_reasons: Mapping[str, str] | None = None,
        target_compilation: Any | None = None,
        current_request: Any | None = None,
        current_graph: Any | None = None,
        current_retrieval_receipt: Any | None = None,
        artifact_store: ContentAddressedContextStore | None = None,
        parent_context_id: str | None = None,
        current_repository_id: str | None = None,
        current_dirty_worktree_root_id: str | None = None,
        current_semantic_roots_digest: str | None = None,
        current_semantic_graph_root_id: str | None = None,
        current_authority_id: str | None = None,
    ) -> DecisionContextRetryResult:
        from .decision_context import (
            DecisionContextChangeKind,
            DecisionContextChangedDependency,
            DecisionContextInvalidatedError,
            DecisionContextReference,
            DecisionContextRetryCapsule,
            DecisionContextRetryError,
        )

        bindings = self._retry_bindings(
            parent,
            repository_id=current_repository_id,
            dirty_worktree_root_id=current_dirty_worktree_root_id,
            semantic_roots_digest=current_semantic_roots_digest,
            authority_id=current_authority_id,
        )
        supplied = (
            current_request is not None,
            current_graph is not None,
            current_retrieval_receipt is not None,
        )
        if any(supplied) and not all(supplied):
            raise DecisionContextRetryError(
                "current request, graph, and receipt must be supplied together"
            )
        if all(supplied):
            target_compilation = self.compile(
                current_request,
                current_graph,
                current_retrieval_receipt,
                artifact_store=artifact_store,
                acceptance=parent.required_core["acceptance"],
                validation=parent.required_core["validation"].get("contract"),
                failure_behavior=parent.required_core["failure_behavior"].get("contract"),
            )
        if (
            current_semantic_graph_root_id is not None
            and str(current_semantic_graph_root_id)
            != (target_compilation or parent).witness.semantic_graph_root_id
        ):
            raise DecisionContextInvalidatedError(
                "retry parent invalidated by changed semantic graph root"
            )
        changes = tuple(
            sorted(
                (
                    item
                    if isinstance(item, DecisionContextChangedDependency)
                    else DecisionContextChangedDependency.from_dict(item)
                    for item in changed_dependencies
                ),
                key=lambda item: (item.kind.value, item.dependency_id),
            )
        )
        if not changes:
            raise DecisionContextRetryError("retry requires a changed dependency")
        if any(_unsafe_decision_delta(item.payload) for item in changes):
            raise DecisionContextInvalidatedError(
                "retry delta attempts corpus browsing or authority/root escalation"
            )
        expanded = tuple(
            item
            if isinstance(item, DecisionContextReference)
            else DecisionContextReference.from_dict(item)
            for item in expanded_evidence
        )
        structural = tuple(
            item
            for item in changes
            if item.kind
            not in {
                DecisionContextChangeKind.DIAGNOSTICS,
                DecisionContextChangeKind.EXPANDED_EVIDENCE,
            }
        )
        if structural and target_compilation is None:
            raise DecisionContextRetryError(
                "structural retry requires a current verified closure"
            )
        target = target_compilation or parent
        if expanded:
            if target_compilation not in (None, parent):
                raise DecisionContextRetryError(
                    "expansion and structural replacement cannot share one delta"
                )
        self.verify(target)
        if (
            target.stable_core_id != parent.stable_core_id
            or target.required_core != parent.required_core
            or target.witness.semantic_graph_root_id
            != parent.witness.semantic_graph_root_id
            or target.witness.roots_digest != parent.witness.roots_digest
            or target.witness.mandatory_node_ids != parent.witness.mandatory_node_ids
            or target.witness.mandatory_edge_ids != parent.witness.mandatory_edge_ids
            or target.witness.dependency_paths != parent.witness.dependency_paths
        ):
            raise DecisionContextRetryError(
                "retry target changed stable core or closure topology"
            )
        previous, current = _decision_references(parent), _decision_references(target)
        admitted = set(previous) | {item.reference_id for item in previous.values()}
        declared_nodes: set[str] = set()
        for change in structural:
            if change.dependency_id not in admitted:
                raise DecisionContextRetryError(
                    "changed dependency is outside the mandatory closure"
                )
            node = change.dependency_id
            if node not in previous:
                node = next(
                    key
                    for key, item in previous.items()
                    if item.reference_id == change.dependency_id
                )
            declared_nodes.add(node)
            if (
                previous[node].node_content_id != change.previous_content_id
                or current[node].node_content_id != change.current_content_id
            ):
                raise DecisionContextRetryError(
                    "changed dependency does not match current closure"
                )
        actual_nodes = {
            node_id
            for node_id in previous
            if previous[node_id].to_record() != current[node_id].to_record()
        }
        if actual_nodes != declared_nodes:
            raise DecisionContextRetryError(
                "retry target contains undeclared structural changes"
            )
        expansion_changes = {
            item.dependency_id: item
            for item in changes
            if item.kind is DecisionContextChangeKind.EXPANDED_EVIDENCE
        }
        for item in expanded:
            prior = next(
                (
                    reference
                    for reference in previous.values()
                    if reference.reference_id == item.reference_id
                ),
                None,
            )
            change = expansion_changes.get(item.reference_id)
            if (
                prior is None
                or prior.expansion_handle is None
                or item.node_id != prior.node_id
                or item.node_content_id != prior.node_content_id
                or change is None
                or change.previous_content_id
                != prior.expansion_handle.referenced_content_id
                or change.current_content_id != item.node_content_id
            ):
                raise DecisionContextRetryError(
                    "expanded evidence is outside the admitted parent closure"
                )
        if set(expansion_changes) != {item.reference_id for item in expanded}:
            raise DecisionContextRetryError(
                "expanded evidence and dependency changes differ"
            )
        omissions = dict(omission_reasons or {})
        for change in changes:
            if change.omission_reason:
                existing = omissions.get(change.dependency_id)
                if existing not in (None, change.omission_reason):
                    raise DecisionContextRetryError(
                        "retry changes a dependency omission reason"
                    )
                omissions[change.dependency_id] = change.omission_reason
        for context in parent.contexts:
            for reference in context.references:
                if reference.expansion_handle is not None:
                    reason = reference.expansion_handle.metadata.get("omission_reason")
                    if not reason:
                        continue
                    if reference.reference_id in omissions and (
                        omissions[reference.reference_id] != reason
                    ):
                        raise DecisionContextRetryError(
                            "retry changes an inherited omission reason"
                        )
                    omissions[reference.reference_id] = str(reason)
        selected = parent_context_id or parent.context.content_id
        if selected not in parent.context_ids:
            raise DecisionContextRetryError("retry does not bind a parent segment")
        capsule_args = {
            "parent_decision_request_id": bindings["decision_request_id"],
            "parent_context_id": selected,
            "parent_completeness_witness_id": bindings["witness_id"],
            "parent_stable_core_id": bindings["stable_core_id"],
            "parent_closure_id": bindings["closure_id"],
            "repository_id": bindings["repository_id"],
            "dirty_worktree_root_id": bindings["dirty_worktree_root_id"],
            "semantic_graph_root_id": bindings["semantic_graph_root_id"],
            "semantic_roots_digest": bindings["semantic_roots_digest"],
            "authority_id": bindings["authority_id"],
            "changed_dependencies": changes,
            "expanded_evidence": expanded,
            "omission_reasons": omissions,
            "reconstructed_context_tokens": tuple(
                item.provider_input_tokens for item in target.contexts
            ),
            "full_replay_input_tokens": target.complete_input_tokens,
        }
        provisional = type("_RetryWire", (), capsule_args)()
        delta_tokens = self.estimator.estimate(
            canonical_context_json_bytes(_decision_retry_wire(provisional)).decode()
        )
        if delta_tokens >= target.complete_input_tokens:
            raise DecisionContextRetryError(
                "retry delta does not use fewer tokens than full replay"
            )
        capsule = DecisionContextRetryCapsule(
            **capsule_args, delta_input_tokens=delta_tokens
        )
        result = DecisionContextRetryResult(parent, capsule, target, self)
        return self.verify_retry(result)

    compile_decision_context_retry = compile_retry

    def expand_decision_context(
        self,
        parent: Any,
        request: Any,
        resolver: ContentAddressedContextStore,
        *,
        elapsed_latency_ms: int | None = None,
        cancelled: Any = None,
        **current: Any,
    ) -> DecisionContextRetryResult:
        from .decision_context import (
            DecisionContextBindingError,
            DecisionContextChangeKind,
            DecisionContextChangedDependency,
            DecisionContextExpansionError,
            DecisionContextExpansionRequest,
            DecisionContextReference,
            DecisionContextRepresentation,
            decision_context_bindings,
        )
        from .semantic_dependency_graph import SemanticNode

        if not isinstance(request, DecisionContextExpansionRequest):
            raise DecisionContextExpansionError("invalid expansion request")
        if not isinstance(resolver, ContentAddressedContextStore):
            raise DecisionContextExpansionError("invalid expansion resolver")
        if _cancelled(cancelled):
            raise ContextExpansionCancelled("decision expansion cancelled")
        bindings = decision_context_bindings(parent)
        if (
            request.parent_decision_request_id != bindings["decision_request_id"]
            or request.parent_context_id not in parent.context_ids
            or request.parent_completeness_witness_id != bindings["witness_id"]
            or request.authority_id != bindings["authority_id"]
            or request.semantic_graph_root_id != bindings["semantic_graph_root_id"]
        ):
            raise DecisionContextBindingError(
                "expansion request does not bind its exact parent"
            )
        self._retry_bindings(
            parent,
            repository_id=current.get("current_repository_id"),
            dirty_worktree_root_id=current.get("current_dirty_worktree_root_id"),
            semantic_roots_digest=current.get("current_semantic_roots_digest"),
            authority_id=current.get("current_authority_id"),
        )
        admitted = next(
            (
                item
                for item in _decision_references(parent).values()
                if item.reference_id == request.expansion_handle.reference_id
            ),
            None,
        )
        if (
            admitted is None
            or admitted.expansion_handle != request.expansion_handle
            or admitted.reference_id not in parent.witness.expansion_reference_ids
            or request.expansion_handle.repository_id != bindings["repository_id"]
            or request.expansion_handle.tree_id != bindings["dirty_worktree_root_id"]
            or request.expansion_handle.metadata.get("semantic_graph_root_id")
            != bindings["semantic_graph_root_id"]
            or request.unresolved_question
            not in tuple(request.expansion_handle.metadata.get("unresolved_questions", ()))
        ):
            raise DecisionContextExpansionError(
                "question or handle is not admitted by the original closure"
            )
        parent_budget = parent.required_core["decision"]["budget"]
        count_limit = min(
            request.budget.max_expansions,
            parent_budget["max_expansions"],
            parent_budget["max_items"],
            self.effective_budget.max_items,
        )
        token_limit = min(
            request.budget.max_tokens,
            parent_budget["max_input_tokens"],
            self.effective_input_limit,
        )
        artifact_byte_limit = min(
            request.budget.max_bytes,
            parent_budget["max_artifact_bytes"],
            self.effective_budget.max_item_bytes,
        )
        wire_byte_limit = min(
            request.budget.max_bytes,
            parent_budget["max_serialized_bytes"],
            self.effective_budget.max_serialized_bytes,
        )
        usage = self._decision_expansion_usage.get(bindings["witness_id"])
        if usage is None:
            usage = {
                "budget_id": request.budget.content_id,
                "ids": set(),
                "count": 0,
                "tokens": 0,
                "bytes": 0,
                "latency": 0,
            }
        if usage["budget_id"] != request.budget.content_id:
            raise DecisionContextExpansionError(
                "expansion lineage cannot replace its cumulative budget"
            )
        elapsed = (
            request.elapsed_latency_ms
            if elapsed_latency_ms is None
            else _integer(elapsed_latency_ms, "elapsed_latency_ms", minimum=0)
        )
        if (
            request.equivalent_request_id in usage["ids"]
            or request.equivalent_request_id in request.prior_request_ids
            or request.request_id in request.prior_request_ids
            or request.expansion_index != usage["count"] + 1
            or usage["count"] + 1 > count_limit
        ):
            raise DecisionContextExpansionError(
                "repeated equivalent expansion or expansion count budget exceeded"
            )
        if not usage["ids"].issubset(set(request.prior_request_ids)):
            raise DecisionContextExpansionError(
                "expansion request omits prior lineage identities"
            )
        if elapsed < usage["latency"] or elapsed > min(
            request.budget.max_latency_ms, parent_budget["max_latency_ms"]
        ):
            raise DecisionContextExpansionError("expansion latency budget exceeded")
        resolver.resolve(
            request.expansion_handle,
            unresolved_question=request.unresolved_question,
            cancelled=cancelled,
        )
        raw = resolver.get(request.expansion_handle.referenced_content_id)
        if len(raw) > artifact_byte_limit:
            raise DecisionContextExpansionError("expansion byte budget exceeded")
        body_tokens = self.estimator.estimate(raw.decode())
        if body_tokens > token_limit:
            raise DecisionContextExpansionError("expansion token budget exceeded")
        try:
            node = SemanticNode.from_dict(json.loads(raw))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise DecisionContextExpansionError(
                "expanded artifact is not a canonical semantic node"
            ) from exc
        if node.node_id != admitted.node_id or node.content_id != admitted.node_content_id:
            raise DecisionContextExpansionError(
                "expanded artifact does not match mandatory dependency"
            )
        expanded = DecisionContextReference(
            reference_id=admitted.reference_id,
            node_id=admitted.node_id,
            node_kind=admitted.node_kind,
            node_content_id=admitted.node_content_id,
            representation=DecisionContextRepresentation.INLINE,
            summary=admitted.summary,
            body=node.to_dict(),
        )
        change = DecisionContextChangedDependency(
            kind=DecisionContextChangeKind.EXPANDED_EVIDENCE,
            dependency_id=admitted.reference_id,
            previous_content_id=request.expansion_handle.referenced_content_id,
            current_content_id=admitted.node_content_id,
            payload={
                "request_id": request.request_id,
                "unresolved_question": request.unresolved_question,
            },
            omission_reason="body_exceeds_inline_bound",
        )
        result = self.compile_retry(
            parent,
            changed_dependencies=(change,),
            expanded_evidence=(expanded,),
            parent_context_id=request.parent_context_id,
            **current,
        )
        delta_bytes = len(canonical_context_json_bytes(result.transmitted_payload))
        new_tokens = usage["tokens"] + result.delta_input_tokens
        new_bytes = usage["bytes"] + delta_bytes
        if new_tokens > token_limit or new_bytes > wire_byte_limit:
            raise DecisionContextExpansionError("expansion aggregate budget exceeded")
        usage["ids"].add(request.equivalent_request_id)
        usage["count"] += 1
        usage["tokens"] = new_tokens
        usage["bytes"] = new_bytes
        usage["latency"] = elapsed
        self._decision_expansion_usage[bindings["witness_id"]] = usage
        return result

    expand = expand_decision_context

    def compile(
        self,
        request: Any,
        graph: Any,
        retrieval_receipt: Any,
        *,
        artifact_store: ContentAddressedContextStore | None = None,
        acceptance: Any = None,
        validation: Any = None,
        failure_behavior: Any = None,
        overflow_behavior: Any = "split",
    ) -> Any:
        from .decision_context import (
            ContextCompletenessEntry,
            ContextCompletenessWitness,
            DecisionContext,
            DecisionContextBindingError,
            DecisionContextCompilation,
            DecisionContextOverflowBehavior,
            DecisionContextOverflowError,
            DecisionContextRepresentation,
        )
        from .decision_contracts import DecisionRequest
        from .proof_directed_retrieval import ProofDirectedRetrievalReceipt
        from .semantic_dependency_graph import (
            ClosureBounds,
            SemanticDependencyGraph,
        )

        if not isinstance(request, DecisionRequest):
            raise DecisionContextBindingError(
                "request must be a DecisionRequest"
            )
        if not isinstance(graph, SemanticDependencyGraph):
            raise DecisionContextBindingError(
                "graph must be a SemanticDependencyGraph"
            )
        if not isinstance(
            retrieval_receipt, ProofDirectedRetrievalReceipt
        ):
            raise DecisionContextBindingError(
                "retrieval_receipt must be a ProofDirectedRetrievalReceipt"
            )
        try:
            behavior = DecisionContextOverflowBehavior(
                str(getattr(overflow_behavior, "value", overflow_behavior))
            )
        except ValueError as exc:
            raise DecisionContextOverflowError(
                "overflow_behavior must be split, request_expansion, or "
                "fail_closed"
            ) from exc
        if self.require_provider_tokenizer and not self.estimator.provider_aware:
            raise DecisionContextOverflowError(
                "decision context requires a provider tokenizer for complete "
                "input remeasurement"
            )
        effective_limit = min(
            self.effective_input_limit, request.budget.max_input_tokens
        )
        if retrieval_receipt.decision_request_id != request.content_id:
            raise DecisionContextBindingError(
                "retrieval receipt belongs to a different decision"
            )
        if (
            retrieval_receipt.roots.get("decision_request_id")
            != request.content_id
            or retrieval_receipt.roots.get("repository_id")
            != request.repository_id
        ):
            raise DecisionContextBindingError(
                "retrieval receipt decision roots do not match the request"
            )
        if (
            retrieval_receipt.snapshot.graph_id != graph.graph_id
            or retrieval_receipt.snapshot.graph_root_id != graph.root_id
            or retrieval_receipt.roots.get("semantic_graph_id")
            != graph.graph_id
            or retrieval_receipt.roots.get("semantic_graph_root_id")
            != graph.root_id
        ):
            raise DecisionContextBindingError(
                "retrieval receipt belongs to a different semantic graph"
            )
        expected_roots = {
            root.kind.value: {
                "artifact_id": root.artifact.artifact_id,
                "cid_v1": root.artifact.cid_v1,
                "supervisor_digest": root.artifact.supervisor_digest,
                "reference_id": root.artifact.content_id,
            }
            for root in request.semantic_roots
        }
        if retrieval_receipt.roots.get("semantic_roots") != expected_roots:
            raise DecisionContextBindingError(
                "retrieval receipt semantic roots do not match the decision"
            )
        closure = graph.mandatory_closure(
            next(iter(retrieval_receipt.paths.values()))[0],
            bounds=ClosureBounds(
                max_nodes=retrieval_receipt.budgets.max_graph_nodes,
                max_edges=retrieval_receipt.budgets.max_graph_edges,
                max_depth=retrieval_receipt.budgets.max_graph_depth,
                max_annotations=max(1, len(graph.nodes)),
            ),
        )
        if (
            closure.closure_id != retrieval_receipt.closure_id
            or closure.node_ids != retrieval_receipt.closure_node_ids
            or closure.edge_ids != retrieval_receipt.closure_edge_ids
            or dict(closure.paths) != dict(retrieval_receipt.paths)
        ):
            raise DecisionContextBindingError(
                "retrieval receipt does not bind the current mandatory closure"
            )
        node_by_id = {node.node_id: node for node in graph.nodes}
        closure_nodes = tuple(
            node_by_id[node_id] for node_id in closure.node_ids
        )
        references = tuple(
            self._build_reference(
                request=request,
                graph=graph,
                node=node,
                artifact_store=artifact_store,
            )
            for node in closure_nodes
        )
        reference_by_node = {
            reference.node_id: reference for reference in references
        }
        entries = tuple(
            ContextCompletenessEntry(
                node_id=node.node_id,
                node_kind=node.kind.value,
                node_content_id=node.content_id,
                path=closure.paths[node.node_id],
                path_edge_ids=self._path_edge_ids(
                    graph, closure.paths[node.node_id]
                ),
                reference_id=reference_by_node[node.node_id].reference_id,
                reference_content_id=(
                    reference_by_node[node.node_id].resolvable_content_id
                ),
                representation=(
                    reference_by_node[node.node_id].representation
                ),
            )
            for node in closure_nodes
        )
        roots_digest = _canonical_digest(
            tuple(root.to_record() for root in request.semantic_roots)
        )
        witness = ContextCompletenessWitness(
            decision_request_id=request.content_id,
            semantic_graph_root_id=graph.root_id,
            semantic_graph_id=graph.graph_id,
            retrieval_receipt_id=retrieval_receipt.receipt_id,
            closure_id=closure.closure_id,
            mandatory_node_ids=closure.node_ids,
            mandatory_edge_ids=closure.edge_ids,
            entries=entries,
            inline_reference_ids=tuple(
                item.reference_id
                for item in references
                if item.representation
                is DecisionContextRepresentation.INLINE
            ),
            expansion_reference_ids=tuple(
                item.reference_id
                for item in references
                if item.representation
                is DecisionContextRepresentation.EXPANSION
            ),
            roots_digest=roots_digest,
        )
        core = self._core(
            request,
            closure_nodes,
            graph,
            acceptance=acceptance,
            validation=validation,
            failure_behavior=failure_behavior,
        )
        metadata = self._bounded_index_metadata(graph, retrieval_receipt)
        entry_by_node = {entry.node_id: entry for entry in entries}

        one_tokens = self._measure_payload(
            core=core,
            references=references,
            witness=witness,
            entries=entries,
            index_metadata=metadata,
            segment_index=0,
            segment_count=1,
            expansion_request="",
        )
        groups: list[tuple[Any, ...]]
        if (
            one_tokens <= effective_limit
            and len(
                canonical_context_json_bytes(
                    {
                        "core": core,
                        "references": [
                            item.to_record() for item in references
                        ],
                        "entries": [item.to_record() for item in entries],
                    }
                )
            )
            <= request.budget.max_serialized_bytes
        ):
            groups = [references]
        else:
            if behavior is DecisionContextOverflowBehavior.FAIL_CLOSED:
                raise DecisionContextOverflowError(
                    "complete mandatory closure exceeds the provider budget"
                )
            groups = []
            current: list[Any] = []
            conservative_count = max(1, len(references))
            for reference in references:
                candidate = tuple((*current, reference))
                candidate_entries = tuple(
                    entry_by_node[item.node_id] for item in candidate
                )
                name = (
                    "expand mandatory decision context segment "
                    f"{len(groups) + 1}"
                )
                tokens = self._measure_payload(
                    core=core,
                    references=candidate,
                    witness=witness,
                    entries=candidate_entries,
                    index_metadata=metadata,
                    segment_index=len(groups),
                    segment_count=conservative_count,
                    expansion_request=name,
                )
                byte_count = len(
                    canonical_context_json_bytes(
                        {
                            "core": core,
                            "references": [
                                item.to_record() for item in candidate
                            ],
                            "entries": [
                                item.to_record()
                                for item in candidate_entries
                            ],
                        }
                    )
                )
                if (
                    tokens <= effective_limit
                    and byte_count <= request.budget.max_serialized_bytes
                ):
                    current.append(reference)
                    continue
                if not current:
                    raise DecisionContextOverflowError(
                        f"mandatory dependency {reference.node_id!r} and "
                        "the immutable core cannot fit one provider segment"
                    )
                groups.append(tuple(current))
                current = []
                candidate = (reference,)
                candidate_entries = (entry_by_node[reference.node_id],)
                tokens = self._measure_payload(
                    core=core,
                    references=candidate,
                    witness=witness,
                    entries=candidate_entries,
                    index_metadata=metadata,
                    segment_index=len(groups),
                    segment_count=conservative_count,
                    expansion_request=(
                        "expand mandatory decision context segment "
                        f"{len(groups) + 1}"
                    ),
                )
                if tokens > effective_limit:
                    raise DecisionContextOverflowError(
                        f"mandatory dependency {reference.node_id!r} and "
                        "the immutable core cannot fit one provider segment"
                    )
                current.append(reference)
            if current:
                groups.append(tuple(current))
            if len(groups) - 1 > request.budget.max_expansions:
                raise DecisionContextOverflowError(
                    "deterministic mandatory-context split exceeds the "
                    "decision expansion budget"
                )

        contexts: list[Any] = []
        for index, group in enumerate(groups):
            group_entries = tuple(
                entry_by_node[item.node_id] for item in group
            )
            expansion_name = (
                ""
                if len(groups) == 1
                else (
                    "expand mandatory decision context segment "
                    f"{index + 1} of {len(groups)}"
                )
            )
            tokens = self._measure_payload(
                core=core,
                references=group,
                witness=witness,
                entries=group_entries,
                index_metadata=metadata,
                segment_index=index,
                segment_count=len(groups),
                expansion_request=expansion_name,
            )
            if tokens > effective_limit:
                raise DecisionContextOverflowError(
                    "final provider-token remeasurement exceeds the budget"
                )
            contexts.append(
                DecisionContext(
                    required_core=core,
                    references=group,
                    completeness_witness_id=witness.content_id,
                    witness_entries=group_entries,
                    index_metadata=metadata,
                    provider_input_tokens=tokens,
                    effective_input_limit=effective_limit,
                    segment_index=index,
                    segment_count=len(groups),
                    expansion_request=expansion_name,
                )
            )
            serialized_bytes = len(
                canonical_context_json_bytes(
                    contexts[-1].provider_payload()
                )
            )
            serialized_limit = min(
                request.budget.max_serialized_bytes,
                self.effective_budget.max_serialized_bytes,
            )
            if serialized_bytes > serialized_limit:
                raise DecisionContextOverflowError(
                    "final mandatory context exceeds the serialized-byte budget"
                )
        result = DecisionContextCompilation(
            contexts=tuple(contexts),
            witness=witness,
            complete_input_tokens=sum(
                item.provider_input_tokens for item in contexts
            ),
            provider_tokenizer=self.estimator.name,
            overflow_behavior=behavior,
            required_nodes_participated_in_value_selection=False,
            verifier=self,
        )
        return self.verify(result)


def compile_decision_context(
    budget: ContextBudget,
    request: Any,
    graph: Any,
    retrieval_receipt: Any,
    **kwargs: Any,
) -> Any:
    """Convenience wrapper for a complete generation-3 decision context."""

    compiler_options = {
        key: kwargs.pop(key)
        for key in tuple(kwargs)
        if key
        in {
            "tokenizer",
            "estimator",
            "provider_context_window",
            "provider_max_input_tokens",
            "reserved_output_tokens",
            "reserved_tool_tokens",
            "require_provider_tokenizer",
            "max_inline_node_bytes",
            "max_inline_bytes",
        }
    }
    return DecisionContextCompiler(
        budget, **compiler_options
    ).compile(request, graph, retrieval_receipt, **kwargs)


def compile_decision_context_retry(
    compiler: DecisionContextCompiler, parent: Any, **kwargs: Any
) -> DecisionContextRetryResult:
    """Compile a compact retry against an exact generation-3 parent."""

    if not isinstance(compiler, DecisionContextCompiler):
        raise ContextDeltaError("compiler must be a DecisionContextCompiler")
    return compiler.compile_retry(parent, **kwargs)


def expand_decision_context(
    compiler: DecisionContextCompiler,
    parent: Any,
    request: Any,
    resolver: ContentAddressedContextStore,
    **kwargs: Any,
) -> DecisionContextRetryResult:
    """Resolve one question-bound handle admitted by the parent witness."""

    if not isinstance(compiler, DecisionContextCompiler):
        raise ContextExpansionError("compiler must be a DecisionContextCompiler")
    return compiler.expand_decision_context(parent, request, resolver, **kwargs)




def compile_code_proof_context_capsule(*args: Any, **kwargs: Any) -> Any:
    """CBP-060: obligation-first context capsule for code-proof agents.

    Delegates to :mod:`code_proof_context` so implementation agents receive
    open obligations, counterexamples, and digest-only satisfied handles
    without bulk source by default.
    """

    from .code_proof_context import (
        compile_code_proof_context_capsule as _compile_code_proof_context_capsule,
    )

    return _compile_code_proof_context_capsule(*args, **kwargs)



def compile_code_proof_context_delta(*args: Any, **kwargs: Any) -> Any:
    """CBP-070: proof_delta-only retry context bound to a parent capsule."""

    from .code_proof_context import (
        compile_code_proof_context_delta as _compile_code_proof_context_delta,
    )

    return _compile_code_proof_context_delta(*args, **kwargs)


def compile_context_capsule(
    budget: ContextBudget,
    **kwargs: Any,
) -> ContextCompileResult:
    """Convenience wrapper around :class:`ContextCompiler`."""

    compiler_options = {
        key: kwargs.pop(key)
        for key in tuple(kwargs)
        if key
        in {
            "tokenizer",
            "estimator",
            "provider_context_window",
            "provider_max_input_tokens",
            "reserved_output_tokens",
            "reserved_tool_tokens",
            "value_policy",
        }
    }
    return ContextCompiler(budget, **compiler_options).compile(**kwargs)


def compile_prefix_context(
    budget: ContextBudget,
    **kwargs: Any,
) -> PrefixContextResult:
    """Convenience wrapper for a prefix-stable stage input."""

    compiler_options = {
        key: kwargs.pop(key)
        for key in tuple(kwargs)
        if key
        in {
            "tokenizer",
            "estimator",
            "provider_context_window",
            "provider_max_input_tokens",
            "reserved_output_tokens",
            "reserved_tool_tokens",
            "value_policy",
        }
    }
    return ContextCompiler(
        budget, **compiler_options
    ).compile_prefix_context(**kwargs)


compile_prefix_context_capsule = compile_prefix_context


def compile_context_delta(
    budget: ContextBudget,
    parent: ContextCapsule,
    **kwargs: Any,
) -> ContextDeltaResult:
    """Convenience wrapper for a parent-bound retry delta."""

    compiler_options = {
        key: kwargs.pop(key)
        for key in tuple(kwargs)
        if key
        in {
            "tokenizer",
            "estimator",
            "provider_context_window",
            "provider_max_input_tokens",
            "reserved_output_tokens",
            "reserved_tool_tokens",
            "value_policy",
        }
    }
    return ContextCompiler(budget, **compiler_options).compile_delta(
        parent, **kwargs
    )


def expand_context(
    compiler: ContextCompiler,
    parent: ContextCapsule,
    references: Iterable[ContextReference | Mapping[str, Any]],
) -> ContextDeltaResult:
    """Request selected expansion handles as a lossless retry delta."""

    selected = _coerce_references(references)
    handles = {
        item.reference_id: item for item in parent.expansion_references
    }
    missing = {
        item.reference_id for item in selected
    }.difference(handles)
    if missing:
        raise MissingContextReferenceError(
            "requested context handle is not present in the parent capsule: "
            + ", ".join(sorted(missing))
        )
    if any(
        handles[item.reference_id].metadata.get(
            "question_bound_expansion", False
        )
        for item in selected
    ):
        raise ContextExpansionError(
            "value-ranked expansion requires a named unresolved question "
            "and content-addressed resolver"
        )
    return compiler.compile_delta(
        parent,
        evidence=tuple(parent.evidence) + selected,
        requested_reference_ids=tuple(item.reference_id for item in selected),
    )


def expand_context_references(
    compiler: ContextCompiler,
    parent: ContextCapsule,
    reference_ids: Iterable[str],
    resolver: ContentAddressedContextStore,
    *,
    repository_id: str | None = None,
    tree_id: str | None = None,
    unresolved_question: str = "",
    cancelled: Any = None,
) -> ContextDeltaResult:
    """Resolve parent handles by content identity and compile their delta."""

    if not isinstance(resolver, ContentAddressedContextStore):
        raise ContextExpansionError(
            "resolver must be a ContentAddressedContextStore"
        )
    return resolver.expand(
        compiler,
        parent,
        reference_ids,
        repository_id=repository_id,
        tree_id=tree_id,
        unresolved_question=unresolved_question,
        cancelled=cancelled,
    )


def expand_context_for_question(
    compiler: ContextCompiler,
    parent: ContextCapsule,
    request: EvidenceExpansionRequest,
    resolver: ContentAddressedContextStore,
    *,
    repository_id: str | None = None,
    tree_id: str | None = None,
    cancelled: Any = None,
) -> ContextDeltaResult:
    """Strict question-bound expansion through parent content-addressed handles."""

    if not isinstance(request, EvidenceExpansionRequest):
        raise ContextExpansionError(
            "request must be an EvidenceExpansionRequest"
        )
    return expand_context_references(
        compiler,
        parent,
        request.reference_ids,
        resolver,
        repository_id=repository_id,
        tree_id=tree_id,
        unresolved_question=request.unresolved_question,
        cancelled=cancelled,
    )


def compile_retry_context(
    compiler: ContextCompiler,
    parent: ContextCapsule,
    *,
    prior_decision_id: str,
    diagnostic_receipt_id: str,
    evidence: Iterable[ContextReference | Mapping[str, Any]],
    failure_evidence_ids: Iterable[str],
    counterexample_evidence_ids: Iterable[str] = (),
    changed_files: Iterable[str] = (),
    changed_symbols: Iterable[str] = (),
    unresolved_requirement_ids: Iterable[str] = (),
    repair_round: int = 1,
    max_repair_rounds: int = 3,
    repository_id: str | None = None,
    tree_id: str | None = None,
    cancelled: Any = None,
) -> RetryContextResult:
    """Compile a semantic repair capsule without replaying the base prompt."""

    if not isinstance(compiler, ContextCompiler):
        raise ContextDeltaError("compiler must be a ContextCompiler")
    if not isinstance(parent, ContextCapsule):
        raise ContextDeltaError("parent must be a ContextCapsule")
    if _cancelled(cancelled):
        raise ContextExpansionCancelled("retry context compilation was cancelled")
    current_repository = _text(
        repository_id or parent.repository_id, "repository_id"
    )
    current_tree = _text(tree_id or parent.tree_id, "tree_id")
    if current_repository != parent.repository_id or current_tree != parent.tree_id:
        raise ChangedTreeContextError(
            "retry parent was invalidated by a changed repository tree"
        )
    references = _coerce_references(evidence)
    failure_ids = _strings(failure_evidence_ids, "failure_evidence_ids")
    counterexample_ids = _strings(
        counterexample_evidence_ids, "counterexample_evidence_ids"
    )
    semantic_ids = set(failure_ids) | set(counterexample_ids)
    candidate_ids = {item.reference_id for item in references}
    if not semantic_ids.issubset(candidate_ids):
        raise ContextDeltaError(
            "retry failure/counterexample IDs must name supplied evidence"
        )
    delta_result = compiler.compile_delta(parent, evidence=references)
    transmitted_ids = {
        item.reference_id for item in delta_result.delta_capsule.evidence
    }
    if not semantic_ids.issubset(transmitted_ids):
        raise ContextDeltaError(
            "retry failure/counterexample evidence must be new or changed"
        )
    if _cancelled(cancelled):
        raise ContextExpansionCancelled("retry context compilation was cancelled")
    capsule = RetryContextCapsule(
        prior_decision_id=prior_decision_id,
        diagnostic_receipt_id=diagnostic_receipt_id,
        repository_id=current_repository,
        tree_id=current_tree,
        delta_capsule=delta_result.delta_capsule,
        failure_evidence_ids=failure_ids,
        counterexample_evidence_ids=counterexample_ids,
        changed_files=tuple(changed_files),
        changed_symbols=tuple(changed_symbols),
        unresolved_requirement_ids=tuple(unresolved_requirement_ids),
        repair_round=repair_round,
        max_repair_rounds=max_repair_rounds,
    )
    return RetryContextResult(capsule, delta_result)


build_context_capsule = compile_context_capsule
build_context_delta = compile_context_delta
build_prefix_context = compile_prefix_context
ContextCompilationResult = ContextCompileResult
ContextRetryResult = ContextDeltaResult
ContextArtifactStore = ContentAddressedContextStore
EvidenceValueReceipt = ValueOfInformationEvidence
ValueOfInformationPolicy = EvidenceValuePolicy
reconstruct_context_capsule = reconstruct_context
render_prefix_stable_context = render_prefix_context


__all__ = [
    "CONTEXT_COMPILATION_RECEIPT_SCHEMA",
    "CONTEXT_COMPILER_VERSION",
    "CONTEXT_DELTA_RECEIPT_SCHEMA",
    "CONTEXT_EVIDENCE_PRODUCERS",
    "CONSERVATIVE_PREFIX_REUSE_BPS",
    "DEFAULT_DIVERSITY_PENALTY_BPS",
    "EVIDENCE_VALUE_FIXTURE_SCHEMA",
    "RETRY_CONTEXT_CAPSULE_SCHEMA",
    "DELTA_RETRY_ACCEPTANCE_CRITERIA",
    "DELTA_RETRY_CONTEXT_EVIDENCE_SCHEMA",
    "DELTA_RETRY_EVIDENCE_ID",
    "DELTA_RETRY_OBJECTIVE_ID",
    "MIN_WARM_PREFIX_REUSE_BPS",
    "MIN_INPUT_TOKEN_REDUCTION_BPS",
    "MIN_RETRY_INPUT_TOKEN_REDUCTION_BPS",
    "PREFIX_CACHE_IDENTITY_SCHEMA",
    "PREFIX_REUSE_ACCEPTANCE_CRITERIA",
    "PREFIX_REUSE_OBJECTIVE_ID",
    "PREFIX_REUSE_RECEIPT_SCHEMA",
    "PREFIX_REUSE_REQUIREMENT_ID",
    "PREFIX_STABLE_CONTEXT_CAPSULE_SCHEMA",
    "REQUIRED_CONTEXT_ACCEPTANCE_CRITERIA",
    "REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID",
    "REQUIRED_CONTEXT_BUDGET_EVIDENCE_SCHEMA",
    "REQUIRED_CONTEXT_OBJECTIVE_ID",
    "VALUE_OF_INFORMATION_EVIDENCE_SCHEMA",
    "VALUE_OF_INFORMATION_ACCEPTANCE_CRITERIA",
    "VALUE_OF_INFORMATION_OBJECTIVE_ID",
    "VALUE_OF_INFORMATION_REQUIREMENT_ID",
    "CalibratedTokenEstimator",
    "ContextCompilationError",
    "ContextCompilationReceipt",
    "ContextCompilationResult",
    "ContextCompileResult",
    "ContextCompiler",
    "DecisionContextCompiler",
    "DecisionContextRetryResult",
    "ContextDeltaError",
    "ContextDeltaReceipt",
    "ContextDeltaResult",
    "ContextRetryResult",
    "PrefixCacheBoundaryError",
    "PrefixCacheDecision",
    "PrefixCacheIdentity",
    "PrefixCacheKind",
    "PrefixContextError",
    "PrefixContextResult",
    "PrefixReuseReceipt",
    "PrefixReuseSource",
    "PrefixStableContextCapsule",
    "ContextExpansionCancelled",
    "ContextExpansionError",
    "ChangedTreeContextError",
    "ContentAddressedContextStore",
    "ContextArtifactStore",
    "DeltaRetryContextEvidence",
    "EvidenceExpansionRequest",
    "EvidenceSelectionDecision",
    "EvidenceValueEstimate",
    "EvidenceValuePairedFixture",
    "EvidenceValuePolicy",
    "EvidenceValueReceipt",
    "ExclusionReason",
    "InclusionReason",
    "RequiredContextBudgetEvidence",
    "RequiredContextOverflowError",
    "MissingContextReferenceError",
    "RetryContextCapsule",
    "RetryContextResult",
    "ValueOfInformationEvidence",
    "ValueOfInformationPolicy",
    "build_text_context_references",
    "build_context_capsule",
    "build_context_delta",
    "build_prefix_context",
    "compile_code_proof_context_capsule",
    "compile_code_proof_context_delta",
    "compile_context_capsule",
    "compile_decision_context",
    "compile_decision_context_retry",
    "compile_context_delta",
    "compile_prefix_context",
    "compile_prefix_context_capsule",
    "compile_retry_context",
    "context_provider_input_payload",
    "expand_context",
    "expand_decision_context",
    "expand_context_for_question",
    "expand_context_references",
    "evaluate_evidence_value_fixtures",
    "render_context_capsule",
    "render_prefix_context",
    "render_prefix_stable_context",
    "render_retry_context",
    "reconstruct_context",
    "reconstruct_context_capsule",
    "reconstruct_decision_context",
]
