"""Deterministic program-world repair operators and typed patch sketches (SAWM-021).

Interface: ``ProgramWorldRepairOperatorRegistry@1``
Evidence: ``sawm/cegis-repair@1``

Extend the current autonomous-repair operator catalogue with a closed SAWM
vocabulary that emits bounded typed patch sketches.  This module does not
create a second patch engine, shell executor, validation gate, or acceptance
authority.  Disk mutation remains with the existing isolated transaction.

Normative constraints:

* Analytical unique repairs run before search or residual model sketches.
* Operators emit bounded typed patch sketches only; they never write, exec,
  or self-admit.
* Candidates must stay inside the admitted write scope.
* Protected, authority, trusted, and test paths cannot be weakened.
* Ambiguous, effectful, unsupported, or non-unique repairs abstain.
* Importing this module performs no I/O and starts no analysis.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_bytes,
    cid_for_structured,
    validate_cid,
)

from ..analysis.program_egraph import (
    ProgramEGraphError,
    SaturationDisposition,
    normalize_program_fragment,
)
from ..proof.formal_verification_contracts import CanonicalContract


PROGRAM_WORLD_REPAIR_OPERATOR_REGISTRY_INTERFACE: Final[str] = (
    "ProgramWorldRepairOperatorRegistry@1"
)
PROGRAM_WORLD_CEGIS_INTERFACE: Final[str] = "ProgramWorldCEGIS@1"
SAWM_CEGIS_REPAIR_EVIDENCE: Final[str] = "sawm/cegis-repair@1"

OPERATOR_DESCRIPTOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-repair-operator@1"
)
OPERATOR_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-repair-operator-registry@1"
)
TYPED_PATCH_SKETCH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-typed-patch-sketch@1"
)
SCOPE_GATE_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-repair-scope-gate@1"
)
OPERATOR_APPLICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-repair-operator-application@1"
)

PRODUCER_ID: Final[str] = "program-world-repair-operators@1"
OPERATOR_REGISTRY_VERSION: Final[str] = "1"

MAX_TEXT_CHARS: Final[int] = 4_096
MAX_SOURCE_BYTES: Final[int] = 1_048_576
MAX_PATH_BYTES: Final[int] = 1_024
MAX_SPAN_BYTES: Final[int] = 65_536
MAX_COLLECTION_ITEMS: Final[int] = 256
MAX_IDENTIFIER_CHARS: Final[int] = 192
ADMITTED_LANGUAGES: Final[frozenset[str]] = frozenset({"python"})

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

DEFAULT_PROTECTED_PATHS: Final[tuple[str, ...]] = (
    ".gitignore",
    "requirements.txt",
    "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
    "docs/architecture/semantic_addressed_world_model.objectives.md",
    "docs/architecture/semantic_addressed_world_model.todo.md",
    "docs/architecture/semantic_addressed_world_model_inventory/repository_baseline.json",
    "docs/architecture/semantic_addressed_world_model_inventory/authority_matrix.json",
    "docs/architecture/semantic_addressed_world_model_inventory/overlap_gap_matrix.json",
    "docs/architecture/semantic_addressed_world_model_inventory/identity_inventory.json",
    "docs/architecture/semantic_addressed_world_model_inventory/interface_inventory.json",
    "docs/architecture/semantic_addressed_world_model_inventory/dependency_graph.json",
    "docs/architecture/semantic_addressed_world_model_inventory/capability_matrix.json",
    "docs/architecture/semantic_addressed_world_model_inventory/rollout_baseline.json",
    "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json",
    "config/semantic_addressed_world_model_dependencies.seal.json",
    "config/semantic_addressed_world_model_native_dependency.authorization.json",
    "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
    "scripts/validate_semantic_addressed_world_model_dependencies.py",
    "scripts/validate_semantic_addressed_world_model_board.py",
    "scripts/materialize_semantic_addressed_world_model_program.py",
    "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
    "ipfs_accelerate_py/agent_implementation_route.py",
    "benchmarks/agent_supervisor/semantic_addressed_world_model/benchmark_freeze.json",
)

DEFAULT_PROTECTED_PREFIXES: Final[tuple[str, ...]] = (
    "docs/architecture/semantic_addressed_world_model_inventory/",
    "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL",
    "docs/architecture/semantic_addressed_world_model.",
    "config/semantic_addressed_world_model",
    "config/agent_supervisor_semantic_addressed_world_model",
    "scripts/validate_semantic_addressed_world_model",
    "scripts/materialize_semantic_addressed_world_model",
    "scripts/ops/agent_supervisor/semantic_addressed_world_model",
)

FORBIDDEN_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "ann_score",
        "cosine",
        "distance",
        "embedding",
        "embeddings",
        "knn",
        "model_cid",
        "rank_score",
        "score",
        "scores",
        "similarity",
        "tokenizer_cid",
        "vector",
        "vector_cid",
        "vectors",
    }
)
_SCOPE_WIDEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "extra_paths",
        "new_dependencies",
        "dependency_paths",
        "write_paths",
        "requested_write_paths",
        "authority_override",
        "policy_override",
        "completion_claim",
        "semantic_change",
        "meaning_change",
        "import_additions",
        "extra_imports",
        "extra_files",
        "grammar_expansion",
        "unprotect_paths",
        "allow_protected",
        "new_transform",
        "new_family",
    }
)
_AUTHORITY_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "write_authority",
        "semantic_authority",
        "proof_authority",
        "completion_authority",
        "grants_write_authority",
        "grants_proof_authority",
        "grants_semantic_authority",
        "mutation_permit",
        "admission",
        "promote_patch",
        "self_promote",
        "create_authority",
        "independently_admitted",
        "authoritative",
        "trusted",
        "verified",
        "admitted",
    }
)
_OBLIGATION_WAIVER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "waive_obligation",
        "waive_obligations",
        "skip_proof",
        "skip_test",
        "skip_reanalysis",
        "ignore_counterexample",
        "force_admit",
        "bypass_gate",
        "weaken_test",
        "xfail_all",
    }
)
_FORBIDDEN_CALL_NAMES: Final[frozenset[str]] = frozenset(
    {"exec", "eval", "compile", "__import__"}
)
_FORBIDDEN_ATTR_PAIRS: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("os", "system"),
        ("os", "popen"),
        ("os", "execv"),
        ("os", "execve"),
        ("os", "execl"),
        ("os", "spawnl"),
        ("os", "spawnv"),
        ("subprocess", "Popen"),
        ("subprocess", "call"),
        ("subprocess", "run"),
        ("subprocess", "check_call"),
        ("subprocess", "check_output"),
        ("pickle", "loads"),
        ("marshal", "loads"),
        ("commands", "getoutput"),
    }
)
_FORBIDDEN_IMPORT_MODULES: Final[frozenset[str]] = frozenset(
    {"subprocess", "multiprocessing", "ctypes", "pickle", "marshal", "commands"}
)
_FORBIDDEN_IMPORT_NAMES: Final[frozenset[str]] = frozenset(
    {"system", "popen", "Popen", "loads", "execv", "execl"}
)
_TEST_SKIP_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "skip",
        "skipIf",
        "skipUnless",
        "xfail",
        "pytest.skip",
        "pytest.xfail",
        "unittest.skip",
        "mark.skip",
        "mark.xfail",
    }
)
_EFFECTFUL_CALLS: Final[frozenset[str]] = frozenset(
    {"open", "print", "input", "write", "append"}
)


class ProgramWorldRepairError(ValueError):
    """Frozen repair-operator inputs cannot produce a trustworthy sketch."""


class ProgramWorldRepairAuthorityError(ProgramWorldRepairError):
    """A candidate attempted to mint authority, widen scope, or weaken gates."""


class ProgramWorldRepairBoundsError(ProgramWorldRepairError):
    """A repair payload exceeded a compactness bound."""


class ProgramWorldRepairStaleError(ProgramWorldRepairAuthorityError):
    """Environment or tree bindings are not current."""


class ProgramWorldRepairOperatorKind(str, Enum):
    """Closed SAWM repair-operator vocabulary.  Not model-extensible."""

    REPLACE_EXACT_SPAN = "replace_exact_span"
    RENAME_EXACT_SYMBOL = "rename_exact_symbol"
    ADD_UNIQUE_ARGUMENT = "add_unique_argument"
    ADD_UNIQUE_IMPORT = "add_unique_import"
    ADD_UNIQUE_REGISTRATION = "add_unique_registration"
    EQUALITY_REWRITE = "equality_rewrite"
    RESTORE_TRACKED_ARTIFACT = "restore_tracked_artifact"


class OperatorApplicationDisposition(str, Enum):
    """Closed outcomes for one operator application."""

    UNIQUE_REPAIR = "unique_repair"
    SKETCHED = "sketched"
    ABSTAINED = "abstained"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    STALE = "stale"


class ScopeGateDisposition(str, Enum):
    """Closed scope/type/effect/proof/test gate outcomes."""

    ADMITTED = "admitted"
    REJECTED = "rejected"
    ABSTAINED = "abstained"


class OperatorReason(str, Enum):
    """Stable machine-readable operator/gate reason codes."""

    UNIQUE_MATCH = "unique_match"
    NO_MATCH = "no_match"
    AMBIGUOUS_MATCH = "ambiguous_match"
    NO_BYTE_CHANGE = "no_byte_change"
    PATH_NOT_IN_SCOPE = "path_not_in_scope"
    PROTECTED_PATH = "protected_path"
    TEST_WEAKENING = "test_weakening"
    AUTHORITY_CLAIM = "authority_claim"
    OBLIGATION_WAIVER = "obligation_waiver"
    SCOPE_WIDENING = "scope_widening"
    ARBITRARY_EXECUTION = "arbitrary_execution"
    EFFECT_VIOLATION = "effect_violation"
    TYPE_GATE_FAILED = "type_gate_failed"
    PROOF_GATE_FAILED = "proof_gate_failed"
    TEST_GATE_FAILED = "test_gate_failed"
    LANGUAGE_UNSUPPORTED = "language_unsupported"
    PARSE_ERROR = "parse_error"
    OPERATOR_NOT_REVIEWED = "operator_not_reviewed"
    EQUALITY_UNPROVED = "equality_unproved"
    ALREADY_PRESENT = "already_present"
    RESIDUAL_QUESTION = "residual_question"
    PROPOSAL_ONLY = "proposal_only"
    STALE_BINDING = "stale_binding"
    SIMILARITY_FORBIDDEN = "similarity_forbidden"


CLOSED_OPERATOR_VOCABULARY: Final[tuple[str, ...]] = tuple(
    item.value for item in ProgramWorldRepairOperatorKind
)


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 24:
        raise ProgramWorldRepairBoundsError("repair payload exceeds depth bound")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise ProgramWorldRepairError("repair payloads reject floating-point values")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        forbidden = set(value) & FORBIDDEN_FIELD_MARKERS
        if forbidden:
            raise ProgramWorldRepairAuthorityError(
                "repair payloads reject non-semantic fields "
                + ", ".join(sorted(forbidden))
            )
        return {
            str(key): _plain(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise ProgramWorldRepairBoundsError("repair payload exceeds collection bound")
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise ProgramWorldRepairError(
        f"repair payload contains unsupported type {type(value).__name__}"
    )


def _text(value: Any, name: str, *, empty: bool = False, limit: int = MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str):
        raise ProgramWorldRepairError(f"{name} must be a string")
    if "\x00" in value:
        raise ProgramWorldRepairError(f"{name} must not contain NUL bytes")
    text = value if name in {"source", "before", "after", "expected_after"} else value.strip()
    if name in {"source", "before", "after", "expected_after"}:
        text = value
    else:
        text = value.strip()
    if not empty and not text:
        raise ProgramWorldRepairError(f"{name} must be a nonempty string")
    encoded = text.encode("utf-8")
    if len(encoded) > limit:
        raise ProgramWorldRepairBoundsError(f"{name} exceeds its byte bound")
    return text


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ProgramWorldRepairError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        raise ProgramWorldRepairError(f"{name} has unsupported value {value!r}") from exc


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise ProgramWorldRepairError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None or value == "":
        return None
    return _cid(value, name)


def _identity_cid(payload: Mapping[str, Any]) -> str:
    return cid_for_structured(_plain(payload))


def _source_cid(source: str) -> str:
    return cid_for_bytes(source.encode("utf-8"))


def _reason_codes(*groups: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            text = str(item or "").strip()
            if text and text not in seen:
                seen.add(text)
                ordered.append(text)
    return tuple(ordered)


def _normalize_path(value: Any, name: str = "path") -> str:
    text = _text(value, name, limit=MAX_PATH_BYTES).replace("\\", "/")
    candidate = PurePosixPath(text)
    if candidate.is_absolute() or ".." in candidate.parts or text.startswith("/"):
        raise ProgramWorldRepairAuthorityError(f"{name} must be a bounded relative path")
    if not text or text in {".", "./"}:
        raise ProgramWorldRepairError(f"{name} is not a file path")
    return candidate.as_posix()


def _paths(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ProgramWorldRepairError(f"{name} must be a sequence of paths")
    if len(values) > MAX_COLLECTION_ITEMS:
        raise ProgramWorldRepairBoundsError(f"{name} exceeds collection bound")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _normalize_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return tuple(ordered)


def _ids(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ProgramWorldRepairError(f"{name} must be a sequence")
    if len(values) > MAX_COLLECTION_ITEMS:
        raise ProgramWorldRepairBoundsError(f"{name} exceeds collection bound")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return tuple(ordered)


def _identifier(value: Any, name: str) -> str:
    text = _text(value, name, limit=MAX_IDENTIFIER_CHARS)
    if not _IDENTIFIER.fullmatch(text):
        raise ProgramWorldRepairError(f"{name} must be a closed identifier")
    return text


def _path_matches(path: str, listed: Sequence[str]) -> bool:
    for item in listed:
        if path == item:
            return True
        prefix = item if item.endswith("/") else f"{item}/"
        if path.startswith(prefix):
            return True
    return False


def _is_test_path(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    return (
        "test" in parts
        or "tests" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _parse_python(source: str) -> ast.AST | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        try:
            return ast.parse(source, mode="eval")
        except SyntaxError:
            return None


def _attr_root(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _attr_root(node.value)
    return None


def _call_qualname(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_qualname(node.value)
        if parent is None:
            return node.attr
        return f"{parent}.{node.attr}"
    return None


def _collect_forbidden_execution(tree: ast.AST) -> tuple[str, ...]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_qualname(node.func)
            if name in _FORBIDDEN_CALL_NAMES or (
                name is not None and name.split(".")[-1] in _FORBIDDEN_CALL_NAMES
            ):
                found.append(name or "call")
            if isinstance(node.func, ast.Attribute):
                root = _attr_root(node.func.value)
                attr = node.func.attr
                if root is not None and (root, attr) in _FORBIDDEN_ATTR_PAIRS:
                    found.append(f"{root}.{attr}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _FORBIDDEN_IMPORT_MODULES:
                    found.append(f"import:{alias.name}")
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in _FORBIDDEN_IMPORT_MODULES:
                found.append(f"import:{node.module}")
            for alias in node.names:
                if alias.name in _FORBIDDEN_IMPORT_NAMES:
                    found.append(f"import:{node.module}.{alias.name}")
    return tuple(sorted(set(found)))


def _count_asserts(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            count += 1
            continue
        if isinstance(node, ast.Call):
            name = _call_qualname(node.func) or ""
            leaf = name.split(".")[-1]
            if leaf.startswith("assert") or name in {"pytest.fail", "self.fail"}:
                count += 1
    return count


def _count_test_functions(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_") or node.name.endswith("_test"):
                count += 1
    return count


def _skip_markers(tree: ast.AST) -> tuple[str, ...]:
    found: list[str] = []
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Call):
            name = _call_qualname(node.func)
        elif isinstance(node, ast.Attribute):
            name = _call_qualname(node)
        if name is None:
            continue
        if name in _TEST_SKIP_MARKERS or name.split(".")[-1] in {"skip", "xfail", "skipIf"}:
            found.append(name)
    return tuple(sorted(set(found)))


def _effectful_calls(tree: ast.AST) -> tuple[str, ...]:
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_qualname(node.func)
        if name is None:
            continue
        leaf = name.split(".")[-1]
        if leaf in _EFFECTFUL_CALLS or name in _EFFECTFUL_CALLS:
            found.append(name)
    return tuple(sorted(set(found)))


def _span_occurrences(source: str, span: str) -> int:
    if not span:
        return 0
    count = 0
    start = 0
    while True:
        index = source.find(span, start)
        if index < 0:
            return count
        count += 1
        start = index + max(len(span), 1)


def _replace_unique_span(source: str, before: str, after: str) -> tuple[str | None, int]:
    count = _span_occurrences(source, before)
    if count != 1:
        return None, count
    return source.replace(before, after, 1), count


def identifier_occurs(source: str, name: str) -> bool:
    if not name:
        return False
    return any(match.group(0) == name for match in _WORD.finditer(source))


def _rename_identifier(source: str, symbol_from: str, symbol_to: str) -> tuple[str, int]:
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        if match.group(0) == symbol_from:
            count += 1
            return symbol_to
        return match.group(0)

    rewritten = _WORD.sub(repl, source)
    return rewritten, count


def _has_import(source: str, module: str, name: str | None = None) -> bool:
    tree = _parse_python(source)
    if tree is None:
        return f"import {module}" in source or f"from {module} import" in source
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module:
                    return True
        if isinstance(node, ast.ImportFrom) and node.module == module:
            if name is None:
                return True
            for alias in node.names:
                if alias.name == name:
                    return True
    return False


def _insert_import(source: str, statement: str) -> str:
    lines = source.splitlines(keepends=True)
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    if insert_at < len(lines) and lines[insert_at].startswith('"""'):
        insert_at += 1
        while insert_at < len(lines) and '"""' not in lines[insert_at - 1 if insert_at else 0]:
            insert_at += 1
    last_import = insert_at
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import = index + 1
    suffix = "" if statement.endswith("\n") else "\n"
    lines.insert(last_import, statement + suffix)
    return "".join(lines)


def _walk_forbidden_claims(value: Any) -> tuple[str, ...]:
    reasons: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                marker = str(key).strip().lower().replace("-", "_")
                if marker in FORBIDDEN_FIELD_MARKERS:
                    reasons.append(OperatorReason.SIMILARITY_FORBIDDEN.value)
                if marker in _AUTHORITY_CLAIM_KEYS and child is True:
                    reasons.append(OperatorReason.AUTHORITY_CLAIM.value)
                if marker in _SCOPE_WIDEN_KEYS and child not in (None, (), [], {}, False, ""):
                    reasons.append(OperatorReason.SCOPE_WIDENING.value)
                if marker in _OBLIGATION_WAIVER_KEYS and child not in (None, (), [], {}, False, ""):
                    reasons.append(OperatorReason.OBLIGATION_WAIVER.value)
                walk(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                walk(child)

    walk(value)
    return _reason_codes(reasons)


def _assert_current_binding(
    environment_binding_cid: str, expected_environment_binding_cid: str | None
) -> None:
    actual = _cid(environment_binding_cid, "environment_binding_cid")
    expected = _optional_cid(
        expected_environment_binding_cid, "expected_environment_binding_cid"
    )
    if expected is not None and expected != actual:
        raise ProgramWorldRepairStaleError("repair roots are not bound to the current environment")


@dataclass(frozen=True)
class ProgramWorldOperatorDescriptor(CanonicalContract):
    """One reviewed, uniqueness-requiring repair operator."""

    SCHEMA: ClassVar[str] = OPERATOR_DESCRIPTOR_SCHEMA

    operator_id: str
    kind: ProgramWorldRepairOperatorKind | str
    uniqueness_required: bool = True
    inverse_kind: str = "restore_exact_before_bytes"
    allowed_effects: tuple[str, ...] = ("pure_local",)
    requires_unique_after: bool = True
    review_ref: str = SAWM_CEGIS_REPAIR_EVIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator_id", _identifier(self.operator_id, "operator_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, ProgramWorldRepairOperatorKind, "kind")
        )
        object.__setattr__(
            self,
            "uniqueness_required",
            _bool(self.uniqueness_required, "uniqueness_required"),
        )
        object.__setattr__(
            self, "inverse_kind", _identifier(self.inverse_kind, "inverse_kind")
        )
        object.__setattr__(self, "allowed_effects", _ids(self.allowed_effects, "allowed_effects"))
        object.__setattr__(
            self,
            "requires_unique_after",
            _bool(self.requires_unique_after, "requires_unique_after"),
        )
        object.__setattr__(self, "review_ref", _text(self.review_ref, "review_ref"))
        if not self.uniqueness_required:
            raise ProgramWorldRepairAuthorityError(
                "SAWM repair operators require uniqueness; non-unique operators are not reviewed"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "kind": self.kind if isinstance(self.kind, str) else self.kind.value,
            "uniqueness_required": True,
            "inverse_kind": self.inverse_kind,
            "allowed_effects": list(self.allowed_effects),
            "requires_unique_after": self.requires_unique_after,
            "review_ref": self.review_ref,
        }

    @property
    def descriptor_cid(self) -> str:
        return self.content_id

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramWorldOperatorDescriptor":
        if not isinstance(payload, Mapping):
            raise ProgramWorldRepairError("operator descriptor must be a mapping")
        return cls(
            operator_id=str(payload.get("operator_id") or ""),
            kind=str(payload.get("kind") or ""),
            uniqueness_required=bool(payload.get("uniqueness_required", True)),
            inverse_kind=str(payload.get("inverse_kind") or "restore_exact_before_bytes"),
            allowed_effects=tuple(payload.get("allowed_effects") or ("pure_local",)),
            requires_unique_after=bool(payload.get("requires_unique_after", True)),
            review_ref=str(payload.get("review_ref") or SAWM_CEGIS_REPAIR_EVIDENCE),
        )


def _default_descriptors() -> tuple[ProgramWorldOperatorDescriptor, ...]:
    return tuple(
        ProgramWorldOperatorDescriptor(
            operator_id=kind.value,
            kind=kind,
        )
        for kind in ProgramWorldRepairOperatorKind
    )


@dataclass(frozen=True)
class ProgramWorldRepairOperatorRegistry(CanonicalContract):
    """Finite reviewed catalogue of SAWM repair operators."""

    SCHEMA: ClassVar[str] = OPERATOR_REGISTRY_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_REPAIR_OPERATOR_REGISTRY_INTERFACE

    descriptors: tuple[ProgramWorldOperatorDescriptor, ...] = field(
        default_factory=_default_descriptors
    )
    review_ref: str = SAWM_CEGIS_REPAIR_EVIDENCE

    def __post_init__(self) -> None:
        parsed = tuple(self.descriptors or ())
        if not parsed:
            raise ProgramWorldRepairError("operator registry must be nonempty")
        if len(parsed) > MAX_COLLECTION_ITEMS:
            raise ProgramWorldRepairBoundsError("operator registry exceeds bound")
        kinds: list[str] = []
        normalized: list[ProgramWorldOperatorDescriptor] = []
        for item in parsed:
            descriptor = (
                item
                if isinstance(item, ProgramWorldOperatorDescriptor)
                else ProgramWorldOperatorDescriptor.from_dict(item)
            )
            kind = descriptor.kind if isinstance(descriptor.kind, str) else descriptor.kind.value
            if kind not in CLOSED_OPERATOR_VOCABULARY:
                raise ProgramWorldRepairAuthorityError("operator kind is not in the closed vocabulary")
            if kind in kinds:
                raise ProgramWorldRepairError("operator kinds must be unique")
            kinds.append(kind)
            normalized.append(descriptor)
        missing = set(CLOSED_OPERATOR_VOCABULARY) - set(kinds)
        if missing:
            raise ProgramWorldRepairError(
                "operator registry must cover the closed vocabulary; missing "
                + ", ".join(sorted(missing))
            )
        object.__setattr__(
            self,
            "descriptors",
            tuple(sorted(normalized, key=lambda item: item.operator_id)),
        )
        object.__setattr__(self, "review_ref", _text(self.review_ref, "review_ref"))

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": PROGRAM_WORLD_REPAIR_OPERATOR_REGISTRY_INTERFACE,
            "version": OPERATOR_REGISTRY_VERSION,
            "review_ref": self.review_ref,
            "closed_vocabulary": list(CLOSED_OPERATOR_VOCABULARY),
            "descriptors": [item.to_dict() for item in self.descriptors],
            "producer_id": PRODUCER_ID,
        }

    @property
    def registry_cid(self) -> str:
        return self.content_id

    def kinds(self) -> tuple[str, ...]:
        return tuple(
            item.kind if isinstance(item.kind, str) else item.kind.value
            for item in self.descriptors
        )

    def get(self, kind: str | ProgramWorldRepairOperatorKind) -> ProgramWorldOperatorDescriptor:
        wanted = kind.value if isinstance(kind, ProgramWorldRepairOperatorKind) else str(kind)
        for item in self.descriptors:
            current = item.kind if isinstance(item.kind, str) else item.kind.value
            if current == wanted or item.operator_id == wanted:
                return item
        raise ProgramWorldRepairError(f"operator {wanted!r} is not reviewed")

    def contains(self, kind: str | ProgramWorldRepairOperatorKind) -> bool:
        try:
            self.get(kind)
            return True
        except ProgramWorldRepairError:
            return False

    @classmethod
    def default(cls) -> "ProgramWorldRepairOperatorRegistry":
        return cls()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProgramWorldRepairOperatorRegistry":
        if not isinstance(payload, Mapping):
            raise ProgramWorldRepairError("registry payload must be a mapping")
        descriptors = payload.get("descriptors") or ()
        return cls(
            descriptors=tuple(
                ProgramWorldOperatorDescriptor.from_dict(item)
                if not isinstance(item, ProgramWorldOperatorDescriptor)
                else item
                for item in descriptors
            ),
            review_ref=str(payload.get("review_ref") or SAWM_CEGIS_REPAIR_EVIDENCE),
        )


def build_default_program_world_operator_registry() -> ProgramWorldRepairOperatorRegistry:
    return ProgramWorldRepairOperatorRegistry.default()


@dataclass(frozen=True)
class ScopeGateDecision(CanonicalContract):
    """Independent scope/type/effect/proof/test admission for one sketch."""

    SCHEMA: ClassVar[str] = SCOPE_GATE_DECISION_SCHEMA

    disposition: ScopeGateDisposition | str
    path: str
    reason_codes: tuple[str, ...] = ()
    in_scope: bool = False
    protected: bool = False
    test_path: bool = False
    proposal_only: bool = True
    grants_write_authority: bool = False
    grants_proof_authority: bool = False
    grants_semantic_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ScopeGateDisposition, "disposition"),
        )
        object.__setattr__(self, "path", _normalize_path(self.path))
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        object.__setattr__(self, "in_scope", _bool(self.in_scope, "in_scope"))
        object.__setattr__(self, "protected", _bool(self.protected, "protected"))
        object.__setattr__(self, "test_path", _bool(self.test_path, "test_path"))
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "grants_proof_authority", False)
        object.__setattr__(self, "grants_semantic_authority", False)
        if self.disposition == ScopeGateDisposition.ADMITTED.value and self.reason_codes:
            raise ProgramWorldRepairError("admitted scope decisions cannot carry rejection reasons")
        if self.disposition == ScopeGateDisposition.ADMITTED.value and not self.in_scope:
            raise ProgramWorldRepairError("admitted sketches must stay in scope")

    def _payload(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition
            if isinstance(self.disposition, str)
            else self.disposition.value,
            "path": self.path,
            "reason_codes": list(self.reason_codes),
            "in_scope": self.in_scope,
            "protected": self.protected,
            "test_path": self.test_path,
            "proposal_only": True,
            "grants_write_authority": False,
            "grants_proof_authority": False,
            "grants_semantic_authority": False,
        }

    @property
    def admitted(self) -> bool:
        return self.disposition == ScopeGateDisposition.ADMITTED.value

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScopeGateDecision":
        if not isinstance(payload, Mapping):
            raise ProgramWorldRepairError("scope gate decision must be a mapping")
        return cls(
            disposition=str(payload.get("disposition") or ""),
            path=str(payload.get("path") or ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            in_scope=bool(payload.get("in_scope", False)),
            protected=bool(payload.get("protected", False)),
            test_path=bool(payload.get("test_path", False)),
        )


class RepairCandidateScopeGate:
    """Fail-closed path/type/effect/proof/test gate for repair sketches."""

    def __init__(
        self,
        *,
        write_scope: Sequence[str],
        protected_paths: Sequence[str] = (),
        required_effects: Sequence[str] = ("pure_local",),
        required_obligations: Sequence[str] = (),
        required_tests: Sequence[str] = (),
    ) -> None:
        scope = _paths(write_scope, "write_scope")
        if not scope:
            raise ProgramWorldRepairAuthorityError("write_scope must be nonempty")
        extra_protected = _paths(protected_paths, "protected_paths")
        self.write_scope = scope
        self.protected_paths = tuple(
            dict.fromkeys((*DEFAULT_PROTECTED_PATHS, *DEFAULT_PROTECTED_PREFIXES, *extra_protected))
        )
        self.required_effects = _ids(required_effects, "required_effects")
        self.required_obligations = _ids(required_obligations, "required_obligations")
        self.required_tests = _ids(required_tests, "required_tests")

    def admit(
        self,
        *,
        path: str,
        before_source: str,
        after_source: str,
        metadata: Mapping[str, Any] | None = None,
        language: str = "python",
        obligations: Sequence[str] = (),
        tests: Sequence[str] = (),
        effects: Sequence[str] = (),
    ) -> ScopeGateDecision:
        normalized = _normalize_path(path)
        in_scope = _path_matches(normalized, self.write_scope)
        protected = _path_matches(normalized, self.protected_paths)
        test_path = _is_test_path(normalized)
        reasons: list[str] = []
        if not in_scope:
            reasons.append(OperatorReason.PATH_NOT_IN_SCOPE.value)
        if protected:
            reasons.append(OperatorReason.PROTECTED_PATH.value)
        meta = dict(metadata or {})
        reasons.extend(_walk_forbidden_claims(meta))
        lang = _text(language, "language")
        if lang not in ADMITTED_LANGUAGES:
            reasons.append(f"{OperatorReason.LANGUAGE_UNSUPPORTED.value}:{lang}")
        before_tree = _parse_python(before_source)
        after_tree = _parse_python(after_source)
        if after_source and after_tree is None and lang == "python":
            reasons.append(OperatorReason.PARSE_ERROR.value)
            reasons.append(OperatorReason.TYPE_GATE_FAILED.value)
        if after_tree is not None:
            before_forbidden = set(_collect_forbidden_execution(before_tree or ast.parse("")))
            after_forbidden = set(_collect_forbidden_execution(after_tree))
            introduced = after_forbidden - before_forbidden
            if introduced:
                reasons.append(OperatorReason.ARBITRARY_EXECUTION.value)
            if "pure_local" in self.required_effects:
                before_effects = set(_effectful_calls(before_tree or ast.parse("")))
                after_effects = set(_effectful_calls(after_tree))
                if after_effects - before_effects:
                    reasons.append(OperatorReason.EFFECT_VIOLATION.value)
            if test_path:
                before_asserts = _count_asserts(before_tree or ast.parse(""))
                after_asserts = _count_asserts(after_tree)
                before_tests = _count_test_functions(before_tree or ast.parse(""))
                after_tests = _count_test_functions(after_tree)
                before_skips = set(_skip_markers(before_tree or ast.parse("")))
                after_skips = set(_skip_markers(after_tree))
                if after_asserts < before_asserts or after_tests < before_tests:
                    reasons.append(OperatorReason.TEST_WEAKENING.value)
                    reasons.append(OperatorReason.TEST_GATE_FAILED.value)
                if after_skips - before_skips:
                    reasons.append(OperatorReason.TEST_WEAKENING.value)
                    reasons.append(OperatorReason.TEST_GATE_FAILED.value)
        preserved_obligations = set(_ids(obligations, "obligations"))
        if self.required_obligations and not set(self.required_obligations).issubset(
            preserved_obligations
        ):
            reasons.append(OperatorReason.PROOF_GATE_FAILED.value)
            reasons.append(OperatorReason.OBLIGATION_WAIVER.value)
        preserved_tests = set(_ids(tests, "tests"))
        if self.required_tests and not set(self.required_tests).issubset(preserved_tests):
            reasons.append(OperatorReason.TEST_GATE_FAILED.value)
            reasons.append(OperatorReason.TEST_WEAKENING.value)
        declared_effects = set(_ids(effects, "effects"))
        if declared_effects and "pure_local" in self.required_effects:
            extra_effects = declared_effects - set(self.required_effects) - {"pure_local"}
            if extra_effects:
                reasons.append(OperatorReason.EFFECT_VIOLATION.value)
        codes = _reason_codes(reasons)
        if codes:
            disposition = (
                ScopeGateDisposition.REJECTED
                if any(
                    code
                    in {
                        OperatorReason.PROTECTED_PATH.value,
                        OperatorReason.PATH_NOT_IN_SCOPE.value,
                        OperatorReason.TEST_WEAKENING.value,
                        OperatorReason.AUTHORITY_CLAIM.value,
                        OperatorReason.ARBITRARY_EXECUTION.value,
                        OperatorReason.OBLIGATION_WAIVER.value,
                        OperatorReason.SCOPE_WIDENING.value,
                    }
                    or code.startswith(OperatorReason.LANGUAGE_UNSUPPORTED.value)
                    for code in codes
                )
                else ScopeGateDisposition.ABSTAINED
            )
            return ScopeGateDecision(
                disposition=disposition,
                path=normalized,
                reason_codes=codes,
                in_scope=in_scope,
                protected=protected,
                test_path=test_path,
            )
        return ScopeGateDecision(
            disposition=ScopeGateDisposition.ADMITTED,
            path=normalized,
            in_scope=True,
            protected=False,
            test_path=test_path,
        )


@dataclass(frozen=True)
class TypedPatchSketch(CanonicalContract):
    """Bounded typed patch sketch.  Never grants write or proof authority."""

    SCHEMA: ClassVar[str] = TYPED_PATCH_SKETCH_SCHEMA

    operator_id: str
    kind: ProgramWorldRepairOperatorKind | str
    path: str
    before: str
    after: str
    before_source: str
    after_source: str
    environment_binding_cid: str
    uniqueness: str = OperatorReason.UNIQUE_MATCH.value
    match_count: int = 1
    language: str = "python"
    inverse_kind: str = "restore_exact_before_bytes"
    obligations: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    effects: tuple[str, ...] = ("pure_local",)
    reason_codes: tuple[str, ...] = ()
    residual_question: str = ""
    proposal_only: bool = True
    grants_write_authority: bool = False
    grants_proof_authority: bool = False
    grants_semantic_authority: bool = False
    independently_admitted: bool = False
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator_id", _identifier(self.operator_id, "operator_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, ProgramWorldRepairOperatorKind, "kind")
        )
        object.__setattr__(self, "path", _normalize_path(self.path))
        object.__setattr__(
            self, "before", _text(self.before, "before", empty=True, limit=MAX_SPAN_BYTES)
        )
        object.__setattr__(
            self, "after", _text(self.after, "after", empty=True, limit=MAX_SPAN_BYTES)
        )
        object.__setattr__(
            self,
            "before_source",
            _text(self.before_source, "source", empty=True, limit=MAX_SOURCE_BYTES),
        )
        object.__setattr__(
            self,
            "after_source",
            _text(self.after_source, "source", empty=True, limit=MAX_SOURCE_BYTES),
        )
        object.__setattr__(
            self,
            "environment_binding_cid",
            _cid(self.environment_binding_cid, "environment_binding_cid"),
        )
        object.__setattr__(self, "uniqueness", _text(self.uniqueness, "uniqueness"))
        if type(self.match_count) is not int or isinstance(self.match_count, bool) or self.match_count < 0:
            raise ProgramWorldRepairError("match_count must be a nonnegative integer")
        object.__setattr__(self, "language", _text(self.language, "language"))
        object.__setattr__(self, "inverse_kind", _identifier(self.inverse_kind, "inverse_kind"))
        object.__setattr__(self, "obligations", _ids(self.obligations, "obligations"))
        object.__setattr__(self, "tests", _ids(self.tests, "tests"))
        object.__setattr__(self, "effects", _ids(self.effects, "effects") or ("pure_local",))
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        object.__setattr__(
            self,
            "residual_question",
            _text(self.residual_question, "residual_question", empty=True),
        )
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "grants_proof_authority", False)
        object.__setattr__(self, "grants_semantic_authority", False)
        object.__setattr__(self, "independently_admitted", False)
        meta = _plain(dict(self.metadata or {}))
        if not isinstance(meta, dict):
            raise ProgramWorldRepairError("metadata must be a mapping")
        claims = _walk_forbidden_claims(meta)
        if claims:
            raise ProgramWorldRepairAuthorityError(
                "patch sketch metadata cannot claim authority or widen scope"
            )
        object.__setattr__(self, "metadata", MappingProxyType(meta))

    def _payload(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "kind": self.kind if isinstance(self.kind, str) else self.kind.value,
            "path": self.path,
            "before": self.before,
            "after": self.after,
            "before_source_cid": self.before_source_cid,
            "after_source_cid": self.after_source_cid,
            "environment_binding_cid": self.environment_binding_cid,
            "uniqueness": self.uniqueness,
            "match_count": self.match_count,
            "language": self.language,
            "inverse_kind": self.inverse_kind,
            "obligations": list(self.obligations),
            "tests": list(self.tests),
            "effects": list(self.effects),
            "reason_codes": list(self.reason_codes),
            "residual_question": self.residual_question,
            "proposal_only": True,
            "grants_write_authority": False,
            "grants_proof_authority": False,
            "grants_semantic_authority": False,
            "independently_admitted": False,
            "metadata": dict(self.metadata),
            "evidence": SAWM_CEGIS_REPAIR_EVIDENCE,
            "producer_id": PRODUCER_ID,
        }

    @property
    def before_source_cid(self) -> str:
        return _source_cid(self.before_source)

    @property
    def after_source_cid(self) -> str:
        return _source_cid(self.after_source)

    @property
    def sketch_cid(self) -> str:
        return self.content_id

    @property
    def fingerprint(self) -> str:
        return _identity_cid(
            {
                "kind": self.kind if isinstance(self.kind, str) else self.kind.value,
                "path": self.path,
                "before": self.before,
                "after": self.after,
                "after_source_cid": self.after_source_cid,
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TypedPatchSketch":
        if not isinstance(payload, Mapping):
            raise ProgramWorldRepairError("typed patch sketch must be a mapping")
        return cls(
            operator_id=str(payload.get("operator_id") or ""),
            kind=str(payload.get("kind") or ""),
            path=str(payload.get("path") or ""),
            before=str(payload.get("before") or ""),
            after=str(payload.get("after") or ""),
            before_source=str(payload.get("before_source") or ""),
            after_source=str(payload.get("after_source") or ""),
            environment_binding_cid=str(payload.get("environment_binding_cid") or ""),
            uniqueness=str(payload.get("uniqueness") or OperatorReason.UNIQUE_MATCH.value),
            match_count=int(payload.get("match_count") or 0),
            language=str(payload.get("language") or "python"),
            inverse_kind=str(payload.get("inverse_kind") or "restore_exact_before_bytes"),
            obligations=tuple(payload.get("obligations") or ()),
            tests=tuple(payload.get("tests") or ()),
            effects=tuple(payload.get("effects") or ("pure_local",)),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            residual_question=str(payload.get("residual_question") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class OperatorApplicationReceipt(CanonicalContract):
    """Receipt for one in-memory operator application."""

    SCHEMA: ClassVar[str] = OPERATOR_APPLICATION_SCHEMA

    disposition: OperatorApplicationDisposition | str
    path: str
    operator_id: str
    kind: ProgramWorldRepairOperatorKind | str
    environment_binding_cid: str
    reason_codes: tuple[str, ...] = ()
    sketch: TypedPatchSketch | None = None
    scope: ScopeGateDecision | None = None
    residual_question: str = ""
    analytical: bool = True
    proposal_only: bool = True
    grants_write_authority: bool = False
    grants_proof_authority: bool = False
    grants_semantic_authority: bool = False
    independently_admitted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, OperatorApplicationDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "path",
            _normalize_path(self.path if self.path else "rejected.py"),
        )
        object.__setattr__(self, "operator_id", _identifier(self.operator_id, "operator_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, ProgramWorldRepairOperatorKind, "kind")
        )
        object.__setattr__(
            self,
            "environment_binding_cid",
            _cid(self.environment_binding_cid, "environment_binding_cid"),
        )
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        if self.sketch is not None and not isinstance(self.sketch, TypedPatchSketch):
            raise ProgramWorldRepairError("sketch must be a TypedPatchSketch")
        if self.scope is not None and not isinstance(self.scope, ScopeGateDecision):
            raise ProgramWorldRepairError("scope must be a ScopeGateDecision")
        object.__setattr__(
            self,
            "residual_question",
            _text(self.residual_question, "residual_question", empty=True),
        )
        object.__setattr__(self, "analytical", _bool(self.analytical, "analytical"))
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "grants_proof_authority", False)
        object.__setattr__(self, "grants_semantic_authority", False)
        object.__setattr__(self, "independently_admitted", False)
        if (
            self.disposition == OperatorApplicationDisposition.UNIQUE_REPAIR.value
            and self.sketch is None
        ):
            raise ProgramWorldRepairError("unique repairs must carry a typed patch sketch")

    def _payload(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition
            if isinstance(self.disposition, str)
            else self.disposition.value,
            "path": self.path,
            "operator_id": self.operator_id,
            "kind": self.kind if isinstance(self.kind, str) else self.kind.value,
            "environment_binding_cid": self.environment_binding_cid,
            "reason_codes": list(self.reason_codes),
            "sketch": None if self.sketch is None else self.sketch.to_dict(),
            "scope": None if self.scope is None else self.scope.to_dict(),
            "residual_question": self.residual_question,
            "analytical": self.analytical,
            "proposal_only": True,
            "grants_write_authority": False,
            "grants_proof_authority": False,
            "grants_semantic_authority": False,
            "independently_admitted": False,
            "evidence": SAWM_CEGIS_REPAIR_EVIDENCE,
            "producer_id": PRODUCER_ID,
        }

    @property
    def unique(self) -> bool:
        return self.disposition == OperatorApplicationDisposition.UNIQUE_REPAIR.value

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperatorApplicationReceipt":
        if not isinstance(payload, Mapping):
            raise ProgramWorldRepairError("operator application receipt must be a mapping")
        sketch_payload = payload.get("sketch")
        scope_payload = payload.get("scope")
        return cls(
            disposition=str(payload.get("disposition") or ""),
            path=str(payload.get("path") or "rejected.py"),
            operator_id=str(payload.get("operator_id") or "replace_exact_span"),
            kind=str(payload.get("kind") or "replace_exact_span"),
            environment_binding_cid=str(payload.get("environment_binding_cid") or ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            sketch=TypedPatchSketch.from_dict(sketch_payload)
            if isinstance(sketch_payload, Mapping)
            else None,
            scope=ScopeGateDecision.from_dict(scope_payload)
            if isinstance(scope_payload, Mapping)
            else None,
            residual_question=str(payload.get("residual_question") or ""),
            analytical=bool(payload.get("analytical", True)),
        )


def _terminal(
    *,
    disposition: OperatorApplicationDisposition,
    path: str,
    operator_id: str,
    kind: str,
    environment_binding_cid: str,
    reason_codes: Sequence[str],
    sketch: TypedPatchSketch | None = None,
    scope: ScopeGateDecision | None = None,
    residual_question: str = "",
    analytical: bool = True,
) -> OperatorApplicationReceipt:
    return OperatorApplicationReceipt(
        disposition=disposition,
        path=path or "rejected.py",
        operator_id=operator_id,
        kind=kind,
        environment_binding_cid=environment_binding_cid,
        reason_codes=tuple(reason_codes),
        sketch=sketch,
        scope=scope,
        residual_question=residual_question,
        analytical=analytical,
    )


def _apply_kind(
    kind: str,
    *,
    source: str,
    before: str,
    after: str,
    symbol_from: str,
    symbol_to: str,
    import_module: str,
    import_name: str,
    argument_name: str,
    argument_value: str,
    registration_anchor: str,
    registration_payload: str,
    environment_binding_cid: str,
) -> tuple[str | None, str, str, int, tuple[str, ...]]:
    if kind == ProgramWorldRepairOperatorKind.REPLACE_EXACT_SPAN.value:
        rewritten, count = _replace_unique_span(source, before, after)
        return rewritten, before, after, count, ()
    if kind == ProgramWorldRepairOperatorKind.RENAME_EXACT_SYMBOL.value:
        rewritten, count = _rename_identifier(source, symbol_from, symbol_to)
        return rewritten, symbol_from, symbol_to, count, ()
    if kind == ProgramWorldRepairOperatorKind.ADD_UNIQUE_IMPORT.value:
        statement = (
            f"from {import_module} import {import_name}"
            if import_name
            else f"import {import_module}"
        )
        if _has_import(source, import_module, import_name or None):
            return None, statement, statement, 0, (OperatorReason.ALREADY_PRESENT.value,)
        rewritten = _insert_import(source, statement)
        return rewritten, "", statement, 1, ()
    if kind == ProgramWorldRepairOperatorKind.ADD_UNIQUE_ARGUMENT.value:
        target = before
        if not target or not argument_name:
            return None, "", "", 0, (OperatorReason.NO_MATCH.value,)
        empty_call = f"{target}()"
        if _span_occurrences(source, empty_call) == 1:
            rewritten, count = _replace_unique_span(
                source, empty_call, f"{target}({argument_name}={argument_value})"
            )
            return rewritten, empty_call, f"{target}({argument_name}={argument_value})", count, ()
        open_call = f"{target}("
        if _span_occurrences(source, open_call) == 1:
            rewritten, count = _replace_unique_span(
                source, open_call, f"{target}({argument_name}={argument_value}, "
            )
            return rewritten, open_call, f"{target}({argument_name}={argument_value}, ", count, ()
        return None, target, argument_name, _span_occurrences(source, open_call), ()
    if kind == ProgramWorldRepairOperatorKind.ADD_UNIQUE_REGISTRATION.value:
        if not registration_anchor or not registration_payload:
            return None, "", "", 0, (OperatorReason.NO_MATCH.value,)
        if registration_payload in source:
            return (
                None,
                registration_anchor,
                registration_payload,
                0,
                (OperatorReason.ALREADY_PRESENT.value,),
            )
        count = _span_occurrences(source, registration_anchor)
        if count != 1:
            return None, registration_anchor, registration_payload, count, ()
        rewritten = source.replace(
            registration_anchor, f"{registration_anchor}\n{registration_payload}", 1
        )
        return rewritten, registration_anchor, registration_payload, 1, ()
    if kind == ProgramWorldRepairOperatorKind.EQUALITY_REWRITE.value:
        fragment = before or source.strip()
        try:
            plan = normalize_program_fragment(
                fragment,
                environment_binding_cid=environment_binding_cid,
            )
        except ProgramEGraphError:
            return (
                None,
                fragment,
                "",
                0,
                (OperatorReason.EQUALITY_UNPROVED.value,),
            )
        disposition = (
            plan.disposition.value
            if isinstance(plan.disposition, SaturationDisposition)
            else str(plan.disposition)
        )
        if disposition not in {
            SaturationDisposition.NORMALIZED.value,
            SaturationDisposition.EQUIVALENT.value,
            SaturationDisposition.SATURATED.value,
        } or not plan.normal_form:
            return (
                None,
                fragment,
                "",
                0,
                (OperatorReason.EQUALITY_UNPROVED.value, *plan.reason_codes),
            )
        if plan.normal_form == fragment:
            return None, fragment, plan.normal_form, 0, (OperatorReason.NO_BYTE_CHANGE.value,)
        rewritten, count = _replace_unique_span(source, fragment, plan.normal_form)
        if rewritten is None and source.strip() == fragment:
            rewritten, count = plan.normal_form, 1
        return rewritten, fragment, plan.normal_form, count, ()
    if kind == ProgramWorldRepairOperatorKind.RESTORE_TRACKED_ARTIFACT.value:
        if not after:
            return None, source, after, 0, (OperatorReason.NO_MATCH.value,)
        if source == after:
            return None, source, after, 0, (OperatorReason.NO_BYTE_CHANGE.value,)
        return after, source, after, 1, ()
    return None, before, after, 0, (OperatorReason.OPERATOR_NOT_REVIEWED.value,)


def apply_program_world_repair_operator(
    *,
    operator: str | ProgramWorldRepairOperatorKind,
    source: str,
    path: str,
    write_scope: Sequence[str],
    environment_binding_cid: str,
    expected_environment_binding_cid: str | None = None,
    protected_paths: Sequence[str] = (),
    before: str = "",
    after: str = "",
    symbol_from: str = "",
    symbol_to: str = "",
    import_module: str = "",
    import_name: str = "",
    argument_name: str = "",
    argument_value: str = "",
    registration_anchor: str = "",
    registration_payload: str = "",
    obligations: Sequence[str] = (),
    tests: Sequence[str] = (),
    effects: Sequence[str] = ("pure_local",),
    language: str = "python",
    metadata: Mapping[str, Any] | None = None,
    registry: ProgramWorldRepairOperatorRegistry | None = None,
    analytical: bool = True,
) -> OperatorApplicationReceipt:
    """Apply one closed operator in memory and emit a typed patch sketch."""

    _assert_current_binding(environment_binding_cid, expected_environment_binding_cid)
    catalogue = registry or build_default_program_world_operator_registry()
    kind_text = (
        operator.value
        if isinstance(operator, ProgramWorldRepairOperatorKind)
        else str(operator).strip()
    )
    path_text = _normalize_path(path)
    source_text = _text(source, "source", empty=True, limit=MAX_SOURCE_BYTES)
    lang = _text(language, "language")
    if lang not in ADMITTED_LANGUAGES:
        return _terminal(
            disposition=OperatorApplicationDisposition.UNSUPPORTED,
            path=path_text,
            operator_id=kind_text or ProgramWorldRepairOperatorKind.REPLACE_EXACT_SPAN.value,
            kind=ProgramWorldRepairOperatorKind.REPLACE_EXACT_SPAN.value,
            environment_binding_cid=environment_binding_cid,
            reason_codes=(f"{OperatorReason.LANGUAGE_UNSUPPORTED.value}:{lang}",),
            residual_question=f"language {lang} is unsupported for deterministic repair",
            analytical=analytical,
        )
    if not catalogue.contains(kind_text):
        return _terminal(
            disposition=OperatorApplicationDisposition.REJECTED,
            path=path_text,
            operator_id="replace_exact_span",
            kind=ProgramWorldRepairOperatorKind.REPLACE_EXACT_SPAN.value,
            environment_binding_cid=environment_binding_cid,
            reason_codes=(OperatorReason.OPERATOR_NOT_REVIEWED.value,),
            residual_question="operator is outside the closed SAWM vocabulary",
            analytical=analytical,
        )
    descriptor = catalogue.get(kind_text)
    operator_id = descriptor.operator_id
    kind_value = descriptor.kind if isinstance(descriptor.kind, str) else descriptor.kind.value
    gate = RepairCandidateScopeGate(
        write_scope=write_scope,
        protected_paths=protected_paths,
        required_effects=effects,
        required_obligations=obligations,
        required_tests=tests,
    )
    rewritten, span_before, span_after, match_count, extra_reasons = _apply_kind(
        kind_value,
        source=source_text,
        before=_text(before, "before", empty=True, limit=MAX_SPAN_BYTES),
        after=_text(after, "after", empty=True, limit=MAX_SPAN_BYTES),
        symbol_from=_text(symbol_from, "symbol_from", empty=True, limit=MAX_IDENTIFIER_CHARS),
        symbol_to=_text(symbol_to, "symbol_to", empty=True, limit=MAX_IDENTIFIER_CHARS),
        import_module=_text(import_module, "import_module", empty=True, limit=MAX_IDENTIFIER_CHARS),
        import_name=_text(import_name, "import_name", empty=True, limit=MAX_IDENTIFIER_CHARS),
        argument_name=_text(argument_name, "argument_name", empty=True, limit=MAX_IDENTIFIER_CHARS),
        argument_value=_text(argument_value, "argument_value", empty=True, limit=MAX_TEXT_CHARS),
        registration_anchor=_text(
            registration_anchor, "registration_anchor", empty=True, limit=MAX_SPAN_BYTES
        ),
        registration_payload=_text(
            registration_payload, "registration_payload", empty=True, limit=MAX_SPAN_BYTES
        ),
        environment_binding_cid=environment_binding_cid,
    )
    if match_count > 1:
        return _terminal(
            disposition=OperatorApplicationDisposition.AMBIGUOUS,
            path=path_text,
            operator_id=operator_id,
            kind=kind_value,
            environment_binding_cid=environment_binding_cid,
            reason_codes=(OperatorReason.AMBIGUOUS_MATCH.value,),
            residual_question="operator matched multiple sites; unique repair required",
            analytical=analytical,
        )
    if match_count == 0 or rewritten is None:
        reasons = _reason_codes(
            extra_reasons
            or (
                (OperatorReason.ALREADY_PRESENT.value,)
                if OperatorReason.ALREADY_PRESENT.value in extra_reasons
                else (OperatorReason.NO_MATCH.value,)
            )
        )
        residual = "repair is not uniquely determined by the closed operator"
        disposition = (
            OperatorApplicationDisposition.UNSUPPORTED
            if OperatorReason.EQUALITY_UNPROVED.value in reasons
            else OperatorApplicationDisposition.ABSTAINED
        )
        return _terminal(
            disposition=disposition,
            path=path_text,
            operator_id=operator_id,
            kind=kind_value,
            environment_binding_cid=environment_binding_cid,
            reason_codes=reasons or (OperatorReason.NO_MATCH.value,),
            residual_question=residual,
            analytical=analytical,
        )
    if rewritten == source_text:
        return _terminal(
            disposition=OperatorApplicationDisposition.ABSTAINED,
            path=path_text,
            operator_id=operator_id,
            kind=kind_value,
            environment_binding_cid=environment_binding_cid,
            reason_codes=(OperatorReason.NO_BYTE_CHANGE.value,),
            residual_question="operator produced no byte change",
            analytical=analytical,
        )
    scope = gate.admit(
        path=path_text,
        before_source=source_text,
        after_source=rewritten,
        metadata=metadata,
        language=lang,
        obligations=obligations,
        tests=tests,
        effects=effects,
    )
    if not scope.admitted:
        rejected = OperatorReason.PROTECTED_PATH.value in scope.reason_codes or (
            OperatorReason.PATH_NOT_IN_SCOPE.value in scope.reason_codes
            or OperatorReason.TEST_WEAKENING.value in scope.reason_codes
            or OperatorReason.ARBITRARY_EXECUTION.value in scope.reason_codes
            or OperatorReason.AUTHORITY_CLAIM.value in scope.reason_codes
        )
        return _terminal(
            disposition=(
                OperatorApplicationDisposition.REJECTED
                if rejected
                else OperatorApplicationDisposition.ABSTAINED
            ),
            path=path_text,
            operator_id=operator_id,
            kind=kind_value,
            environment_binding_cid=environment_binding_cid,
            reason_codes=scope.reason_codes,
            scope=scope,
            residual_question="candidate failed scope/type/effect/proof/test gates",
            analytical=analytical,
        )
    sketch = TypedPatchSketch(
        operator_id=operator_id,
        kind=kind_value,
        path=path_text,
        before=span_before,
        after=span_after,
        before_source=source_text,
        after_source=rewritten,
        environment_binding_cid=environment_binding_cid,
        uniqueness=OperatorReason.UNIQUE_MATCH.value,
        match_count=match_count,
        language=lang,
        inverse_kind=descriptor.inverse_kind,
        obligations=obligations,
        tests=tests,
        effects=effects,
        reason_codes=(OperatorReason.UNIQUE_MATCH.value, OperatorReason.PROPOSAL_ONLY.value),
        metadata=metadata or {},
    )
    return OperatorApplicationReceipt(
        disposition=OperatorApplicationDisposition.UNIQUE_REPAIR,
        path=path_text,
        operator_id=operator_id,
        kind=kind_value,
        environment_binding_cid=environment_binding_cid,
        reason_codes=sketch.reason_codes,
        sketch=sketch,
        scope=scope,
        analytical=analytical,
    )


__all__ = [
    "CLOSED_OPERATOR_VOCABULARY",
    "DEFAULT_PROTECTED_PATHS",
    "OPERATOR_APPLICATION_SCHEMA",
    "PROGRAM_WORLD_CEGIS_INTERFACE",
    "PROGRAM_WORLD_REPAIR_OPERATOR_REGISTRY_INTERFACE",
    "SAWM_CEGIS_REPAIR_EVIDENCE",
    "OperatorApplicationDisposition",
    "OperatorApplicationReceipt",
    "OperatorReason",
    "ProgramWorldOperatorDescriptor",
    "ProgramWorldRepairAuthorityError",
    "ProgramWorldRepairBoundsError",
    "ProgramWorldRepairError",
    "ProgramWorldRepairOperatorKind",
    "ProgramWorldRepairOperatorRegistry",
    "ProgramWorldRepairStaleError",
    "RepairCandidateScopeGate",
    "ScopeGateDecision",
    "ScopeGateDisposition",
    "TypedPatchSketch",
    "apply_program_world_repair_operator",
    "build_default_program_world_operator_registry",
    "identifier_occurs",
]
