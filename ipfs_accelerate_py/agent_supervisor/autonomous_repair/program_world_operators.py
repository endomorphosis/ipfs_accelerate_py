"""Closed program-world repair operators (SAWM-021).

Interface: ``ProgramWorldRepairOperatorRegistry@1``
Evidence: ``sawm/cegis-repair@1``

Deterministic, uniqueness-checked operators that emit *typed patch sketches*
only.  This module does not create a second patch engine, shell executor,
validation gate, or acceptance authority.  Production mutation remains the
current autonomous-repair transaction.

Normative constraints:

* The operator vocabulary is closed and not model-expandable.
* A repair is unique only when a single closed operator produces one sketch.
* Candidates stay inside admitted scope and cannot target protected paths.
* Sketches cannot weaken tests, authority claims, or trusted paths.
* Arbitrary execution, deserialization, and grammar expansion are rejected.
* Unsupported repairs abstain or become an explicit residual question.
* Importing this module performs no I/O and starts no analysis.
"""

from __future__ import annotations

import ast
import hashlib
import io
import re
import tokenize
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import CanonicalContract, content_identity


PROGRAM_WORLD_REPAIR_OPERATOR_REGISTRY_INTERFACE: Final[str] = (
    "ProgramWorldRepairOperatorRegistry@1"
)
PROGRAM_WORLD_REPAIR_OPERATOR_INTERFACE: Final[str] = (
    "apply_program_world_repair_operator@1"
)
SAWM_CEGIS_REPAIR_EVIDENCE: Final[str] = "sawm/cegis-repair@1"

PROGRAM_WORLD_OPERATOR_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-repair-operator-spec@1"
)
PROGRAM_WORLD_PATCH_SKETCH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-patch-sketch@1"
)
PROGRAM_WORLD_OPERATOR_APPLICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-operator-application@1"
)
PROGRAM_WORLD_OPERATOR_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-world-repair-operator-registry@1"
)

PRODUCER_ID: Final[str] = "program-world-repair-operators@1"
OPERATOR_VERSION: Final[str] = "1"
TASK_ID: Final[str] = "SAWM-021"

MAX_TEXT_CHARS: Final[int] = 4_096
MAX_SPAN_BYTES: Final[int] = 65_536
MAX_SOURCE_BYTES: Final[int] = 1_048_576
MAX_PATH_BYTES: Final[int] = 1_024
MAX_COLLECTION_ITEMS: Final[int] = 256
MAX_METADATA_KEYS: Final[int] = 64

# Existing PDR/DCR operator kinds this vocabulary extends (no second engine).
EXISTING_OPERATOR_KIND_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "replace_exact_bytes": "replace_exact_bytes",
        "rename_exact_symbol": "exact_rename",
        "replace_unique_registration": "add_registration",
        "add_import": "add_import",
        "add_argument": "add_argument",
        "thread_argument": "thread_argument",
        "equality_rewrite": "equality_rewrite",
        "restore_tracked_artifact": "restore_tracked_artifact",
        "finite_adapter": "finite_adapter",
    }
)

DEFAULT_PROTECTED_PATHS: Final[tuple[str, ...]] = (
    ".gitignore",
    "requirements.txt",
    "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
    "docs/architecture/semantic_addressed_world_model.objectives.md",
    "docs/architecture/semantic_addressed_world_model.todo.md",
    "docs/architecture/semantic_addressed_world_model_inventory/",
    "config/semantic_addressed_world_model_dependencies.seal.json",
    "config/semantic_addressed_world_model_native_dependency.authorization.json",
    "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
    "scripts/validate_semantic_addressed_world_model_dependencies.py",
    "scripts/validate_semantic_addressed_world_model_board.py",
    "scripts/materialize_semantic_addressed_world_model_program.py",
    "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
    "benchmarks/agent_supervisor/semantic_addressed_world_model/benchmark_freeze.json",
)

_SCOPE_WIDEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "authority_override",
        "completion_claim",
        "dependency_paths",
        "extra_files",
        "extra_imports",
        "extra_paths",
        "grammar_expansion",
        "import_additions",
        "meaning_change",
        "new_dependencies",
        "new_family",
        "new_transform",
        "policy_override",
        "requested_write_paths",
        "semantic_change",
        "write_paths",
    }
)
_AUTHORITY_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "admitted",
        "admission",
        "authoritative",
        "complete",
        "completion_authority",
        "create_authority",
        "grants_completion_authority",
        "grants_proof_authority",
        "grants_semantic_authority",
        "grants_write_authority",
        "kernel_checked",
        "mutation_permit",
        "proof_authority",
        "proof_success",
        "promote_patch",
        "self_approve",
        "self_promote",
        "semantic_authority",
        "trusted",
        "verified",
        "write_authority",
    }
)
_OBLIGATION_WAIVER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "bypass_gate",
        "force_admit",
        "ignore_counterexample",
        "skip_proof",
        "skip_reanalysis",
        "skip_test",
        "waive_obligation",
        "waive_obligations",
    }
)
_FORBIDDEN_CALLS: Final[frozenset[str]] = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "globals",
        "input",
        "locals",
        "setattr",
    }
)
_FORBIDDEN_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "exec",
        "execv",
        "execve",
        "loads",
        "popen",
        "system",
        "wait",
    }
)
_FORBIDDEN_MODULES: Final[frozenset[str]] = frozenset(
    {
        "ctypes",
        "http",
        "multiprocessing",
        "pickle",
        "requests",
        "socket",
        "subprocess",
        "urllib",
    }
)
_SKIP_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "pytest.mark.skip",
        "pytest.skip",
        "pytest.xfail",
        "unittest.skip",
        "mark.skip",
    }
)
_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MODULE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$"
)


class ProgramWorldOperatorError(ValueError):
    """Program-world operator inputs cannot produce a trustworthy sketch."""


class ProgramWorldOperatorAuthorityError(ProgramWorldOperatorError):
    """A sketch attempted to mint authority, widen scope, or execute code."""


class ProgramWorldOperatorBoundsError(ProgramWorldOperatorError):
    """A sketch or payload exceeded a deterministic bound."""


class ProgramWorldOperatorKind(str, Enum):
    """Closed SAWM repair-operator vocabulary.  Models cannot expand it."""

    REPLACE_EXACT_BYTES = "replace_exact_bytes"
    RENAME_EXACT_SYMBOL = "rename_exact_symbol"
    REPLACE_UNIQUE_REGISTRATION = "replace_unique_registration"
    ADD_IMPORT = "add_import"
    ADD_ARGUMENT = "add_argument"
    THREAD_ARGUMENT = "thread_argument"
    EQUALITY_REWRITE = "equality_rewrite"
    RESTORE_TRACKED_ARTIFACT = "restore_tracked_artifact"
    FINITE_ADAPTER = "finite_adapter"


class ProgramWorldDefectFamily(str, Enum):
    """Closed defect families that select analytical operators first."""

    UNIQUE_BYTES = "unique_bytes"
    WRONG_SYMBOL = "wrong_symbol"
    MISSING_IMPORT = "missing_import"
    MISSING_ARGUMENT = "missing_argument"
    EQUALITY_DEBT = "equality_debt"
    MISSING_REGISTRATION = "missing_registration"
    TRACKED_ARTIFACT = "tracked_artifact"
    FINITE_ADAPTER = "finite_adapter"
    UNSUPPORTED = "unsupported"


class ProgramWorldOperatorDisposition(str, Enum):
    """Closed outcomes of one operator application.  Never self-admits."""

    UNIQUE = "unique"
    APPLIED = "applied"
    AMBIGUOUS = "ambiguous"
    ABSTAINED = "abstained"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    OUT_OF_SCOPE = "out_of_scope"


class ProgramWorldSpanKind(str, Enum):
    BYTES = "bytes"
    SYMBOL = "symbol"
    IMPORT = "import"
    ARGUMENT = "argument"
    REGISTRATION = "registration"
    EQUALITY = "equality"
    ARTIFACT = "artifact"
    ADAPTER = "adapter"


class ProgramWorldEffectClass(str, Enum):
    PURE_LOCAL = "pure_local"
    UNKNOWN = "unknown"
    DYNAMIC = "dynamic"
    GENERATED = "generated"
    STATEFUL = "stateful"
    NATIVE = "native"
    PUBLIC_API = "public_api"
    DEPENDENCY_CHANGING = "dependency_changing"


FAMILY_TO_OPERATORS: Final[Mapping[ProgramWorldDefectFamily, tuple[ProgramWorldOperatorKind, ...]]] = (
    MappingProxyType(
        {
            ProgramWorldDefectFamily.UNIQUE_BYTES: (
                ProgramWorldOperatorKind.REPLACE_EXACT_BYTES,
            ),
            ProgramWorldDefectFamily.WRONG_SYMBOL: (
                ProgramWorldOperatorKind.RENAME_EXACT_SYMBOL,
            ),
            ProgramWorldDefectFamily.MISSING_IMPORT: (
                ProgramWorldOperatorKind.ADD_IMPORT,
            ),
            ProgramWorldDefectFamily.MISSING_ARGUMENT: (
                ProgramWorldOperatorKind.ADD_ARGUMENT,
                ProgramWorldOperatorKind.THREAD_ARGUMENT,
            ),
            ProgramWorldDefectFamily.EQUALITY_DEBT: (
                ProgramWorldOperatorKind.EQUALITY_REWRITE,
            ),
            ProgramWorldDefectFamily.MISSING_REGISTRATION: (
                ProgramWorldOperatorKind.REPLACE_UNIQUE_REGISTRATION,
            ),
            ProgramWorldDefectFamily.TRACKED_ARTIFACT: (
                ProgramWorldOperatorKind.RESTORE_TRACKED_ARTIFACT,
            ),
            ProgramWorldDefectFamily.FINITE_ADAPTER: (
                ProgramWorldOperatorKind.FINITE_ADAPTER,
            ),
            ProgramWorldDefectFamily.UNSUPPORTED: (),
        }
    )
)


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text(value: Any, name: str, *, required: bool = True, limit: int = MAX_TEXT_CHARS) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise ProgramWorldOperatorError(f"{name} must be a string")
    else:
        text = value
    if required and not text.strip():
        raise ProgramWorldOperatorError(f"{name} is required")
    encoded = text.encode("utf-8")
    if len(encoded) > limit:
        raise ProgramWorldOperatorBoundsError(f"{name} exceeds its byte bound")
    if "\x00" in text:
        raise ProgramWorldOperatorError(f"{name} must not contain NUL bytes")
    return text


def _optional_text(value: Any, name: str, *, limit: int = MAX_TEXT_CHARS) -> str:
    return _text(value, name, required=False, limit=limit)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ProgramWorldOperatorError(f"{name} must be a boolean")
    return value


def _enum(value: Any, enum_cls: type[Enum], name: str) -> Any:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(sorted(item.value for item in enum_cls))
        raise ProgramWorldOperatorError(f"{name} must be one of: {allowed}") from exc


def _ids(values: Any, name: str, *, limit: int = MAX_COLLECTION_ITEMS) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ProgramWorldOperatorError(f"{name} must be a sequence of identifiers")
    if len(values) > limit:
        raise ProgramWorldOperatorBoundsError(f"{name} exceeds its bound")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return tuple(ordered)


def normalize_repo_path(value: Any, name: str = "path") -> str:
    text = _text(value, name, limit=MAX_PATH_BYTES).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    candidate = PurePosixPath(text)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or text.startswith("/")
        or (candidate.parts and candidate.parts[0].endswith(":"))
    ):
        raise ProgramWorldOperatorAuthorityError(
            f"{name} must be a bounded relative repository path"
        )
    return candidate.as_posix()


def _paths(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ProgramWorldOperatorError(f"{name} must be a sequence of paths")
    if len(values) > MAX_COLLECTION_ITEMS:
        raise ProgramWorldOperatorBoundsError(f"{name} exceeds its bound")
    return tuple(normalize_repo_path(item, name) for item in values)


def _source_bytes(value: Any, name: str = "source_text") -> str:
    text = _text(value, name, required=True, limit=MAX_SOURCE_BYTES)
    if len(text.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ProgramWorldOperatorBoundsError(f"{name} exceeds source bound")
    return text


def _span(value: Any, name: str, *, required: bool = False) -> str:
    return _text(value, name, required=required, limit=MAX_SPAN_BYTES)


def path_is_protected(path: str, protected: Sequence[str]) -> bool:
    candidate = normalize_repo_path(path)
    for item in protected:
        prefix = normalize_repo_path(item)
        if candidate == prefix:
            return True
        if prefix.endswith("/"):
            if candidate.startswith(prefix) or f"{candidate}/".startswith(prefix):
                return True
        elif candidate.startswith(prefix + "/"):
            return True
    return False


def path_is_admitted(path: str, admitted_scope: Sequence[str]) -> bool:
    if not admitted_scope:
        return False
    candidate = normalize_repo_path(path)
    for item in admitted_scope:
        prefix = normalize_repo_path(item)
        if candidate == prefix:
            return True
        if prefix.endswith("/"):
            if candidate.startswith(prefix):
                return True
        elif candidate.startswith(prefix + "/"):
            return True
        if prefix.startswith(candidate + "/"):
            return True
    return False


def is_test_path(path: str) -> bool:
    candidate = normalize_repo_path(path)
    name = PurePosixPath(candidate).name
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or "/test/" in f"/{candidate}/"
        or "/tests/" in f"/{candidate}/"
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProgramWorldOperatorError(f"{name} must be an object")
    if len(value) > MAX_METADATA_KEYS:
        raise ProgramWorldOperatorBoundsError(f"{name} exceeds key bound")
    return {str(key): item for key, item in value.items()}


def _walk_forbidden_keys(value: Any, *, path: str = "") -> tuple[str, ...]:
    reasons: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_s = str(key)
            norm = key_s.casefold().replace("-", "_")
            child = f"{path}.{key_s}" if path else key_s
            if norm in _AUTHORITY_CLAIM_KEYS and item is True:
                reasons.append(f"authority_claim:{child}")
            if norm in _SCOPE_WIDEN_KEYS and item not in (None, (), [], {}, False, ""):
                reasons.append(f"scope_key:{child}")
            if norm in _OBLIGATION_WAIVER_KEYS and item not in (None, (), [], {}, False, ""):
                reasons.append(f"obligation_waiver:{child}")
            reasons.extend(_walk_forbidden_keys(item, path=child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            reasons.extend(_walk_forbidden_keys(item, path=f"{path}[{index}]"))
    return tuple(reasons)


def detect_authority_claims(payload: Mapping[str, Any] | None) -> tuple[str, ...]:
    return _walk_forbidden_keys(payload or {})


def _parse_python(source: str) -> ast.AST | None:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError):
        return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def detect_arbitrary_execution(source: str) -> tuple[str, ...]:
    tree = _parse_python(source)
    if tree is None:
        lowered = source.casefold()
        hits: list[str] = []
        for marker in ("eval(", "exec(", "subprocess", "pickle.loads", "os.system"):
            if marker in lowered:
                hits.append(f"arbitrary_execution:{marker.rstrip('(')}")
        return tuple(hits)
    hits_set: list[str] = []
    seen: set[str] = set()

    def _add(reason: str) -> None:
        if reason not in seen:
            seen.add(reason)
            hits_set.append(reason)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = (alias.name or "").split(".", 1)[0]
                if root in _FORBIDDEN_MODULES:
                    _add(f"arbitrary_execution:import:{root}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in _FORBIDDEN_MODULES:
                _add(f"arbitrary_execution:import:{root}")
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            leaf = name.rsplit(".", 1)[-1]
            if name in _FORBIDDEN_CALLS or leaf in _FORBIDDEN_CALLS:
                _add(f"arbitrary_execution:call:{name or leaf}")
            if leaf in _FORBIDDEN_ATTRS:
                _add(f"arbitrary_execution:attr:{name or leaf}")
            root = name.split(".", 1)[0]
            if root in _FORBIDDEN_MODULES:
                _add(f"arbitrary_execution:module:{root}")
    return tuple(hits_set)


def _test_function_names(tree: ast.AST) -> frozenset[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            names.add(node.name)
    return frozenset(names)


def _decorator_is_skip(node: ast.AST) -> bool:
    name = _call_name(node if not isinstance(node, ast.Call) else node.func)
    lowered = name.replace(" ", "")
    return any(marker in lowered for marker in ("skip", "xfail"))


def _assert_count(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            count += 1
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            leaf = name.rsplit(".", 1)[-1]
            if leaf.startswith("assert") or name.endswith("raises"):
                count += 1
    return count


def _skip_count(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                if _decorator_is_skip(decorator):
                    count += 1
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if any(marker in name for marker in ("skip", "xfail")):
                count += 1
    return count


def _trivial_assert_count(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            test = node.test
            if isinstance(test, ast.Constant) and test.value in {True, 1, "ok"}:
                count += 1
            if isinstance(test, ast.Name) and test.id in {"True"}:
                count += 1
    return count


def detect_test_weakening(before: str, after: str, path: str) -> tuple[str, ...]:
    if not is_test_path(path) and "pytest.skip" not in after and "pytest.mark.skip" not in after:
        return ()
    reasons: list[str] = []
    before_tree = _parse_python(before)
    after_tree = _parse_python(after)
    if before_tree is not None and after_tree is None:
        reasons.append("test_weakening:after_unparseable")
        return tuple(reasons)
    if before_tree is None or after_tree is None:
        lowered_after = after.casefold()
        lowered_before = before.casefold()
        for marker in _SKIP_MARKERS:
            if marker in lowered_after and marker not in lowered_before:
                reasons.append(f"test_weakening:{marker}")
        if after.count("assert ") < before.count("assert "):
            reasons.append("test_weakening:assert_removed")
        return tuple(reasons)
    before_tests = _test_function_names(before_tree)
    after_tests = _test_function_names(after_tree)
    removed = sorted(before_tests - after_tests)
    if removed:
        reasons.append("test_weakening:test_removed")
    if _assert_count(after_tree) < _assert_count(before_tree):
        reasons.append("test_weakening:assert_removed")
    if _skip_count(after_tree) > _skip_count(before_tree):
        reasons.append("test_weakening:skip_added")
    if _trivial_assert_count(after_tree) > _trivial_assert_count(before_tree):
        reasons.append("test_weakening:trivial_assert")
    return tuple(reasons)


def new_execution_risks(before: str, after: str) -> tuple[str, ...]:
    before_hits = set(detect_arbitrary_execution(before))
    after_hits = detect_arbitrary_execution(after)
    return tuple(item for item in after_hits if item not in before_hits)


def _source_offset(source: str, start: tuple[int, int]) -> int:
    line, col = start
    if line < 1:
        return 0
    rows = source.splitlines(keepends=True)
    prefix = sum(len(rows[index]) for index in range(min(line - 1, len(rows))))
    return prefix + col


def _replace_name_tokens(source: str, old: str, new: str) -> tuple[str, int]:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError as exc:
        raise ProgramWorldOperatorError("source is not a well-formed Python token stream") from exc
    pieces: list[str] = []
    last = 0
    count = 0
    for tok in tokens:
        if tok.type == tokenize.NAME and tok.string == old:
            start = _source_offset(source, tok.start)
            end = _source_offset(source, tok.end)
            pieces.append(source[last:start])
            pieces.append(new)
            last = end
            count += 1
    pieces.append(source[last:])
    return "".join(pieces), count


def _count_exact(source: str, span: str) -> int:
    if not span:
        return 0
    return source.count(span)


def _replace_exact_once(source: str, before: str, after: str) -> str:
    return source.replace(before, after, 1)


def _import_line(*, module: str, name: str = "") -> str:
    if not _MODULE_RE.fullmatch(module):
        raise ProgramWorldOperatorError("import_module is not a closed module path")
    if name:
        if not _IDENTIFIER_RE.fullmatch(name):
            raise ProgramWorldOperatorError("import_name is not a closed identifier")
        return f"from {module} import {name}"
    return f"import {module}"


def _insert_import(source: str, line: str) -> str:
    existing = {item.strip() for item in source.splitlines()}
    if line in existing:
        return source
    rows = source.splitlines(keepends=True)
    last_import = -1
    for index, row in enumerate(rows):
        stripped = row.lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import = index
        elif last_import >= 0 and stripped.strip() and not stripped.startswith("#"):
            break
    insertion = line if line.endswith("\n") else line + "\n"
    if last_import >= 0:
        rows.insert(last_import + 1, insertion)
        return "".join(rows)
    if rows and rows[0].startswith("#!"):
        rows.insert(1, insertion)
        return "".join(rows)
    return insertion + source


def _add_argument(source: str, symbol: str, argument: str) -> tuple[str, int]:
    tree = _parse_python(source)
    if tree is None:
        raise ProgramWorldOperatorError("source is not parseable Python")
    if not _IDENTIFIER_RE.fullmatch(argument):
        raise ProgramWorldOperatorError("argument is not a closed identifier")
    matches: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
            matches.append(node)
    if len(matches) != 1:
        return source, len(matches)
    func = matches[0]
    existing = [item.arg for item in func.args.args] + [item.arg for item in func.args.kwonlyargs]
    if argument in existing:
        return source, 1
    if func.lineno < 1:
        return source, 0
    rows = source.splitlines(keepends=True)
    header_index = func.lineno - 1
    header = rows[header_index]
    # Insert before the closing parenthesis of the definition line when unique.
    if ")" not in header or "(" not in header:
        return source, 0
    open_at = header.rfind("(")
    close_at = header.find(")", open_at)
    if close_at < 0:
        return source, 0
    inside = header[open_at + 1 : close_at].strip()
    addition = argument if not inside else f"{inside.rstrip().rstrip(',')}, {argument}"
    rows[header_index] = header[: open_at + 1] + addition + header[close_at:]
    return "".join(rows), 1


def _thread_argument(source: str, symbol: str, argument: str) -> tuple[str, int]:
    after, count = _add_argument(source, symbol, argument)
    if count != 1 or after == source:
        return after, count
    tree = _parse_python(after)
    if tree is None:
        return after, count
    # Thread the new argument into unique same-file calls of ``symbol``.
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(after).readline))
    except tokenize.TokenError:
        return after, count
    pieces: list[str] = []
    last = 0
    threaded = 0
    index = 0
    while index < len(tokens):
        tok = tokens[index]
        nxt = tokens[index + 1] if index + 1 < len(tokens) else None
        if (
            tok.type == tokenize.NAME
            and tok.string == symbol
            and nxt is not None
            and nxt.string == "("
        ):
            close = None
            depth = 0
            for look in range(index + 1, len(tokens)):
                current = tokens[look]
                if current.string == "(":
                    depth += 1
                elif current.string == ")":
                    depth -= 1
                    if depth == 0:
                        close = current
                        break
            if close is not None:
                close_start = _source_offset(after, close.start)
                open_end = _source_offset(after, nxt.end)
                inside = after[open_end:close_start]
                addition = argument if not inside.strip() else f"{inside.rstrip().rstrip(',')}, {argument}"
                pieces.append(after[last:open_end])
                pieces.append(addition)
                last = close_start
                threaded += 1
                index += 1
        index += 1
    pieces.append(after[last:])
    if threaded == 0:
        return after, count
    return "".join(pieces), count


def _equality_rewrite(
    source: str,
    source_term: str,
    target_term: str,
    *,
    environment_binding_cid: str = "",
) -> tuple[str, str, tuple[str, ...]]:
    if not source_term or source_term not in source:
        return source, "abstained", ("equality_term_absent",)
    if source_term == target_term:
        return source, "rejected", ("no_byte_change",)
    try:
        from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes

        from ..analysis.program_egraph import (
            SaturationDisposition,
            normalize_program_fragment,
        )
    except Exception as exc:  # pragma: no cover - import surface is current-tree bound
        raise ProgramWorldOperatorError("equality rewrite requires the landed e-graph") from exc
    env_cid = environment_binding_cid or cid_for_bytes(b"program-world-repair-operators@1")
    try:
        plan = normalize_program_fragment(
            source_term,
            peer_fragment=target_term or None,
            environment_binding_cid=env_cid,
        )
    except Exception:
        return source, "unsupported", ("equality_engine_error",)
    reasons = tuple(plan.reason_codes)
    if plan.disposition == SaturationDisposition.UNSUPPORTED.value:
        return source, "unsupported", reasons or ("equality_unsupported",)
    if plan.disposition == SaturationDisposition.CONFLICT.value:
        return source, "abstained", reasons or ("equality_conflict",)
    if plan.disposition == SaturationDisposition.BOUND_EXHAUSTED.value:
        return source, "abstained", reasons or ("equality_bound_exhausted",)
    replacement = target_term or plan.normal_form
    if not replacement:
        return source, "abstained", reasons or ("equality_no_normal_form",)
    if not plan.equivalent and target_term:
        return source, "abstained", reasons or ("equality_unproved",)
    if _count_exact(source, source_term) != 1:
        return source, "ambiguous", reasons or ("equality_span_not_unique",)
    return _replace_exact_once(source, source_term, replacement), "unique", reasons


@dataclass(frozen=True)
class ProgramWorldRepairOperatorSpec(CanonicalContract):
    """One closed, reviewed program-world operator.  Lookup is not admission."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_OPERATOR_SPEC_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_REPAIR_OPERATOR_REGISTRY_INTERFACE

    kind: ProgramWorldOperatorKind
    span_kind: ProgramWorldSpanKind
    effect_class: ProgramWorldEffectClass = ProgramWorldEffectClass.PURE_LOCAL
    required_fields: tuple[str, ...] = ()
    uniqueness: str = "unique_span"
    existing_kind_alias: str = ""
    proposal_only: bool = True
    grants_write_authority: bool = False
    grants_semantic_authority: bool = False
    grants_proof_authority: bool = False
    grants_completion_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, ProgramWorldOperatorKind, "kind"))
        object.__setattr__(
            self, "span_kind", _enum(self.span_kind, ProgramWorldSpanKind, "span_kind")
        )
        object.__setattr__(
            self,
            "effect_class",
            _enum(self.effect_class, ProgramWorldEffectClass, "effect_class"),
        )
        object.__setattr__(self, "required_fields", _ids(self.required_fields, "required_fields"))
        object.__setattr__(self, "uniqueness", _text(self.uniqueness, "uniqueness"))
        alias = _optional_text(self.existing_kind_alias, "existing_kind_alias")
        expected = EXISTING_OPERATOR_KIND_ALIASES[self.kind.value]
        if alias and alias != expected:
            raise ProgramWorldOperatorAuthorityError(
                "operator alias must extend the current closed kind"
            )
        object.__setattr__(self, "existing_kind_alias", expected)
        if self.proposal_only is not True:
            raise ProgramWorldOperatorAuthorityError("operators must remain proposal-only")
        if (
            self.grants_write_authority
            or self.grants_semantic_authority
            or self.grants_proof_authority
            or self.grants_completion_authority
        ):
            raise ProgramWorldOperatorAuthorityError(
                "operators cannot grant write, semantic, proof, or completion authority"
            )
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "grants_semantic_authority", False)
        object.__setattr__(self, "grants_proof_authority", False)
        object.__setattr__(self, "grants_completion_authority", False)

    @property
    def operator_id(self) -> str:
        return f"operator:{self.kind.value}"

    def _payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "span_kind": self.span_kind.value,
            "effect_class": self.effect_class.value,
            "required_fields": list(self.required_fields),
            "uniqueness": self.uniqueness,
            "existing_kind_alias": self.existing_kind_alias,
            "proposal_only": True,
            "grants_write_authority": False,
            "grants_semantic_authority": False,
            "grants_proof_authority": False,
            "grants_completion_authority": False,
            "evidence": SAWM_CEGIS_REPAIR_EVIDENCE,
            "producer_id": PRODUCER_ID,
        }


def default_program_world_operator_specs() -> tuple[ProgramWorldRepairOperatorSpec, ...]:
    return (
        ProgramWorldRepairOperatorSpec(
            kind=ProgramWorldOperatorKind.REPLACE_EXACT_BYTES,
            span_kind=ProgramWorldSpanKind.BYTES,
            required_fields=("path", "before_span", "after_span"),
            uniqueness="unique_span",
        ),
        ProgramWorldRepairOperatorSpec(
            kind=ProgramWorldOperatorKind.RENAME_EXACT_SYMBOL,
            span_kind=ProgramWorldSpanKind.SYMBOL,
            required_fields=("path", "symbol", "replacement"),
            uniqueness="unique_symbol",
        ),
        ProgramWorldRepairOperatorSpec(
            kind=ProgramWorldOperatorKind.REPLACE_UNIQUE_REGISTRATION,
            span_kind=ProgramWorldSpanKind.REGISTRATION,
            required_fields=("path", "before_span", "after_span"),
            uniqueness="unique_span",
        ),
        ProgramWorldRepairOperatorSpec(
            kind=ProgramWorldOperatorKind.ADD_IMPORT,
            span_kind=ProgramWorldSpanKind.IMPORT,
            required_fields=("path", "import_module"),
            uniqueness="canonical_import_site",
        ),
        ProgramWorldRepairOperatorSpec(
            kind=ProgramWorldOperatorKind.ADD_ARGUMENT,
            span_kind=ProgramWorldSpanKind.ARGUMENT,
            required_fields=("path", "symbol", "argument"),
            uniqueness="unique_function",
        ),
        ProgramWorldRepairOperatorSpec(
            kind=ProgramWorldOperatorKind.THREAD_ARGUMENT,
            span_kind=ProgramWorldSpanKind.ARGUMENT,
            required_fields=("path", "symbol", "argument"),
            uniqueness="unique_function",
        ),
        ProgramWorldRepairOperatorSpec(
            kind=ProgramWorldOperatorKind.EQUALITY_REWRITE,
            span_kind=ProgramWorldSpanKind.EQUALITY,
            required_fields=("path", "source_term"),
            uniqueness="unique_span",
        ),
        ProgramWorldRepairOperatorSpec(
            kind=ProgramWorldOperatorKind.RESTORE_TRACKED_ARTIFACT,
            span_kind=ProgramWorldSpanKind.ARTIFACT,
            required_fields=("path", "after_span", "artifact_digest"),
            uniqueness="unique_path",
            effect_class=ProgramWorldEffectClass.PURE_LOCAL,
        ),
        ProgramWorldRepairOperatorSpec(
            kind=ProgramWorldOperatorKind.FINITE_ADAPTER,
            span_kind=ProgramWorldSpanKind.ADAPTER,
            required_fields=("path", "before_span", "after_span"),
            uniqueness="unique_span",
        ),
    )


@dataclass(frozen=True)
class ProgramWorldPatchSketch(CanonicalContract):
    """Bounded typed patch sketch.  Never a mutation permit."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_PATCH_SKETCH_SCHEMA

    operator_kind: ProgramWorldOperatorKind
    path: str
    before_span: str
    after_span: str
    before_hash: str
    after_hash: str
    occurrence_count: int
    span_kind: ProgramWorldSpanKind
    uniqueness: str
    proof_obligation_ids: tuple[str, ...] = ()
    test_obligation_ids: tuple[str, ...] = ()
    effect_class: ProgramWorldEffectClass = ProgramWorldEffectClass.PURE_LOCAL
    proposal_only: bool = True
    grants_write_authority: bool = False
    grants_semantic_authority: bool = False
    grants_proof_authority: bool = False
    grants_completion_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operator_kind",
            _enum(self.operator_kind, ProgramWorldOperatorKind, "operator_kind"),
        )
        object.__setattr__(self, "path", normalize_repo_path(self.path))
        object.__setattr__(self, "before_span", _span(self.before_span, "before_span"))
        object.__setattr__(self, "after_span", _span(self.after_span, "after_span"))
        object.__setattr__(self, "before_hash", _text(self.before_hash, "before_hash"))
        object.__setattr__(self, "after_hash", _text(self.after_hash, "after_hash"))
        if type(self.occurrence_count) is not int or self.occurrence_count < 0:
            raise ProgramWorldOperatorError("occurrence_count must be a nonnegative integer")
        object.__setattr__(
            self, "span_kind", _enum(self.span_kind, ProgramWorldSpanKind, "span_kind")
        )
        object.__setattr__(self, "uniqueness", _text(self.uniqueness, "uniqueness"))
        object.__setattr__(
            self,
            "proof_obligation_ids",
            _ids(self.proof_obligation_ids, "proof_obligation_ids"),
        )
        object.__setattr__(
            self, "test_obligation_ids", _ids(self.test_obligation_ids, "test_obligation_ids")
        )
        object.__setattr__(
            self,
            "effect_class",
            _enum(self.effect_class, ProgramWorldEffectClass, "effect_class"),
        )
        if self.proposal_only is not True:
            raise ProgramWorldOperatorAuthorityError("patch sketches must remain proposal-only")
        if (
            self.grants_write_authority
            or self.grants_semantic_authority
            or self.grants_proof_authority
            or self.grants_completion_authority
        ):
            raise ProgramWorldOperatorAuthorityError(
                "patch sketches cannot grant authority"
            )
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "grants_semantic_authority", False)
        object.__setattr__(self, "grants_proof_authority", False)
        object.__setattr__(self, "grants_completion_authority", False)

    @property
    def sketch_cid(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "operator_kind": self.operator_kind.value,
            "path": self.path,
            "before_span": self.before_span,
            "after_span": self.after_span,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "occurrence_count": self.occurrence_count,
            "span_kind": self.span_kind.value,
            "uniqueness": self.uniqueness,
            "proof_obligation_ids": list(self.proof_obligation_ids),
            "test_obligation_ids": list(self.test_obligation_ids),
            "effect_class": self.effect_class.value,
            "proposal_only": True,
            "grants_write_authority": False,
            "grants_semantic_authority": False,
            "grants_proof_authority": False,
            "grants_completion_authority": False,
            "evidence": SAWM_CEGIS_REPAIR_EVIDENCE,
        }


@dataclass(frozen=True)
class ProgramWorldOperatorApplication(CanonicalContract):
    """Result of applying one closed operator.  Proposal-only."""

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_OPERATOR_APPLICATION_SCHEMA

    disposition: ProgramWorldOperatorDisposition
    operator_kind: ProgramWorldOperatorKind
    path: str
    reason_codes: tuple[str, ...]
    before_hash: str
    after_source: str = ""
    sketch: ProgramWorldPatchSketch | None = None
    proof_obligation_ids: tuple[str, ...] = ()
    test_obligation_ids: tuple[str, ...] = ()
    analytical: bool = True
    proposal_only: bool = True
    independently_admitted: bool = False
    grants_write_authority: bool = False
    grants_semantic_authority: bool = False
    grants_proof_authority: bool = False
    grants_completion_authority: bool = False
    producer_id: str = PRODUCER_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, ProgramWorldOperatorDisposition, "disposition"),
        )
        object.__setattr__(
            self,
            "operator_kind",
            _enum(self.operator_kind, ProgramWorldOperatorKind, "operator_kind"),
        )
        object.__setattr__(self, "path", normalize_repo_path(self.path))
        object.__setattr__(self, "reason_codes", _ids(self.reason_codes, "reason_codes"))
        object.__setattr__(self, "before_hash", _text(self.before_hash, "before_hash"))
        object.__setattr__(
            self, "after_source", _optional_text(self.after_source, "after_source", limit=MAX_SOURCE_BYTES)
        )
        if self.sketch is not None and not isinstance(self.sketch, ProgramWorldPatchSketch):
            raise ProgramWorldOperatorError("sketch must be a ProgramWorldPatchSketch")
        success = self.disposition in {
            ProgramWorldOperatorDisposition.UNIQUE,
            ProgramWorldOperatorDisposition.APPLIED,
        }
        if success and self.sketch is None:
            raise ProgramWorldOperatorError("successful applications require a patch sketch")
        if not success and self.sketch is not None:
            raise ProgramWorldOperatorError("failed applications cannot carry a patch sketch")
        object.__setattr__(self, "analytical", _bool(self.analytical, "analytical"))
        if self.proposal_only is not True:
            raise ProgramWorldOperatorAuthorityError("applications must remain proposal-only")
        if self.independently_admitted:
            raise ProgramWorldOperatorAuthorityError(
                "operator applications cannot self-admit"
            )
        if (
            self.grants_write_authority
            or self.grants_semantic_authority
            or self.grants_proof_authority
            or self.grants_completion_authority
        ):
            raise ProgramWorldOperatorAuthorityError("applications cannot grant authority")
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(self, "independently_admitted", False)
        object.__setattr__(self, "grants_write_authority", False)
        object.__setattr__(self, "grants_semantic_authority", False)
        object.__setattr__(self, "grants_proof_authority", False)
        object.__setattr__(self, "grants_completion_authority", False)
        object.__setattr__(self, "producer_id", _text(self.producer_id, "producer_id"))

    @property
    def unique(self) -> bool:
        return self.disposition is ProgramWorldOperatorDisposition.UNIQUE

    @property
    def accepted_sketch(self) -> bool:
        return self.disposition in {
            ProgramWorldOperatorDisposition.UNIQUE,
            ProgramWorldOperatorDisposition.APPLIED,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "operator_kind": self.operator_kind.value,
            "path": self.path,
            "reason_codes": list(self.reason_codes),
            "before_hash": self.before_hash,
            "after_hash": _sha256_text(self.after_source) if self.after_source else self.before_hash,
            "sketch": None if self.sketch is None else self.sketch.to_dict(),
            "proof_obligation_ids": list(self.proof_obligation_ids),
            "test_obligation_ids": list(self.test_obligation_ids),
            "analytical": self.analytical,
            "proposal_only": True,
            "independently_admitted": False,
            "grants_write_authority": False,
            "grants_semantic_authority": False,
            "grants_proof_authority": False,
            "grants_completion_authority": False,
            "producer_id": self.producer_id,
            "evidence": SAWM_CEGIS_REPAIR_EVIDENCE,
            "interface": PROGRAM_WORLD_REPAIR_OPERATOR_INTERFACE,
        }


class ProgramWorldRepairOperatorRegistry:
    """Immutable closed catalogue of program-world repair operators."""

    INTERFACE: ClassVar[str] = PROGRAM_WORLD_REPAIR_OPERATOR_REGISTRY_INTERFACE

    def __init__(
        self, specs: Sequence[ProgramWorldRepairOperatorSpec] | None = None
    ) -> None:
        rows = tuple(specs or default_program_world_operator_specs())
        if not rows:
            raise ProgramWorldOperatorError("operator registry cannot be empty")
        kinds = [item.kind for item in rows]
        if len(set(kinds)) != len(kinds):
            raise ProgramWorldOperatorError("operator kinds must be unique")
        unexpected = [item.kind.value for item in rows if item.kind not in ProgramWorldOperatorKind]
        if unexpected:
            raise ProgramWorldOperatorAuthorityError("registry contains an unreviewed operator")
        self._specs = rows
        self._by_kind = {item.kind: item for item in rows}

    @property
    def operators(self) -> tuple[ProgramWorldRepairOperatorSpec, ...]:
        return self._specs

    def kinds(self) -> tuple[ProgramWorldOperatorKind, ...]:
        return tuple(item.kind for item in self._specs)

    def get(self, kind: ProgramWorldOperatorKind | str) -> ProgramWorldRepairOperatorSpec:
        resolved = _enum(kind, ProgramWorldOperatorKind, "operator_kind")
        spec = self._by_kind.get(resolved)
        if spec is None:
            raise ProgramWorldOperatorError(f"operator {resolved.value} is not in the closed vocabulary")
        return spec

    def contains(self, kind: ProgramWorldOperatorKind | str) -> bool:
        try:
            self.get(kind)
        except ProgramWorldOperatorError:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROGRAM_WORLD_OPERATOR_REGISTRY_SCHEMA,
            "interface": self.INTERFACE,
            "evidence": SAWM_CEGIS_REPAIR_EVIDENCE,
            "producer_id": PRODUCER_ID,
            "operators": [item.to_dict() for item in self._specs],
            "closed_vocabulary": [item.kind.value for item in self._specs],
            "proposal_only": True,
            "grants_write_authority": False,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


def list_program_world_repair_operators(
    registry: ProgramWorldRepairOperatorRegistry | None = None,
) -> tuple[ProgramWorldRepairOperatorSpec, ...]:
    return (registry or ProgramWorldRepairOperatorRegistry()).operators


def operators_for_family(
    family: ProgramWorldDefectFamily | str,
) -> tuple[ProgramWorldOperatorKind, ...]:
    resolved = _enum(family, ProgramWorldDefectFamily, "family")
    return FAMILY_TO_OPERATORS.get(resolved, ())


def _fail(
    *,
    kind: ProgramWorldOperatorKind,
    path: str,
    disposition: ProgramWorldOperatorDisposition,
    reasons: Sequence[str],
    before_hash: str,
    proof_obligation_ids: Sequence[str] = (),
    test_obligation_ids: Sequence[str] = (),
    after_source: str = "",
) -> ProgramWorldOperatorApplication:
    return ProgramWorldOperatorApplication(
        disposition=disposition,
        operator_kind=kind,
        path=path,
        reason_codes=tuple(reasons),
        before_hash=before_hash,
        after_source=after_source,
        proof_obligation_ids=tuple(proof_obligation_ids),
        test_obligation_ids=tuple(test_obligation_ids),
    )


def _succeed(
    *,
    kind: ProgramWorldOperatorKind,
    spec: ProgramWorldRepairOperatorSpec,
    path: str,
    source_text: str,
    after_source: str,
    before_span: str,
    after_span: str,
    occurrence_count: int,
    uniqueness: str,
    proof_obligation_ids: Sequence[str],
    test_obligation_ids: Sequence[str],
    unique: bool,
) -> ProgramWorldOperatorApplication:
    before_hash = _sha256_text(source_text)
    after_hash = _sha256_text(after_source)
    sketch = ProgramWorldPatchSketch(
        operator_kind=kind,
        path=path,
        before_span=before_span,
        after_span=after_span,
        before_hash=before_hash,
        after_hash=after_hash,
        occurrence_count=occurrence_count,
        span_kind=spec.span_kind,
        uniqueness=uniqueness,
        proof_obligation_ids=tuple(proof_obligation_ids),
        test_obligation_ids=tuple(test_obligation_ids),
        effect_class=spec.effect_class,
    )
    return ProgramWorldOperatorApplication(
        disposition=(
            ProgramWorldOperatorDisposition.UNIQUE
            if unique
            else ProgramWorldOperatorDisposition.APPLIED
        ),
        operator_kind=kind,
        path=path,
        reason_codes=("deterministic_sketch", "proposal_only"),
        before_hash=before_hash,
        after_source=after_source,
        sketch=sketch,
        proof_obligation_ids=tuple(proof_obligation_ids),
        test_obligation_ids=tuple(test_obligation_ids),
    )


def apply_program_world_repair_operator(
    *,
    operator_kind: ProgramWorldOperatorKind | str,
    path: str,
    source_text: str,
    before_span: str = "",
    after_span: str = "",
    symbol: str = "",
    replacement: str = "",
    argument: str = "",
    import_module: str = "",
    import_name: str = "",
    source_term: str = "",
    target_term: str = "",
    artifact_digest: str = "",
    admitted_scope: Sequence[str] = (),
    protected_paths: Sequence[str] = (),
    trusted_paths: Sequence[str] = (),
    proof_obligation_ids: Sequence[str] = (),
    test_obligation_ids: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    registry: ProgramWorldRepairOperatorRegistry | None = None,
    environment_binding_cid: str = "",
    tree_id: str = "",
) -> ProgramWorldOperatorApplication:
    """Apply one closed operator and emit a typed patch sketch, or abstain."""

    catalogue = registry or ProgramWorldRepairOperatorRegistry()
    kind = _enum(operator_kind, ProgramWorldOperatorKind, "operator_kind")
    if not catalogue.contains(kind):
        raise ProgramWorldOperatorAuthorityError(
            "operator is outside the closed program-world vocabulary"
        )
    spec = catalogue.get(kind)
    normalized_path = normalize_repo_path(path)
    source = _source_bytes(source_text)
    env_cid = _optional_text(environment_binding_cid, "environment_binding_cid")
    _optional_text(tree_id, "tree_id")
    before_hash = _sha256_text(source)
    proofs = _ids(proof_obligation_ids, "proof_obligation_ids")
    tests = _ids(test_obligation_ids, "test_obligation_ids")
    protected = tuple(DEFAULT_PROTECTED_PATHS) + _paths(protected_paths, "protected_paths")
    trusted = _paths(trusted_paths, "trusted_paths")
    scope = _paths(admitted_scope, "admitted_scope")
    meta = _mapping(metadata, "metadata")
    forbidden = detect_authority_claims(meta)
    if forbidden:
        return _fail(
            kind=kind,
            path=normalized_path,
            disposition=ProgramWorldOperatorDisposition.REJECTED,
            reasons=forbidden,
            before_hash=before_hash,
            proof_obligation_ids=proofs,
            test_obligation_ids=tests,
        )
    if not scope:
        return _fail(
            kind=kind,
            path=normalized_path,
            disposition=ProgramWorldOperatorDisposition.OUT_OF_SCOPE,
            reasons=("scope_unspecified",),
            before_hash=before_hash,
            proof_obligation_ids=proofs,
            test_obligation_ids=tests,
        )
    if not path_is_admitted(normalized_path, scope):
        return _fail(
            kind=kind,
            path=normalized_path,
            disposition=ProgramWorldOperatorDisposition.OUT_OF_SCOPE,
            reasons=("path_not_in_admitted_scope",),
            before_hash=before_hash,
            proof_obligation_ids=proofs,
            test_obligation_ids=tests,
        )
    if path_is_protected(normalized_path, protected):
        return _fail(
            kind=kind,
            path=normalized_path,
            disposition=ProgramWorldOperatorDisposition.REJECTED,
            reasons=("protected_path",),
            before_hash=before_hash,
            proof_obligation_ids=proofs,
            test_obligation_ids=tests,
        )
    if path_is_protected(normalized_path, trusted) and kind is not ProgramWorldOperatorKind.RESTORE_TRACKED_ARTIFACT:
        # Trusted paths may only be restored from a verified artifact digest.
        if is_test_path(normalized_path):
            pass
        else:
            return _fail(
                kind=kind,
                path=normalized_path,
                disposition=ProgramWorldOperatorDisposition.REJECTED,
                reasons=("trusted_path",),
                before_hash=before_hash,
                proof_obligation_ids=proofs,
                test_obligation_ids=tests,
            )

    before = _span(before_span, "before_span")
    after = _span(after_span, "after_span")
    symbol_text = _optional_text(symbol, "symbol")
    replacement_text = _optional_text(replacement, "replacement")
    argument_text = _optional_text(argument, "argument") or replacement_text
    module_text = _optional_text(import_module, "import_module")
    import_name_text = _optional_text(import_name, "import_name")
    source_term_text = _optional_text(source_term, "source_term", limit=MAX_SPAN_BYTES)
    target_term_text = _optional_text(target_term, "target_term", limit=MAX_SPAN_BYTES)
    digest_text = _optional_text(artifact_digest, "artifact_digest")

    after_source = source
    occurrence = 0
    uniqueness = spec.uniqueness
    sketch_before = before
    sketch_after = after
    unique = False
    extra_reasons: tuple[str, ...] = ()

    try:
        if kind is ProgramWorldOperatorKind.REPLACE_EXACT_BYTES:
            if not before or after is None:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.ABSTAINED,
                    reasons=("missing_span",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            if before == after:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.REJECTED,
                    reasons=("no_byte_change",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            occurrence = _count_exact(source, before)
            if occurrence == 0:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.ABSTAINED,
                    reasons=("span_absent",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            if occurrence != 1:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.AMBIGUOUS,
                    reasons=("span_not_unique",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            after_source = _replace_exact_once(source, before, after)
            unique = True
            sketch_before, sketch_after = before, after
        elif kind is ProgramWorldOperatorKind.REPLACE_UNIQUE_REGISTRATION:
            if not before or not after:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.ABSTAINED,
                    reasons=("missing_registration_span",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            occurrence = _count_exact(source, before)
            if occurrence != 1:
                disposition = (
                    ProgramWorldOperatorDisposition.ABSTAINED
                    if occurrence == 0
                    else ProgramWorldOperatorDisposition.AMBIGUOUS
                )
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=disposition,
                    reasons=("registration_not_unique" if occurrence else "registration_absent",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            after_source = _replace_exact_once(source, before, after)
            unique = True
            sketch_before, sketch_after = before, after
        elif kind is ProgramWorldOperatorKind.RENAME_EXACT_SYMBOL:
            if not symbol_text or not replacement_text:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.ABSTAINED,
                    reasons=("missing_symbol",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            if not _IDENTIFIER_RE.fullmatch(symbol_text) or not _IDENTIFIER_RE.fullmatch(
                replacement_text
            ):
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.REJECTED,
                    reasons=("identifier_not_closed",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            if symbol_text == replacement_text:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.REJECTED,
                    reasons=("no_byte_change",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            after_source, occurrence = _replace_name_tokens(source, symbol_text, replacement_text)
            if occurrence == 0:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.ABSTAINED,
                    reasons=("symbol_absent",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            unique = True
            sketch_before, sketch_after = symbol_text, replacement_text
        elif kind is ProgramWorldOperatorKind.ADD_IMPORT:
            if not module_text:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.ABSTAINED,
                    reasons=("missing_import_module",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            line = _import_line(module=module_text, name=import_name_text)
            if line in {item.strip() for item in source.splitlines()}:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.ABSTAINED,
                    reasons=("import_already_present",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            after_source = _insert_import(source, line)
            occurrence = 1
            unique = True
            sketch_before, sketch_after = "", line
        elif kind is ProgramWorldOperatorKind.ADD_ARGUMENT:
            if not symbol_text or not argument_text:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.ABSTAINED,
                    reasons=("missing_argument_binding",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            after_source, occurrence = _add_argument(source, symbol_text, argument_text)
            if occurrence == 0:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.ABSTAINED,
                    reasons=("function_absent",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            if occurrence != 1:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.AMBIGUOUS,
                    reasons=("function_not_unique",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            if after_source == source:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.ABSTAINED,
                    reasons=("argument_already_present",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            unique = True
            sketch_before, sketch_after = symbol_text, argument_text
        elif kind is ProgramWorldOperatorKind.THREAD_ARGUMENT:
            if not symbol_text or not argument_text:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.ABSTAINED,
                    reasons=("missing_argument_binding",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            after_source, occurrence = _thread_argument(source, symbol_text, argument_text)
            if occurrence != 1 or after_source == source:
                disposition = (
                    ProgramWorldOperatorDisposition.AMBIGUOUS
                    if occurrence > 1
                    else ProgramWorldOperatorDisposition.ABSTAINED
                )
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=disposition,
                    reasons=("thread_argument_not_unique",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            unique = True
            sketch_before, sketch_after = symbol_text, argument_text
        elif kind is ProgramWorldOperatorKind.EQUALITY_REWRITE:
            term = source_term_text or before
            target = target_term_text or after
            if not term:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.ABSTAINED,
                    reasons=("missing_equality_term",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            after_source, status, extra_reasons = _equality_rewrite(
                source, term, target, environment_binding_cid=env_cid
            )
            if status != "unique":
                mapping = {
                    "ambiguous": ProgramWorldOperatorDisposition.AMBIGUOUS,
                    "unsupported": ProgramWorldOperatorDisposition.UNSUPPORTED,
                    "rejected": ProgramWorldOperatorDisposition.REJECTED,
                }
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=mapping.get(status, ProgramWorldOperatorDisposition.ABSTAINED),
                    reasons=extra_reasons or (status,),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            occurrence = 1
            unique = True
            sketch_before, sketch_after = term, target or after_source
        elif kind is ProgramWorldOperatorKind.RESTORE_TRACKED_ARTIFACT:
            if not after or not digest_text:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.ABSTAINED,
                    reasons=("missing_artifact_bytes",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            expected = digest_text
            if expected != _sha256_text(after):
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.REJECTED,
                    reasons=("artifact_digest_mismatch",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            if after == source:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.ABSTAINED,
                    reasons=("artifact_already_current",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            after_source = after
            occurrence = 1
            unique = True
            sketch_before, sketch_after = source[:MAX_TEXT_CHARS], after[:MAX_TEXT_CHARS]
        elif kind is ProgramWorldOperatorKind.FINITE_ADAPTER:
            if not before or not after or before == after:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=ProgramWorldOperatorDisposition.ABSTAINED,
                    reasons=("adapter_mapping_incomplete",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            occurrence = _count_exact(source, before)
            if occurrence != 1:
                return _fail(
                    kind=kind,
                    path=normalized_path,
                    disposition=(
                        ProgramWorldOperatorDisposition.AMBIGUOUS
                        if occurrence
                        else ProgramWorldOperatorDisposition.ABSTAINED
                    ),
                    reasons=("adapter_span_not_unique" if occurrence else "adapter_span_absent",),
                    before_hash=before_hash,
                    proof_obligation_ids=proofs,
                    test_obligation_ids=tests,
                )
            after_source = _replace_exact_once(source, before, after)
            unique = True
            sketch_before, sketch_after = before, after
        else:
            return _fail(
                kind=kind,
                path=normalized_path,
                disposition=ProgramWorldOperatorDisposition.UNSUPPORTED,
                reasons=("operator_unsupported",),
                before_hash=before_hash,
                proof_obligation_ids=proofs,
                test_obligation_ids=tests,
            )
    except ProgramWorldOperatorError as exc:
        return _fail(
            kind=kind,
            path=normalized_path,
            disposition=ProgramWorldOperatorDisposition.ABSTAINED,
            reasons=(str(exc).split(":")[0].replace(" ", "_")[:64] or "operator_error",),
            before_hash=before_hash,
            proof_obligation_ids=proofs,
            test_obligation_ids=tests,
        )

    if after_source == source:
        return _fail(
            kind=kind,
            path=normalized_path,
            disposition=ProgramWorldOperatorDisposition.ABSTAINED,
            reasons=("no_byte_change",),
            before_hash=before_hash,
            proof_obligation_ids=proofs,
            test_obligation_ids=tests,
            after_source=after_source,
        )

    gate_reasons: list[str] = []
    if new_execution_risks(source, after_source):
        gate_reasons.extend(new_execution_risks(source, after_source))
    weakening = detect_test_weakening(source, after_source, normalized_path)
    if weakening:
        gate_reasons.extend(weakening)
    if is_test_path(normalized_path) and weakening:
        gate_reasons.append("cannot_weaken_tests")
    if extra_reasons:
        # Equality rewrite reasons are informational on success.
        pass
    if gate_reasons:
        return _fail(
            kind=kind,
            path=normalized_path,
            disposition=ProgramWorldOperatorDisposition.REJECTED,
            reasons=tuple(dict.fromkeys(gate_reasons)),
            before_hash=before_hash,
            proof_obligation_ids=proofs,
            test_obligation_ids=tests,
            after_source=after_source,
        )
    if after_source and _parse_python(source) is not None and _parse_python(after_source) is None:
        return _fail(
            kind=kind,
            path=normalized_path,
            disposition=ProgramWorldOperatorDisposition.REJECTED,
            reasons=("type_gate:after_unparseable",),
            before_hash=before_hash,
            proof_obligation_ids=proofs,
            test_obligation_ids=tests,
            after_source=after_source,
        )

    return _succeed(
        kind=kind,
        spec=spec,
        path=normalized_path,
        source_text=source,
        after_source=after_source,
        before_span=sketch_before,
        after_span=sketch_after,
        occurrence_count=occurrence,
        uniqueness=uniqueness,
        proof_obligation_ids=proofs,
        test_obligation_ids=tests,
        unique=unique,
    )


def failure_fingerprint(
    *,
    counterexample_id: str,
    operator_kind: str,
    sketch_cid: str,
    evidence_cid: str,
    strategy: str = "analytical",
) -> str:
    """Content-addressed identity of one failed attempt."""

    return content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/program-world-failure-fingerprint@1",
            "counterexample_id": _text(counterexample_id, "counterexample_id"),
            "operator_kind": _text(operator_kind, "operator_kind"),
            "sketch_cid": _optional_text(sketch_cid, "sketch_cid"),
            "evidence_cid": _text(evidence_cid, "evidence_cid"),
            "strategy": _text(strategy, "strategy"),
        }
    )
