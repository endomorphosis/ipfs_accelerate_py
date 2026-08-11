"""Deterministically adjudicate bounded implementation scope expansions.

Task output declarations are an authority boundary, but they can be
incomplete.  This module distinguishes a demonstrably related companion
change from unrelated scope drift without granting broad path authority.
Every decision is bound to one proposal and repository tree, and remains
non-authoritative for proof, validation, completion, or merge.
"""

from __future__ import annotations

import ast
import fnmatch
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from ..proof.code_proof_obligations import CandidateDiffEntry, DiffChangeKind
from ..proof.formal_verification_contracts import canonical_json, content_identity
from .validation_commands import (
    infer_validation_impact_paths,
    validation_command_repository_root,
)


SCOPE_ADJUDICATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/scope-adjudication-receipt@1"
)
SCOPE_ADJUDICATION_POLICY_VERSION = "deterministic-scope-expansion-v1"
DEFAULT_MAX_SCOPE_EXPANSION_PATHS = 8
DEFAULT_MAX_IMPORT_CLOSURE_DEPTH = 4
DEFAULT_MAX_IMPORT_CLOSURE_FILES = 128
_SUPPORTED_CHANGE_KINDS = frozenset(
    {DiffChangeKind.ADD, DiffChangeKind.MODIFY}
)


class ScopeExpansionVerdict(str, Enum):
    """A deterministic decision for one undeclared candidate path."""

    JUSTIFIED = "justified"
    DENIED = "denied"


class ScopeExpansionReason(str, Enum):
    """Bounded reason codes emitted by the scope adjudicator."""

    EXPLICIT_VALIDATION_TARGET = "explicit_validation_target"
    DECLARED_PATH_IMPORTS_CANDIDATE = "declared_path_imports_candidate"
    CANDIDATE_IMPORTS_DECLARED_PATH = "candidate_imports_declared_path"
    DECLARED_PATH_TRANSITIVELY_IMPORTS_CANDIDATE = (
        "declared_path_transitively_imports_candidate"
    )
    REGRESSION_TEST_IMPORTS_DECLARED_PATH = (
        "regression_test_imports_declared_path"
    )
    INITIAL_GATE_NOT_SCOPE_ONLY = "initial_gate_not_scope_only"
    SCOPE_NOT_DECLARED = "scope_not_declared"
    EXPANSION_LIMIT_EXCEEDED = "expansion_limit_exceeded"
    UNSUPPORTED_CHANGE_KIND = "unsupported_change_kind"
    BINARY_CHANGE = "binary_change"
    SOURCE_UNAVAILABLE = "source_unavailable"
    PYTHON_SYNTAX_ERROR = "python_syntax_error"
    TEST_WEAKENING = "test_weakening"
    TEST_CHANGE_UNVERIFIABLE = "test_change_unverifiable"
    TEST_WITHOUT_REGRESSION_EVIDENCE = "test_without_regression_evidence"
    NO_DEPENDENCY_EVIDENCE = "no_dependency_evidence"


def _normalize_path(value: Any) -> str:
    path = str(value or "").strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if (
        not path
        or path.startswith("/")
        or "\0" in path
        or ".." in PurePosixPath(path).parts
    ):
        return ""
    return PurePosixPath(path).as_posix()


def _normalized_paths(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                path
                for value in values
                if (path := _normalize_path(value))
            }
        )
    )


def _path_matches(path: str, pattern: str) -> bool:
    normalized_path = _normalize_path(path)
    normalized_pattern = _normalize_path(pattern)
    if not normalized_path or not normalized_pattern:
        return False
    if any(character in normalized_pattern for character in "*?["):
        return fnmatch.fnmatchcase(normalized_path, normalized_pattern)
    if str(pattern).strip().replace("\\", "/").endswith("/"):
        return normalized_path.startswith(normalized_pattern.rstrip("/") + "/")
    return (
        normalized_path == normalized_pattern
        or normalized_path.startswith(normalized_pattern.rstrip("/") + "/")
    )


def _is_test_path(path: str) -> bool:
    pure = PurePosixPath(path)
    stem = pure.stem.lower()
    parts = {part.lower() for part in pure.parts[:-1]}
    return (
        stem.startswith("test_")
        or stem.endswith("_test")
        or bool(parts.intersection({"test", "tests"}))
    )


def _module_name(path: str) -> str:
    normalized = _normalize_path(path)
    if not normalized.lower().endswith((".py", ".pyi")):
        return ""
    pure = PurePosixPath(normalized)
    parts = list(pure.parts)
    filename = parts.pop()
    stem = filename.rsplit(".", 1)[0]
    if stem != "__init__":
        parts.append(stem)
    return ".".join(parts)


def _module_names(
    path: str,
    *,
    validation_roots: Sequence[str] = (),
) -> frozenset[str]:
    """Return exact module names under repository and validated roots.

    A monorepo path such as ``project/pkg/module.py`` has the repository-root
    name ``project.pkg.module``.  When a declared validation command has one
    bounded leading ``cd project &&`` clause, Python code in that validation
    may instead import it as ``pkg.module``.  Preserve both exact names so the
    scope gate can prove that relationship without trusting ``sys.path`` edits
    in candidate code or guessing arbitrary source roots.
    """

    normalized_path = _normalize_path(path)
    if not normalized_path:
        return frozenset()
    names = {_module_name(normalized_path)}
    for root in _normalized_paths(validation_roots):
        prefix = root.rstrip("/") + "/"
        if not normalized_path.startswith(prefix):
            continue
        names.add(_module_name(normalized_path[len(prefix) :]))
    return frozenset(name for name in names if name)


def _imported_modules(
    path: str,
    source: str,
) -> tuple[set[str], ast.AST]:
    tree = ast.parse(source, filename=path)
    module = _module_name(path)
    package_parts = module.split(".")[:-1] if module else []
    if PurePosixPath(path).stem == "__init__":
        package_parts = module.split(".") if module else []
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names if alias.name)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            keep = max(0, len(package_parts) - (node.level - 1))
            base_parts = package_parts[:keep]
        else:
            base_parts = []
        if node.module:
            base_parts.extend(
                part for part in node.module.split(".") if part
            )
        base = ".".join(base_parts)
        if base:
            imported.add(base)
        for alias in node.names:
            if alias.name == "*":
                continue
            candidate = ".".join((*base_parts, alias.name))
            if candidate:
                imported.add(candidate)
    return imported, tree


def _test_shape(tree: ast.AST) -> tuple[frozenset[str], int]:
    tests: set[str] = set()
    assertions = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                tests.add(node.name)
        if isinstance(node, ast.Assert):
            assertions += 1
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Attribute):
                if (
                    function.attr.startswith("assert")
                    or function.attr == "raises"
                ):
                    assertions += 1
            elif isinstance(function, ast.Name) and function.id.startswith(
                "assert"
            ):
                assertions += 1
    return frozenset(tests), assertions


def _test_change_preserves_checks(
    entry: CandidateDiffEntry,
    after_tree: ast.AST,
) -> bool:
    after_tests, after_assertions = _test_shape(after_tree)
    if not after_tests or after_assertions < 1:
        return False
    if entry.change_kind is DiffChangeKind.ADD:
        return True
    if entry.before_source is None:
        return False
    try:
        before_tree = ast.parse(entry.before_source, filename=entry.path)
    except (SyntaxError, TypeError, ValueError):
        return False
    before_tests, before_assertions = _test_shape(before_tree)
    return (
        before_tests.issubset(after_tests)
        and after_assertions >= before_assertions
    )


def _read_source(
    workspace_path: Path | None,
    path: str,
    *,
    max_bytes: int = 1_000_000,
) -> str | None:
    if workspace_path is None:
        return None
    candidate = workspace_path.joinpath(*PurePosixPath(path).parts)
    try:
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or candidate.stat().st_size > max_bytes
        ):
            return None
        return candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _resolve_imported_python_paths(
    workspace_path: Path | None,
    module: str,
) -> tuple[str, ...]:
    """Resolve the bounded in-repository files Python may execute for import."""

    if workspace_path is None:
        return ()
    parts = tuple(part for part in str(module).split(".") if part)
    if not parts or any(not part.isidentifier() for part in parts):
        return ()

    paths: list[str] = []
    for depth in range(1, len(parts)):
        package_path = "/".join((*parts[:depth], "__init__.py"))
        if _read_source(workspace_path, package_path) is not None:
            paths.append(package_path)

    module_path = "/".join(parts) + ".py"
    package_path = "/".join((*parts, "__init__.py"))
    if _read_source(workspace_path, module_path) is not None:
        paths.append(module_path)
    if _read_source(workspace_path, package_path) is not None:
        paths.append(package_path)
    return tuple(dict.fromkeys(paths))


def _bounded_import_closure_evidence(
    *,
    workspace_path: Path | None,
    declared_paths: Sequence[str],
    candidate_paths: Sequence[str],
    known_imports: Mapping[str, set[str]],
    max_depth: int = DEFAULT_MAX_IMPORT_CLOSURE_DEPTH,
    max_files: int = DEFAULT_MAX_IMPORT_CLOSURE_FILES,
) -> dict[str, tuple[str, ...]]:
    """Find short, static import chains from declared files to candidates."""

    if workspace_path is None:
        return {}
    candidates = set(candidate_paths)
    evidence: dict[str, tuple[str, ...]] = {}
    import_cache = dict(known_imports)
    discovered_paths: set[str] = set(import_cache)

    for declared in sorted(set(declared_paths)):
        queue: list[tuple[str, tuple[str, ...], int]] = [
            (declared, (declared,), 0)
        ]
        visited = {declared}
        discovered_paths.add(declared)
        while queue and len(discovered_paths) <= max_files:
            current, chain, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            imported_modules = import_cache.get(current)
            if imported_modules is None:
                source = _read_source(workspace_path, current)
                if source is None:
                    continue
                try:
                    imported_modules, _ = _imported_modules(current, source)
                except (SyntaxError, TypeError, ValueError):
                    continue
                import_cache[current] = imported_modules
            for module in sorted(imported_modules):
                for imported_path in _resolve_imported_python_paths(
                    workspace_path,
                    module,
                ):
                    next_chain = (*chain, imported_path)
                    if imported_path in candidates:
                        previous = evidence.get(imported_path)
                        if previous is None or (
                            len(next_chain),
                            next_chain,
                        ) < (len(previous), previous):
                            evidence[imported_path] = next_chain
                    if (
                        imported_path not in visited
                        and len(discovered_paths) < max_files
                    ):
                        visited.add(imported_path)
                        discovered_paths.add(imported_path)
                        queue.append(
                            (imported_path, next_chain, depth + 1)
                        )
    return evidence


@dataclass(frozen=True)
class ScopePathDecision:
    """One source-free explanation for an undeclared path decision."""

    path: str
    verdict: ScopeExpansionVerdict
    reason_codes: tuple[ScopeExpansionReason, ...]
    evidence_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        path = _normalize_path(self.path)
        if not path:
            raise ValueError("scope decision requires a safe repository path")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "verdict",
            ScopeExpansionVerdict(self.verdict),
        )
        reasons = tuple(
            sorted(
                {
                    ScopeExpansionReason(reason)
                    for reason in self.reason_codes
                },
                key=lambda item: item.value,
            )
        )
        if not reasons:
            raise ValueError("scope decision requires at least one reason")
        justified_reasons = {
            ScopeExpansionReason.EXPLICIT_VALIDATION_TARGET,
            ScopeExpansionReason.DECLARED_PATH_IMPORTS_CANDIDATE,
            ScopeExpansionReason.CANDIDATE_IMPORTS_DECLARED_PATH,
            (
                ScopeExpansionReason
                .DECLARED_PATH_TRANSITIVELY_IMPORTS_CANDIDATE
            ),
            ScopeExpansionReason.REGRESSION_TEST_IMPORTS_DECLARED_PATH,
        }
        if self.verdict is ScopeExpansionVerdict.JUSTIFIED:
            if any(reason not in justified_reasons for reason in reasons):
                raise ValueError(
                    "justified scope decisions require positive evidence"
                )
        elif any(reason in justified_reasons for reason in reasons):
            raise ValueError(
                "denied scope decisions cannot claim positive evidence"
            )
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self,
            "evidence_paths",
            _normalized_paths(self.evidence_paths),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "verdict": self.verdict.value,
            "reason_codes": [reason.value for reason in self.reason_codes],
            "evidence_paths": list(self.evidence_paths),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScopePathDecision":
        allowed = {
            "path",
            "verdict",
            "reason_codes",
            "evidence_paths",
        }
        if set(payload).difference(allowed):
            raise ValueError("scope path decision contains unsupported fields")
        return cls(
            path=str(payload.get("path") or ""),
            verdict=ScopeExpansionVerdict(
                str(payload.get("verdict") or "")
            ),
            reason_codes=tuple(
                ScopeExpansionReason(str(reason))
                for reason in payload.get("reason_codes") or ()
            ),
            evidence_paths=tuple(payload.get("evidence_paths") or ()),
        )


@dataclass(frozen=True)
class ScopeAdjudicationReceipt:
    """Content-addressed authority for one bounded proposal expansion."""

    task_id: str
    proposal_id: str
    initial_policy_id: str
    repository_id: str
    repository_tree_id: str
    baseline_id: str
    original_scope_paths: tuple[str, ...]
    candidate_paths: tuple[str, ...]
    initial_finding_codes: tuple[str, ...]
    decisions: tuple[ScopePathDecision, ...]
    authorized_policy_id: str = ""
    policy_version: str = SCOPE_ADJUDICATION_POLICY_VERSION

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "proposal_id",
            "initial_policy_id",
            "repository_id",
            "repository_tree_id",
            "baseline_id",
            "policy_version",
        ):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "authorized_policy_id",
            str(self.authorized_policy_id or "").strip(),
        )
        object.__setattr__(
            self,
            "original_scope_paths",
            _normalized_paths(self.original_scope_paths),
        )
        object.__setattr__(
            self,
            "candidate_paths",
            _normalized_paths(self.candidate_paths),
        )
        object.__setattr__(
            self,
            "initial_finding_codes",
            tuple(
                sorted(
                    {
                        str(code).strip()
                        for code in self.initial_finding_codes
                        if str(code).strip()
                    }
                )
            ),
        )
        decisions = tuple(sorted(self.decisions, key=lambda item: item.path))
        if len({decision.path for decision in decisions}) != len(decisions):
            raise ValueError("scope decisions must have unique paths")
        expected_decision_paths = tuple(
            path
            for path in self.candidate_paths
            if not any(
                _path_matches(path, declared)
                for declared in self.original_scope_paths
            )
        )
        if tuple(decision.path for decision in decisions) != (
            expected_decision_paths
        ):
            raise ValueError(
                "scope decisions must cover every undeclared candidate path"
            )
        object.__setattr__(self, "decisions", decisions)

    @property
    def requested_paths(self) -> tuple[str, ...]:
        return tuple(decision.path for decision in self.decisions)

    @property
    def justified_paths(self) -> tuple[str, ...]:
        return tuple(
            decision.path
            for decision in self.decisions
            if decision.verdict is ScopeExpansionVerdict.JUSTIFIED
        )

    @property
    def authorized_paths(self) -> tuple[str, ...]:
        return self.justified_paths if self.accepted else ()

    @property
    def denied_paths(self) -> tuple[str, ...]:
        return tuple(
            decision.path
            for decision in self.decisions
            if decision.verdict is ScopeExpansionVerdict.DENIED
        )

    @property
    def justified(self) -> bool:
        return (
            self.initial_finding_codes == ("path_outside_scope",)
            and bool(self.decisions)
            and not self.denied_paths
        )

    @property
    def accepted(self) -> bool:
        return self.justified and bool(self.authorized_policy_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCOPE_ADJUDICATION_SCHEMA,
            "task_id": self.task_id,
            "proposal_id": self.proposal_id,
            "initial_policy_id": self.initial_policy_id,
            "repository_id": self.repository_id,
            "repository_tree_id": self.repository_tree_id,
            "baseline_id": self.baseline_id,
            "original_scope_paths": list(self.original_scope_paths),
            "candidate_paths": list(self.candidate_paths),
            "initial_finding_codes": list(self.initial_finding_codes),
            "decisions": [decision.to_dict() for decision in self.decisions],
            "justified_paths": list(self.justified_paths),
            "authorized_paths": list(self.authorized_paths),
            "denied_paths": list(self.denied_paths),
            "justified": self.justified,
            "accepted": self.accepted,
            "authorized_policy_id": self.authorized_policy_id,
            "policy_version": self.policy_version,
            "proof_authoritative": False,
            "completion_authoritative": False,
        }

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict())

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "receipt_id": self.receipt_id}

    def bind_authorized_policy(
        self,
        policy_id: str,
    ) -> "ScopeAdjudicationReceipt":
        """Bind a justified expansion to the exact revalidation policy."""

        normalized = str(policy_id or "").strip()
        if not normalized:
            raise ValueError("authorized policy ID is required")
        if not self.justified:
            raise ValueError("a denied expansion cannot authorize a policy")
        if (
            self.authorized_policy_id
            and self.authorized_policy_id != normalized
        ):
            raise ValueError(
                "scope adjudication is already bound to another policy"
            )
        return replace(self, authorized_policy_id=normalized)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ScopeAdjudicationReceipt":
        if payload.get("schema") != SCOPE_ADJUDICATION_SCHEMA:
            raise ValueError("unsupported scope adjudication schema")
        required = {
            "schema",
            "task_id",
            "proposal_id",
            "initial_policy_id",
            "repository_id",
            "repository_tree_id",
            "baseline_id",
            "original_scope_paths",
            "candidate_paths",
            "initial_finding_codes",
            "decisions",
            "authorized_policy_id",
            "policy_version",
        }
        if required.difference(payload):
            raise ValueError(
                "scope adjudication is missing required fields"
            )
        decisions = tuple(
            ScopePathDecision.from_dict(item)
            for item in payload.get("decisions") or ()
            if isinstance(item, Mapping)
        )
        result = cls(
            task_id=str(payload.get("task_id") or ""),
            proposal_id=str(payload.get("proposal_id") or ""),
            initial_policy_id=str(payload.get("initial_policy_id") or ""),
            repository_id=str(payload.get("repository_id") or ""),
            repository_tree_id=str(
                payload.get("repository_tree_id") or ""
            ),
            baseline_id=str(payload.get("baseline_id") or ""),
            original_scope_paths=tuple(
                payload.get("original_scope_paths") or ()
            ),
            candidate_paths=tuple(payload.get("candidate_paths") or ()),
            initial_finding_codes=tuple(
                payload.get("initial_finding_codes") or ()
            ),
            decisions=decisions,
            authorized_policy_id=str(
                payload.get("authorized_policy_id") or ""
            ),
            policy_version=str(
                payload.get("policy_version")
                or SCOPE_ADJUDICATION_POLICY_VERSION
            ),
        )
        for field_name, expected in (
            ("justified_paths", list(result.justified_paths)),
            ("authorized_paths", list(result.authorized_paths)),
            ("denied_paths", list(result.denied_paths)),
            ("justified", result.justified),
            ("accepted", result.accepted),
            ("proof_authoritative", False),
            ("completion_authoritative", False),
            ("receipt_id", result.receipt_id),
        ):
            if field_name in payload and payload[field_name] != expected:
                raise ValueError(
                    f"scope adjudication {field_name} is inconsistent"
                )
        # Reject hidden fields while permitting the derived public projection.
        allowed = set(result.to_record())
        if set(payload).difference(allowed):
            raise ValueError(
                "scope adjudication contains unsupported fields"
            )
        return result


def _denied_decisions(
    paths: Sequence[str],
    reason: ScopeExpansionReason,
) -> tuple[ScopePathDecision, ...]:
    return tuple(
        ScopePathDecision(
            path=path,
            verdict=ScopeExpansionVerdict.DENIED,
            reason_codes=(reason,),
        )
        for path in paths
    )


def adjudicate_scope_expansion(
    *,
    task_id: str,
    proposal_id: str,
    initial_policy_id: str,
    repository_id: str,
    repository_tree_id: str,
    baseline_id: str,
    original_scope_paths: Sequence[str],
    candidate_diff: Sequence[CandidateDiffEntry],
    initial_finding_codes: Sequence[str],
    validation_commands: Sequence[str] = (),
    workspace_path: Path | None = None,
    max_expansion_paths: int = DEFAULT_MAX_SCOPE_EXPANSION_PATHS,
) -> ScopeAdjudicationReceipt:
    """Adjudicate only undeclared paths in one otherwise-safe proposal.

    A candidate path is justified when it is an explicit validation target or
    has a bounded static Python import path to a declared path. Package
    initializers are included because Python executes them before imported
    submodules. Existing tests must preserve their test functions and
    assertion count. Every other case remains denied; no natural-language or
    model-generated rationale is treated as authority.
    """

    if (
        isinstance(max_expansion_paths, bool)
        or not isinstance(max_expansion_paths, int)
        or max_expansion_paths < 1
    ):
        raise ValueError("max_expansion_paths must be a positive integer")
    scope_paths = _normalized_paths(original_scope_paths)
    candidate_paths = _normalized_paths(
        path
        for entry in candidate_diff
        for path in (entry.old_path, entry.new_path)
        if path
    )
    extra_paths = tuple(
        path
        for path in candidate_paths
        if not any(_path_matches(path, declared) for declared in scope_paths)
    )
    normalized_findings = tuple(
        sorted(
            {
                str(code).strip()
                for code in initial_finding_codes
                if str(code).strip()
            }
        )
    )

    def receipt(
        decisions: Sequence[ScopePathDecision],
    ) -> ScopeAdjudicationReceipt:
        return ScopeAdjudicationReceipt(
            task_id=task_id,
            proposal_id=proposal_id,
            initial_policy_id=initial_policy_id,
            repository_id=repository_id,
            repository_tree_id=repository_tree_id,
            baseline_id=baseline_id,
            original_scope_paths=scope_paths,
            candidate_paths=candidate_paths,
            initial_finding_codes=normalized_findings,
            decisions=tuple(decisions),
        )

    if normalized_findings != ("path_outside_scope",):
        return receipt(
            _denied_decisions(
                extra_paths,
                ScopeExpansionReason.INITIAL_GATE_NOT_SCOPE_ONLY,
            )
        )
    if not scope_paths:
        return receipt(
            _denied_decisions(
                extra_paths,
                ScopeExpansionReason.SCOPE_NOT_DECLARED,
            )
        )
    if len(extra_paths) > max_expansion_paths:
        return receipt(
            _denied_decisions(
                extra_paths,
                ScopeExpansionReason.EXPANSION_LIMIT_EXCEEDED,
            )
        )

    entries_by_path: dict[str, CandidateDiffEntry] = {}
    for entry in candidate_diff:
        for path in (entry.old_path, entry.new_path):
            normalized = _normalize_path(path)
            if normalized:
                entries_by_path[normalized] = entry

    explicit_validation_paths = _normalized_paths(
        path
        for command in validation_commands
        for path in infer_validation_impact_paths(str(command))
    )
    validation_roots = _normalized_paths(
        root
        for command in validation_commands
        if (
            root := validation_command_repository_root(str(command))
        )
    )
    concrete_scope_paths = tuple(
        path
        for path in scope_paths
        if not any(character in path for character in "*?[")
        and path.lower().endswith((".py", ".pyi"))
    )
    python_paths = tuple(
        sorted(
            {
                *concrete_scope_paths,
                *(
                    path
                    for path in extra_paths
                    if path.lower().endswith((".py", ".pyi"))
                ),
            }
        )
    )
    sources: dict[str, str] = {}
    parse_errors: set[str] = set()
    trees: dict[str, ast.AST] = {}
    imports: dict[str, set[str]] = {}
    before_imports: dict[str, set[str]] = {}
    for path in python_paths:
        entry = entries_by_path.get(path)
        source = (
            entry.after_source
            if entry is not None and entry.after_source is not None
            else _read_source(workspace_path, path)
        )
        if source is None:
            continue
        sources[path] = source
        try:
            imported, tree = _imported_modules(path, source)
        except (SyntaxError, TypeError, ValueError):
            parse_errors.add(path)
            continue
        imports[path] = imported
        trees[path] = tree
        if (
            entry is not None
            and entry.before_source is not None
            and _is_test_path(path)
        ):
            try:
                imported_before, _ = _imported_modules(
                    path,
                    entry.before_source,
                )
            except (SyntaxError, TypeError, ValueError):
                imported_before = set()
            before_imports[path] = imported_before
    modules = {
        path: module_names
        for path in python_paths
        if (
            module_names := _module_names(
                path,
                validation_roots=validation_roots,
            )
        )
    }
    transitive_import_evidence = _bounded_import_closure_evidence(
        workspace_path=workspace_path,
        declared_paths=concrete_scope_paths,
        candidate_paths=extra_paths,
        known_imports=imports,
    )

    decisions: list[ScopePathDecision] = []
    for path in extra_paths:
        entry = entries_by_path.get(path)
        if entry is None:
            decisions.append(
                ScopePathDecision(
                    path=path,
                    verdict=ScopeExpansionVerdict.DENIED,
                    reason_codes=(ScopeExpansionReason.SOURCE_UNAVAILABLE,),
                )
            )
            continue
        if entry.binary:
            decisions.append(
                ScopePathDecision(
                    path=path,
                    verdict=ScopeExpansionVerdict.DENIED,
                    reason_codes=(ScopeExpansionReason.BINARY_CHANGE,),
                )
            )
            continue
        if entry.change_kind not in _SUPPORTED_CHANGE_KINDS:
            decisions.append(
                ScopePathDecision(
                    path=path,
                    verdict=ScopeExpansionVerdict.DENIED,
                    reason_codes=(
                        ScopeExpansionReason.UNSUPPORTED_CHANGE_KIND,
                    ),
                )
            )
            continue
        is_python = path.lower().endswith((".py", ".pyi"))
        is_test = _is_test_path(path)
        if is_test and not is_python:
            decisions.append(
                ScopePathDecision(
                    path=path,
                    verdict=ScopeExpansionVerdict.DENIED,
                    reason_codes=(
                        ScopeExpansionReason.TEST_CHANGE_UNVERIFIABLE,
                    ),
                )
            )
            continue
        if is_python:
            if path not in sources:
                decisions.append(
                    ScopePathDecision(
                        path=path,
                        verdict=ScopeExpansionVerdict.DENIED,
                        reason_codes=(
                            ScopeExpansionReason.SOURCE_UNAVAILABLE,
                        ),
                    )
                )
                continue
            if path in parse_errors or path not in trees:
                decisions.append(
                    ScopePathDecision(
                        path=path,
                        verdict=ScopeExpansionVerdict.DENIED,
                        reason_codes=(
                            ScopeExpansionReason.PYTHON_SYNTAX_ERROR,
                        ),
                    )
                )
                continue
            if is_test and not _test_change_preserves_checks(
                entry,
                trees[path],
            ):
                decisions.append(
                    ScopePathDecision(
                        path=path,
                        verdict=ScopeExpansionVerdict.DENIED,
                        reason_codes=(ScopeExpansionReason.TEST_WEAKENING,),
                    )
                )
                continue

        targeted_by = tuple(
            target
            for target in explicit_validation_paths
            if _path_matches(path, target) or _path_matches(target, path)
        )
        if targeted_by and is_python:
            decisions.append(
                ScopePathDecision(
                    path=path,
                    verdict=ScopeExpansionVerdict.JUSTIFIED,
                    reason_codes=(
                        ScopeExpansionReason.EXPLICIT_VALIDATION_TARGET,
                    ),
                    evidence_paths=targeted_by,
                )
            )
            continue

        path_modules = modules.get(path, frozenset())
        path_imports = imports.get(path, set())
        candidate_imports_declared = tuple(
            declared
            for declared in concrete_scope_paths
            if any(
                module in path_imports
                for module in modules.get(declared, ())
            )
        )
        if (
            is_test
            and entry.change_kind is not DiffChangeKind.ADD
        ):
            previous_modules = before_imports.get(path, set())
            candidate_imports_declared = tuple(
                declared
                for declared in candidate_imports_declared
                if any(
                    module in previous_modules
                    for module in modules.get(declared, ())
                )
            )
        declared_imports_candidate = tuple(
            declared
            for declared in concrete_scope_paths
            if path_modules.intersection(imports.get(declared, set()))
        )
        if is_test and candidate_imports_declared:
            decisions.append(
                ScopePathDecision(
                    path=path,
                    verdict=ScopeExpansionVerdict.JUSTIFIED,
                    reason_codes=(
                        ScopeExpansionReason.REGRESSION_TEST_IMPORTS_DECLARED_PATH,
                    ),
                    evidence_paths=candidate_imports_declared,
                )
            )
        elif declared_imports_candidate and not is_test:
            decisions.append(
                ScopePathDecision(
                    path=path,
                    verdict=ScopeExpansionVerdict.JUSTIFIED,
                    reason_codes=(
                        ScopeExpansionReason.DECLARED_PATH_IMPORTS_CANDIDATE,
                    ),
                    evidence_paths=declared_imports_candidate,
                )
            )
        elif candidate_imports_declared:
            decisions.append(
                ScopePathDecision(
                    path=path,
                    verdict=ScopeExpansionVerdict.JUSTIFIED,
                    reason_codes=(
                        ScopeExpansionReason.CANDIDATE_IMPORTS_DECLARED_PATH,
                    ),
                    evidence_paths=candidate_imports_declared,
                )
            )
        elif path in transitive_import_evidence and not is_test:
            decisions.append(
                ScopePathDecision(
                    path=path,
                    verdict=ScopeExpansionVerdict.JUSTIFIED,
                    reason_codes=(
                        ScopeExpansionReason
                        .DECLARED_PATH_TRANSITIVELY_IMPORTS_CANDIDATE,
                    ),
                    evidence_paths=transitive_import_evidence[path],
                )
            )
        else:
            decisions.append(
                ScopePathDecision(
                    path=path,
                    verdict=ScopeExpansionVerdict.DENIED,
                    reason_codes=(
                        (
                            ScopeExpansionReason.TEST_WITHOUT_REGRESSION_EVIDENCE
                            if is_test
                            else ScopeExpansionReason.NO_DEPENDENCY_EVIDENCE
                        ),
                    ),
                )
            )
    return receipt(decisions)


def compact_scope_adjudication(
    receipt: ScopeAdjudicationReceipt,
) -> dict[str, Any]:
    """Return the bounded event/diagnostic projection of a receipt."""

    payload = {
        "receipt_id": receipt.receipt_id,
        "accepted": receipt.accepted,
        "proposal_id": receipt.proposal_id,
        "initial_policy_id": receipt.initial_policy_id,
        "repository_tree_id": receipt.repository_tree_id,
        "authorized_policy_id": receipt.authorized_policy_id,
        "justified_paths": list(receipt.justified_paths),
        "authorized_paths": list(receipt.authorized_paths),
        "denied_paths": list(receipt.denied_paths),
        "decisions": [decision.to_dict() for decision in receipt.decisions],
        "policy_version": receipt.policy_version,
        "proof_authoritative": False,
        "completion_authoritative": False,
    }
    # Exercise canonical serialization here so event persistence cannot fail
    # later on a malformed extension value.
    canonical_json(payload)
    return payload


__all__ = [
    "DEFAULT_MAX_IMPORT_CLOSURE_DEPTH",
    "DEFAULT_MAX_IMPORT_CLOSURE_FILES",
    "DEFAULT_MAX_SCOPE_EXPANSION_PATHS",
    "SCOPE_ADJUDICATION_POLICY_VERSION",
    "SCOPE_ADJUDICATION_SCHEMA",
    "ScopeAdjudicationReceipt",
    "ScopeExpansionReason",
    "ScopeExpansionVerdict",
    "ScopePathDecision",
    "adjudicate_scope_expansion",
    "compact_scope_adjudication",
]
