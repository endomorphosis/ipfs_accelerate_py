"""DuckDB + Quack control plane for per-board isolation and artefacts.

Each supervisor task board is registered here with:

* a dedicated implementation branch (``implementation/<namespace>`` when the
  configured merge target would otherwise be a shared default such as
  ``main`` / ``master``)
* board-scoped checkout-mutation lock names so sibling boards no longer
  serialize on ``implementation-main-merge.lock``
* a DuckDB taskboard projection aggregated by this catalog
* AST / embedding / BM25 / knowledge-graph / proof-cache artefacts

DuckLake + Quack are loaded when the local DuckDB build can INSTALL/LOAD
them.  When those extensions are absent the same schema is stored in a
hermetic DuckDB catalog and sibling board databases are ATTACH-ed.  Import
is side-effect free: no DuckDB handle is opened until
:func:`open_board_control_plane`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .duckdb_state import DuckDBConnection
from .task_identity import normalize_board_namespace


BOARD_CONTROL_PLANE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/board-control-plane@1"
)
BOARD_CONTROL_PLANE_SCHEMA_VERSION: Final = 1
IMPLEMENTATION_BRANCH_PREFIX: Final = "implementation/"
CONTROL_PLANE_DIRECTORY_NAME: Final = "agent-board-control-plane"
CONTROL_DATABASE_NAME: Final = "control.duckdb"
SHARED_IMPLEMENTATION_BRANCHES: Final[frozenset[str]] = frozenset(
    {
        "",
        "HEAD",
        "head",
        "main",
        "master",
        "origin/HEAD",
        "origin/head",
        "origin/main",
        "origin/master",
    }
)
ARTEFACT_KINDS: Final[tuple[str, ...]] = (
    "ast",
    "embedding",
    "vector_index",
    "bm25",
    "knowledge_graph",
    "proof_cache",
    "other",
)
_ARTEFACT_TABLES: Final[dict[str, str]] = {
    "ast": "artefact_ast",
    "embedding": "artefact_embeddings",
    "vector_index": "artefact_embeddings",
    "bm25": "artefact_bm25",
    "knowledge_graph": "artefact_knowledge_graph",
    "proof_cache": "artefact_proof_cache",
    "other": "artefact_generic",
}
_LAKE_SNAPSHOT_TABLES: Final[tuple[str, ...]] = (
    "board_catalog",
    "board_tasks",
    "artefact_ast",
    "artefact_embeddings",
    "artefact_bm25",
    "artefact_knowledge_graph",
    "artefact_proof_cache",
    "artefact_generic",
    "control_metadata",
)
_CODEBASE_SKIP_DIRS: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".hg",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".worktrees",
        "worktrees",
        ".pytest_cache",
        "dist",
        "build",
        "site-packages",
    }
)
_CODEBASE_DEFAULT_SUFFIXES: Final[tuple[str, ...]] = (".py",)
_CODEBASE_MAX_FILES: Final = 400
_CODEBASE_MAX_FILE_BYTES: Final = 256_000
_VECTOR_INDEX_DIMENSIONS: Final = 64
_SAFE_BRANCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_TASK_HEADER = re.compile(r"^##\s+(\S+)\s*(.*)$")
_MAX_JSON_BYTES: Final = 2 * 1024 * 1024
_MAX_TASKS: Final = 8_192


class BoardControlPlaneError(RuntimeError):
    """Control-plane construction or mutation failed."""


class BoardControlPlaneUnavailableError(BoardControlPlaneError):
    """DuckDB is not installed in this interpreter."""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _canonical_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    if len(encoded.encode("utf-8")) > _MAX_JSON_BYTES:
        raise BoardControlPlaneError("control-plane payload exceeds JSON bound")
    return encoded


def _safe_token(value: str, *, fallback: str = "default") -> str:
    text = "".join(
        character if character.isalnum() or character in "-._" else "-"
        for character in str(value or "")
    ).strip("-._")
    return text or fallback


def board_namespace_digest(namespace: str, *, length: int = 16) -> str:
    normalized = normalize_board_namespace(namespace)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:length]


def is_shared_implementation_branch(branch: str | None) -> bool:
    """Return whether ``branch`` is a repo-wide default rather than a board."""

    text = str(branch or "").strip()
    if not text:
        return True
    return text in SHARED_IMPLEMENTATION_BRANCHES or text.lower() in {
        "head",
        "main",
        "master",
    }


def board_implementation_branch(namespace: str) -> str:
    """Return the isolated implementation ref for one board namespace."""

    return f"{IMPLEMENTATION_BRANCH_PREFIX}{_safe_token(normalize_board_namespace(namespace))}"


def resolve_board_implementation_branch(
    explicit_branch: str | None,
    namespace: str,
) -> str:
    """Keep an already-isolated branch; rewrite shared defaults per board."""

    text = str(explicit_branch or "").strip()
    if text and not is_shared_implementation_branch(text):
        if text.startswith("-") or any(character.isspace() for character in text):
            raise BoardControlPlaneError(
                f"unsafe merge target branch: {text!r}"
            )
        return text
    return board_implementation_branch(namespace)


def board_merge_lock_name(namespace: str) -> str:
    """Return the git-common-dir merge lock used by one board namespace."""

    return f"implementation-board-{board_namespace_digest(namespace)}-merge.lock"


def board_protected_path_lock_name(namespace: str) -> str:
    """Return the git-common-dir protected-path lock for one board."""

    return (
        "implementation-board-"
        f"{board_namespace_digest(namespace)}-protected-path-maintenance.lock"
    )


def infer_board_namespace(
    *,
    board_namespace: str | None = None,
    merge_target_branch: str | None = None,
    todo_path: str | os.PathLike[str] | None = None,
    state_prefix: str | None = None,
) -> str:
    """Infer a stable board namespace from the supervisor's existing identity."""

    explicit = str(board_namespace or "").strip()
    if explicit:
        return normalize_board_namespace(explicit)
    branch = str(merge_target_branch or "").strip()
    if branch and not is_shared_implementation_branch(branch):
        leaf = branch.rsplit("/", 1)[-1]
        if leaf.startswith(IMPLEMENTATION_BRANCH_PREFIX):
            leaf = leaf[len(IMPLEMENTATION_BRANCH_PREFIX) :]
        return normalize_board_namespace(leaf)
    if todo_path is not None:
        name = Path(todo_path).name
        for suffix in (".todo.md", ".md", ".duckdb", ".ddb"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        if name:
            return normalize_board_namespace(name)
    prefix = str(state_prefix or "").strip()
    if prefix:
        return normalize_board_namespace(prefix)
    return "default"


def discover_board_todo_path(
    repo_root: str | os.PathLike[str],
    namespace: str,
) -> Path | None:
    """Return the markdown taskboard for ``namespace`` when one is checked in."""

    token = _safe_token(normalize_board_namespace(namespace))
    unversioned = re.sub(r"-v\d+$", "", token)
    stems = []
    for raw in (token, unversioned):
        if raw and raw not in stems:
            stems.append(raw)
    names: list[str] = []
    for stem in stems:
        underscored = stem.replace("-", "_")
        hyphenated = stem.replace("_", "-")
        for candidate_name in (
            f"{underscored}.todo.md",
            f"{hyphenated}.todo.md",
            f"{stem}.todo.md",
            "TASK_BOARD.md",
            "task_board.md",
        ):
            if candidate_name not in names:
                names.append(candidate_name)
    root = Path(repo_root)
    architecture = root / "docs" / "architecture"
    search_roots = [architecture, root / "docs", root]
    for stem in stems:
        search_roots.append(architecture / stem.replace("-", "_"))
        search_roots.append(architecture / stem.replace("_", "-"))
    for directory in search_roots:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def _token_embedding(text: str, *, dim: int = _VECTOR_INDEX_DIMENSIONS) -> list[float]:
    """Return a deterministic unit vector for BM25/vector-index fallbacks."""

    tokens = re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())
    vector = [0.0] * max(1, int(dim))
    if not tokens:
        return vector
    width = len(vector)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % width
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 6) for value in vector]


def _iter_codebase_files(
    root: Path,
    *,
    suffixes: Sequence[str] = _CODEBASE_DEFAULT_SUFFIXES,
    max_files: int = _CODEBASE_MAX_FILES,
    max_file_bytes: int = _CODEBASE_MAX_FILE_BYTES,
) -> list[Path]:
    allowed = {str(item).lower() for item in suffixes if str(item).strip()}
    found: list[Path] = []
    if not root.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [
            name
            for name in sorted(dirnames)
            if name not in _CODEBASE_SKIP_DIRS and not name.startswith(".")
        ]
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() not in allowed:
                continue
            try:
                if path.stat().st_size > max_file_bytes:
                    continue
            except OSError:
                continue
            found.append(path)
            if len(found) >= max_files:
                return found
    return found


def _python_module_name(path: Path, repo_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        relative = Path(path.name)
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or path.stem


def _python_ast_summary(text: str) -> dict[str, Any]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "functions": [],
            "classes": [],
            "imports": [],
        }
    functions: list[str] = []
    classes: list[str] = []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names if alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return {
        "ok": True,
        "error": "",
        "functions": sorted(set(functions))[:128],
        "classes": sorted(set(classes))[:128],
        "imports": sorted(set(imports))[:128],
    }


def control_plane_git_common_dir(repo_root: Path) -> Path:
    """Return the Git common directory without importing ``merge``."""

    root = Path(repo_root)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return root / ".git"
    stdout = result.stdout or ""
    if result.returncode != 0 or not stdout.strip():
        return root / ".git"
    path = Path(stdout.strip())
    return path if path.is_absolute() else root / path


def default_control_plane_root(repo_root: Path) -> Path:
    """Return the shared control-plane directory for every worktree of ``repo``."""

    return control_plane_git_common_dir(repo_root) / CONTROL_PLANE_DIRECTORY_NAME


def board_database_path(root: Path, namespace: str) -> Path:
    digest = board_namespace_digest(namespace)
    safe = _safe_token(normalize_board_namespace(namespace))[:48]
    return Path(root) / "boards" / f"{safe}-{digest}.duckdb"


def parse_markdown_board_tasks(text: str) -> list[dict[str, Any]]:
    """Extract a bounded ``## TASK`` projection from a markdown taskboard."""

    tasks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in str(text or "").splitlines():
        header = _TASK_HEADER.match(line)
        if header is not None:
            if current is not None:
                tasks.append(current)
            current = {
                "task_id": header.group(1).strip(),
                "title": header.group(2).strip(),
                "status": "",
                "depends_on": [],
            }
            if len(tasks) + 1 >= _MAX_TASKS:
                break
            continue
        if current is None:
            continue
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("- status:"):
            current["status"] = stripped.split(":", 1)[1].strip()
        elif lowered.startswith("- depends on:"):
            raw = stripped.split(":", 1)[1].strip()
            current["depends_on"] = [
                item.strip()
                for item in raw.split(",")
                if item.strip() and item.strip().lower() not in {"none", "n/a", "-"}
            ]
    if current is not None and len(tasks) < _MAX_TASKS:
        tasks.append(current)
    return tasks


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def _ref_exists(repo_root: Path, ref: str) -> bool:
    result = _git(repo_root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    return result.returncode == 0


def _default_start_point(repo_root: Path) -> str:
    for candidate in ("main", "master", "HEAD"):
        if _ref_exists(repo_root, candidate):
            return candidate
    return ""


def default_board_worktree_parent(repo_root: Path) -> Path:
    """Return the persistent parent directory for per-board Git worktrees."""

    common = control_plane_git_common_dir(repo_root)
    parent = Path(common).resolve().parent
    return parent / "workspace" / "agent-board-worktrees"


def board_implementation_worktree_path(
    repo_root: Path,
    namespace: str,
    *,
    worktree_parent: str | os.PathLike[str] | None = None,
) -> Path:
    parent = (
        Path(worktree_parent)
        if worktree_parent is not None
        else default_board_worktree_parent(repo_root)
    )
    return parent / _safe_token(normalize_board_namespace(namespace))


def _worktree_entries(repo_root: Path) -> list[dict[str, str]]:
    result = _git(repo_root, "worktree", "list", "--porcelain")
    if result.returncode != 0:
        return []
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in (result.stdout or "").splitlines():
        if line.startswith("worktree "):
            if current:
                entries.append(current)
            current = {"worktree": line.split(" ", 1)[1]}
        elif line.startswith("branch "):
            current["branch"] = line.split(" ", 1)[1].removeprefix("refs/heads/")
        elif line.startswith("HEAD "):
            current["head"] = line.split(" ", 1)[1]
    if current:
        entries.append(current)
    return entries


def ensure_board_implementation_worktree(
    repo_root: Path,
    branch: str,
    namespace: str,
    *,
    worktree_parent: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Ensure one durable worktree exists for the board implementation branch.

    If the current checkout is already on ``branch``, that checkout is reused.
    If another worktree already has the branch, that path is reused. Otherwise
    a new worktree is added without checking out the caller's tree.
    """

    name = str(branch or "").strip()
    if not name or name.startswith("-") or not _SAFE_BRANCH.fullmatch(name):
        raise BoardControlPlaneError(f"unsafe implementation branch: {branch!r}")
    root = Path(repo_root)
    current = _git(root, "branch", "--show-current")
    if current.returncode == 0 and current.stdout.strip() == name:
        return {
            "schema": BOARD_CONTROL_PLANE_SCHEMA,
            "created": False,
            "branch": name,
            "worktree": str(root.resolve()),
            "reason": "already_on_branch",
        }
    for entry in _worktree_entries(root):
        if entry.get("branch") == name and entry.get("worktree"):
            return {
                "schema": BOARD_CONTROL_PLANE_SCHEMA,
                "created": False,
                "branch": name,
                "worktree": entry["worktree"],
                "reason": "existing_worktree",
            }
    path = board_implementation_worktree_path(
        root,
        namespace,
        worktree_parent=worktree_parent,
    )
    if path.exists():
        try:
            if path.resolve() == root.resolve():
                return {
                    "schema": BOARD_CONTROL_PLANE_SCHEMA,
                    "created": False,
                    "branch": name,
                    "worktree": str(path),
                    "reason": "path_is_repo_root",
                }
        except OSError:
            pass
        probe = _git(path, "rev-parse", "--is-inside-work-tree")
        if probe.returncode == 0 and probe.stdout.strip() == "true":
            return {
                "schema": BOARD_CONTROL_PLANE_SCHEMA,
                "created": False,
                "branch": name,
                "worktree": str(path),
                "reason": "path_exists",
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    start = _default_start_point(root) or "HEAD"
    result = _git(
        root,
        "worktree",
        "add",
        "-B",
        name,
        str(path),
        start,
    )
    if result.returncode != 0:
        return {
            "schema": BOARD_CONTROL_PLANE_SCHEMA,
            "created": False,
            "branch": name,
            "worktree": str(path),
            "start_point": start,
            "reason": "git_worktree_add_failed",
            "error": (result.stderr or result.stdout or "").strip()[-500:],
        }
    return {
        "schema": BOARD_CONTROL_PLANE_SCHEMA,
        "created": True,
        "branch": name,
        "worktree": str(path),
        "start_point": start,
        "reason": "created",
    }


def ensure_board_implementation_branch(
    repo_root: Path,
    branch: str,
    *,
    start_point: str | None = None,
) -> dict[str, Any]:
    """Create ``branch`` from ``start_point`` when the ref is missing.

    This creates a ref only (``git branch``).  It does not check out the
    branch, so it does not need the checkout-mutation lock.
    """

    name = str(branch or "").strip()
    if not name or name.startswith("-") or not _SAFE_BRANCH.fullmatch(name):
        raise BoardControlPlaneError(f"unsafe implementation branch: {branch!r}")
    root = Path(repo_root)
    if _ref_exists(root, name):
        return {
            "schema": BOARD_CONTROL_PLANE_SCHEMA,
            "created": False,
            "branch": name,
            "reason": "already_exists",
        }
    start = str(start_point or "").strip() or _default_start_point(root)
    if not start:
        return {
            "schema": BOARD_CONTROL_PLANE_SCHEMA,
            "created": False,
            "branch": name,
            "reason": "no_start_point",
        }
    result = _git(root, "branch", "--", name, start)
    if result.returncode != 0:
        if _ref_exists(root, name):
            return {
                "schema": BOARD_CONTROL_PLANE_SCHEMA,
                "created": False,
                "branch": name,
                "reason": "already_exists",
            }
        return {
            "schema": BOARD_CONTROL_PLANE_SCHEMA,
            "created": False,
            "branch": name,
            "start_point": start,
            "reason": "git_branch_failed",
            "error": (result.stderr or result.stdout or "").strip()[-500:],
        }
    return {
        "schema": BOARD_CONTROL_PLANE_SCHEMA,
        "created": True,
        "branch": name,
        "start_point": start,
        "reason": "created",
    }


BOARD_EXTENSION_INSTALL_POLICY_ENV = (
    "IPFS_ACCELERATE_AGENT_BOARD_EXTENSION_INSTALL_POLICY"
)
BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY = "load_only"


def _try_load_extension(
    connection: Any,
    name: str,
    *,
    allow_install: bool = True,
) -> str:
    """Load a local extension, optionally permitting an explicit INSTALL fallback.

    The sealed :class:`DuckDBConnection` policy locks
    ``enable_external_access=false`` at connect time, which makes community
    extensions such as Quack/DuckLake appear unavailable. The control plane
    therefore opens a raw DuckDB handle and loads already-installed
    extensions from the local extension directory.
    """

    try:
        connection.execute(f"LOAD {name}")
        return ""
    except Exception as load_exc:
        if not allow_install:
            return (
                f"LOAD {type(load_exc).__name__}: {load_exc}; "
                "INSTALL disabled by policy"
            )
        try:
            connection.execute(f"INSTALL {name}")
            connection.execute(f"LOAD {name}")
        except Exception as exc:
            return (
                f"LOAD {type(load_exc).__name__}: {load_exc}; "
                f"INSTALL {type(exc).__name__}: {exc}"
            )
        return ""


def _attach_ducklake(connection: DuckDBConnection, root: Path) -> str:
    lake_meta = (root / "lake.ducklake").as_posix().replace("'", "''")
    lake_data = (root / "lake-data").as_posix().replace("'", "''")
    statements = (
        f"ATTACH 'ducklake:{lake_meta}' AS lake (DATA_PATH '{lake_data}')",
        f"ATTACH 'ducklake:{lake_meta}' AS lake",
    )
    errors: list[str] = []
    for statement in statements:
        try:
            connection.execute(statement)
            return ""
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return "; ".join(errors)


@dataclass
class BoardControlPlane:
    """One DuckDB catalog aggregating every board on a Git common directory."""

    root: Path
    database_path: Path
    quack_loaded: bool = False
    ducklake_loaded: bool = False
    ducklake_attached: bool = False
    backend: str = "hermetic-duckdb"
    extension_errors: tuple[str, ...] = ()
    _connection: DuckDBConnection | None = field(default=None, init=False, repr=False)

    @property
    def available(self) -> bool:
        return self._connection is not None

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            connection.close()

    def __enter__(self) -> "BoardControlPlane":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _conn(self) -> DuckDBConnection:
        if self._connection is None:
            raise BoardControlPlaneUnavailableError(
                "board control plane is not open"
            )
        return self._connection

    def _install_schema(self) -> None:
        connection = self._conn()
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS control_metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR,
                value_json VARCHAR
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS board_catalog (
                board_namespace VARCHAR PRIMARY KEY,
                implementation_branch VARCHAR NOT NULL,
                merge_lock_name VARCHAR NOT NULL,
                protected_path_lock_name VARCHAR NOT NULL,
                source_path VARCHAR,
                source_kind VARCHAR,
                duckdb_path VARCHAR,
                registered_at_ms BIGINT,
                updated_at_ms BIGINT,
                extra_json VARCHAR
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS board_tasks (
                board_namespace VARCHAR,
                task_id VARCHAR,
                status VARCHAR,
                title VARCHAR,
                depends_on_json VARCHAR,
                body_json VARCHAR,
                updated_at_ms BIGINT,
                PRIMARY KEY (board_namespace, task_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS artefact_ast (
                board_namespace VARCHAR,
                artefact_id VARCHAR,
                path VARCHAR,
                digest VARCHAR,
                payload_json VARCHAR,
                updated_at_ms BIGINT,
                PRIMARY KEY (board_namespace, artefact_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS artefact_embeddings (
                board_namespace VARCHAR,
                artefact_id VARCHAR,
                model VARCHAR,
                dim INTEGER,
                vector_json VARCHAR,
                text VARCHAR,
                payload_json VARCHAR,
                updated_at_ms BIGINT,
                PRIMARY KEY (board_namespace, artefact_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS artefact_bm25 (
                board_namespace VARCHAR,
                document_id VARCHAR,
                field VARCHAR,
                tokens VARCHAR,
                payload_json VARCHAR,
                updated_at_ms BIGINT,
                PRIMARY KEY (board_namespace, document_id, field)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS artefact_knowledge_graph (
                board_namespace VARCHAR,
                edge_id VARCHAR,
                subject VARCHAR,
                predicate VARCHAR,
                object VARCHAR,
                payload_json VARCHAR,
                updated_at_ms BIGINT,
                PRIMARY KEY (board_namespace, edge_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS artefact_proof_cache (
                board_namespace VARCHAR,
                proof_id VARCHAR,
                obligation_id VARCHAR,
                status VARCHAR,
                payload_json VARCHAR,
                updated_at_ms BIGINT,
                PRIMARY KEY (board_namespace, proof_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS artefact_generic (
                board_namespace VARCHAR,
                kind VARCHAR,
                artefact_id VARCHAR,
                payload_json VARCHAR,
                updated_at_ms BIGINT,
                PRIMARY KEY (board_namespace, kind, artefact_id)
            )
            """
        )
        now = str(_now_ms())
        for key, value in (
            ("schema", BOARD_CONTROL_PLANE_SCHEMA),
            ("schema_version", str(BOARD_CONTROL_PLANE_SCHEMA_VERSION)),
            ("backend", self.backend),
            ("quack_loaded", "true" if self.quack_loaded else "false"),
            ("ducklake_loaded", "true" if self.ducklake_loaded else "false"),
            ("updated_at_ms", now),
        ):
            connection.execute(
                """
                INSERT INTO control_metadata (key, value, value_json)
                VALUES (?, ?, NULL)
                ON CONFLICT (key) DO UPDATE SET value = excluded.value
                """,
                [key, value],
            )

    def register_board(
        self,
        namespace: str,
        *,
        source_path: str | os.PathLike[str] | None = None,
        source_kind: str = "",
        merge_target_branch: str = "",
        extra: Mapping[str, Any] | None = None,
        tasks: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Upsert one board into the catalog and optional task snapshot."""

        board = normalize_board_namespace(namespace)
        branch = resolve_board_implementation_branch(merge_target_branch, board)
        duckdb_path = board_database_path(self.root, board)
        duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        now = _now_ms()
        extra_json = _canonical_json(dict(extra or {}))
        kind = str(source_kind or "").strip() or (
            "duckdb"
            if source_path and Path(source_path).suffix.lower() in {".duckdb", ".ddb"}
            else "markdown"
        )
        connection = self._conn()
        existing = connection.execute(
            "SELECT registered_at_ms FROM board_catalog WHERE board_namespace = ?",
            [board],
        ).fetchone()
        registered_at = (
            int(existing[0])
            if existing is not None and existing[0] is not None
            else now
        )
        connection.execute(
            """
            INSERT INTO board_catalog (
                board_namespace,
                implementation_branch,
                merge_lock_name,
                protected_path_lock_name,
                source_path,
                source_kind,
                duckdb_path,
                registered_at_ms,
                updated_at_ms,
                extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (board_namespace) DO UPDATE SET
                implementation_branch = excluded.implementation_branch,
                merge_lock_name = excluded.merge_lock_name,
                protected_path_lock_name = excluded.protected_path_lock_name,
                source_path = excluded.source_path,
                source_kind = excluded.source_kind,
                duckdb_path = excluded.duckdb_path,
                updated_at_ms = excluded.updated_at_ms,
                extra_json = excluded.extra_json
            """,
            [
                board,
                branch,
                board_merge_lock_name(board),
                board_protected_path_lock_name(board),
                str(source_path or ""),
                kind,
                str(duckdb_path),
                registered_at,
                now,
                extra_json,
            ],
        )
        ingested = 0
        if tasks is not None:
            ingested = self.replace_board_tasks(board, tasks)
        self._materialize_board_database(board, duckdb_path)
        self.aggregate_boards()
        return {
            "schema": BOARD_CONTROL_PLANE_SCHEMA,
            "board_namespace": board,
            "implementation_branch": branch,
            "merge_lock_name": board_merge_lock_name(board),
            "protected_path_lock_name": board_protected_path_lock_name(board),
            "source_path": str(source_path or ""),
            "source_kind": kind,
            "duckdb_path": str(duckdb_path),
            "task_count": ingested,
            "backend": self.backend,
            "quack_loaded": self.quack_loaded,
            "ducklake_loaded": self.ducklake_loaded,
        }

    def replace_board_tasks(
        self,
        namespace: str,
        tasks: Sequence[Mapping[str, Any]],
    ) -> int:
        board = normalize_board_namespace(namespace)
        connection = self._conn()
        now = _now_ms()
        connection.execute(
            "DELETE FROM board_tasks WHERE board_namespace = ?",
            [board],
        )
        count = 0
        for raw in tasks:
            if not isinstance(raw, Mapping):
                continue
            task_id = str(
                raw.get("task_id") or raw.get("id") or raw.get("task_alias") or ""
            ).strip()
            if not task_id:
                continue
            depends = raw.get("depends_on") or raw.get("dependencies") or []
            if isinstance(depends, str):
                depends_list = [
                    item.strip()
                    for item in depends.split(",")
                    if item.strip()
                ]
            elif isinstance(depends, Sequence):
                depends_list = [str(item).strip() for item in depends if str(item).strip()]
            else:
                depends_list = []
            connection.execute(
                """
                INSERT INTO board_tasks (
                    board_namespace, task_id, status, title,
                    depends_on_json, body_json, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (board_namespace, task_id) DO UPDATE SET
                    status = excluded.status,
                    title = excluded.title,
                    depends_on_json = excluded.depends_on_json,
                    body_json = excluded.body_json,
                    updated_at_ms = excluded.updated_at_ms
                """,
                [
                    board,
                    task_id,
                    str(raw.get("status") or "").strip(),
                    str(raw.get("title") or "").strip(),
                    _canonical_json(depends_list),
                    _canonical_json(dict(raw)),
                    now,
                ],
            )
            count += 1
            if count >= _MAX_TASKS:
                break
        self._ingest_task_dependency_graph(board, tasks)
        return count

    def _ingest_task_dependency_graph(
        self,
        namespace: str,
        tasks: Sequence[Mapping[str, Any]],
    ) -> int:
        """Project board dependency edges into the DuckLake knowledge graph."""

        count = 0
        for raw in tasks:
            if not isinstance(raw, Mapping):
                continue
            task_id = str(
                raw.get("task_id") or raw.get("id") or raw.get("task_alias") or ""
            ).strip()
            if not task_id:
                continue
            depends = raw.get("depends_on") or raw.get("dependencies") or []
            if isinstance(depends, str):
                items = [item.strip() for item in depends.split(",") if item.strip()]
            elif isinstance(depends, Sequence):
                items = [str(item).strip() for item in depends if str(item).strip()]
            else:
                items = []
            for index, dependency in enumerate(items):
                if dependency.lower() in {"none", "n/a", "-"}:
                    continue
                self.put_artefact(
                    "knowledge_graph",
                    board_namespace=namespace,
                    artefact_id=f"dep:{task_id}:{index}:{dependency}",
                    subject=task_id,
                    predicate="depends_on",
                    object=dependency,
                    payload={"kind": "board_dependency"},
                )
                count += 1
        return count

    def ingest_markdown_board(
        self,
        namespace: str,
        todo_path: str | os.PathLike[str],
        *,
        merge_target_branch: str = "",
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = Path(todo_path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise BoardControlPlaneError(
                f"could not read taskboard {path}: {exc}"
            ) from exc
        tasks = parse_markdown_board_tasks(text)
        return self.register_board(
            namespace,
            source_path=path,
            source_kind="markdown",
            merge_target_branch=merge_target_branch,
            extra=extra,
            tasks=tasks,
        )

    def list_boards(self) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            """
            SELECT board_namespace, implementation_branch, merge_lock_name,
                   protected_path_lock_name, source_path, source_kind,
                   duckdb_path, registered_at_ms, updated_at_ms, extra_json
            FROM board_catalog
            ORDER BY board_namespace
            """
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            extra: Any = {}
            if row[9]:
                try:
                    extra = json.loads(str(row[9]))
                except json.JSONDecodeError:
                    extra = {}
            result.append(
                {
                    "board_namespace": row[0],
                    "implementation_branch": row[1],
                    "merge_lock_name": row[2],
                    "protected_path_lock_name": row[3],
                    "source_path": row[4],
                    "source_kind": row[5],
                    "duckdb_path": row[6],
                    "registered_at_ms": int(row[7] or 0),
                    "updated_at_ms": int(row[8] or 0),
                    "extra": extra,
                }
            )
        return result

    def list_board_tasks(self, namespace: str) -> list[dict[str, Any]]:
        board = normalize_board_namespace(namespace)
        rows = self._conn().execute(
            """
            SELECT task_id, status, title, depends_on_json, body_json
            FROM board_tasks
            WHERE board_namespace = ?
            ORDER BY task_id
            """,
            [board],
        ).fetchall()
        tasks: list[dict[str, Any]] = []
        for row in rows:
            depends: Any = []
            body: Any = {}
            if row[3]:
                try:
                    depends = json.loads(str(row[3]))
                except json.JSONDecodeError:
                    depends = []
            if row[4]:
                try:
                    body = json.loads(str(row[4]))
                except json.JSONDecodeError:
                    body = {}
            tasks.append(
                {
                    "task_id": row[0],
                    "status": row[1],
                    "title": row[2],
                    "depends_on": depends,
                    "body": body,
                }
            )
        return tasks

    def put_artefact(
        self,
        kind: str,
        *,
        board_namespace: str,
        artefact_id: str,
        payload: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Store one planning/repair artefact in the DuckDB catalog."""

        selected = str(kind or "other").strip().lower()
        if selected not in ARTEFACT_KINDS:
            selected = "other"
        if selected == "vector_index":
            selected = "embedding"
        board = normalize_board_namespace(board_namespace)
        identity = str(artefact_id or "").strip()
        if not identity:
            raise BoardControlPlaneError("artefact_id must not be empty")
        now = _now_ms()
        body = _canonical_json(dict(payload or {}))
        connection = self._conn()
        if selected == "ast":
            connection.execute(
                """
                INSERT INTO artefact_ast (
                    board_namespace, artefact_id, path, digest,
                    payload_json, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (board_namespace, artefact_id) DO UPDATE SET
                    path = excluded.path,
                    digest = excluded.digest,
                    payload_json = excluded.payload_json,
                    updated_at_ms = excluded.updated_at_ms
                """,
                [
                    board,
                    identity,
                    str(fields.get("path") or ""),
                    str(fields.get("digest") or ""),
                    body,
                    now,
                ],
            )
        elif selected == "embedding":
            vector = fields.get("vector") or fields.get("embedding") or []
            if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
                vector = []
            connection.execute(
                """
                INSERT INTO artefact_embeddings (
                    board_namespace, artefact_id, model, dim,
                    vector_json, text, payload_json, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (board_namespace, artefact_id) DO UPDATE SET
                    model = excluded.model,
                    dim = excluded.dim,
                    vector_json = excluded.vector_json,
                    text = excluded.text,
                    payload_json = excluded.payload_json,
                    updated_at_ms = excluded.updated_at_ms
                """,
                [
                    board,
                    identity,
                    str(fields.get("model") or ""),
                    int(fields.get("dim") or len(vector) or 0),
                    _canonical_json(list(vector)),
                    str(fields.get("text") or ""),
                    body,
                    now,
                ],
            )
        elif selected == "bm25":
            connection.execute(
                """
                INSERT INTO artefact_bm25 (
                    board_namespace, document_id, field, tokens,
                    payload_json, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (board_namespace, document_id, field) DO UPDATE SET
                    tokens = excluded.tokens,
                    payload_json = excluded.payload_json,
                    updated_at_ms = excluded.updated_at_ms
                """,
                [
                    board,
                    identity,
                    str(fields.get("field") or "body"),
                    str(fields.get("tokens") or fields.get("text") or ""),
                    body,
                    now,
                ],
            )
        elif selected == "knowledge_graph":
            connection.execute(
                """
                INSERT INTO artefact_knowledge_graph (
                    board_namespace, edge_id, subject, predicate, object,
                    payload_json, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (board_namespace, edge_id) DO UPDATE SET
                    subject = excluded.subject,
                    predicate = excluded.predicate,
                    object = excluded.object,
                    payload_json = excluded.payload_json,
                    updated_at_ms = excluded.updated_at_ms
                """,
                [
                    board,
                    identity,
                    str(fields.get("subject") or ""),
                    str(fields.get("predicate") or ""),
                    str(fields.get("object") or ""),
                    body,
                    now,
                ],
            )
        elif selected == "proof_cache":
            connection.execute(
                """
                INSERT INTO artefact_proof_cache (
                    board_namespace, proof_id, obligation_id, status,
                    payload_json, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (board_namespace, proof_id) DO UPDATE SET
                    obligation_id = excluded.obligation_id,
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    updated_at_ms = excluded.updated_at_ms
                """,
                [
                    board,
                    identity,
                    str(fields.get("obligation_id") or ""),
                    str(fields.get("status") or ""),
                    body,
                    now,
                ],
            )
        else:
            connection.execute(
                """
                INSERT INTO artefact_generic (
                    board_namespace, kind, artefact_id, payload_json,
                    updated_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (board_namespace, kind, artefact_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at_ms = excluded.updated_at_ms
                """,
                [board, selected, identity, body, now],
            )
        return {
            "schema": BOARD_CONTROL_PLANE_SCHEMA,
            "kind": selected,
            "board_namespace": board,
            "artefact_id": identity,
            "updated_at_ms": now,
        }

    def list_artefacts(
        self,
        kind: str,
        *,
        board_namespace: str = "",
        limit: int = 256,
    ) -> list[dict[str, Any]]:
        selected = str(kind or "other").strip().lower()
        table = _ARTEFACT_TABLES.get(selected, "artefact_generic")
        bound = max(1, min(int(limit), 1_000))
        connection = self._conn()
        namespace = str(board_namespace or "").strip()
        if namespace:
            cursor = connection.execute(
                f'SELECT * FROM "{table}" WHERE board_namespace = ? LIMIT ?',
                [normalize_board_namespace(namespace), bound],
            )
        else:
            cursor = connection.execute(
                f'SELECT * FROM "{table}" LIMIT ?',
                [bound],
            )
        columns = tuple(str(column) for column in cursor._columns)
        result: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            result.append(
                {
                    columns[index]: row[index]
                    for index in range(min(len(columns), len(row)))
                }
            )
        return result

    def metadata(self) -> dict[str, str]:
        rows = self._conn().execute(
            "SELECT key, value FROM control_metadata ORDER BY key"
        ).fetchall()
        return {str(row[0]): str(row[1] or "") for row in rows}

    def aggregate_boards(self) -> dict[str, Any]:
        """Rebuild the cross-board view used by the control plane."""

        connection = self._conn()
        connection.execute("DROP VIEW IF EXISTS all_board_tasks")
        connection.execute(
            """
            CREATE VIEW all_board_tasks AS
            SELECT board_namespace, task_id, status, title,
                   depends_on_json, body_json, updated_at_ms
            FROM board_tasks
            """
        )
        if self.ducklake_attached:
            try:
                self._project_relations_to_lake()
            except Exception:
                self.ducklake_attached = False
        count_row = connection.execute(
            "SELECT COUNT(*) FROM board_catalog"
        ).fetchone()
        return {
            "schema": BOARD_CONTROL_PLANE_SCHEMA,
            "backend": self.backend,
            "board_count": int(count_row[0] if count_row else 0),
            "ducklake_attached": self.ducklake_attached,
            "quack_loaded": self.quack_loaded,
        }

    def _project_relations_to_lake(self) -> None:
        """Copy catalog, tasks, and codebase artefacts into the DuckLake catalog."""

        if not self.ducklake_attached:
            return
        connection = self._conn()
        for table in _LAKE_SNAPSHOT_TABLES:
            safe = "".join(
                character if character.isalnum() or character == "_" else "_"
                for character in table
            )
            connection.execute(
                f'CREATE OR REPLACE TABLE lake."{safe}" AS SELECT * FROM "{safe}"'
            )

    def ingest_codebase_artefacts(
        self,
        repo_root: str | os.PathLike[str],
        namespace: str,
        *,
        source_root: str | os.PathLike[str] | None = None,
        max_files: int = _CODEBASE_MAX_FILES,
        suffixes: Sequence[str] = _CODEBASE_DEFAULT_SUFFIXES,
        proof_cache_paths: Sequence[str | os.PathLike[str]] = (),
    ) -> dict[str, Any]:
        """Index one codebase snapshot into AST / vector / KG / proof DuckLake tables."""

        board = normalize_board_namespace(namespace)
        root = Path(repo_root)
        scan_root = Path(source_root) if source_root is not None else root
        files = _iter_codebase_files(
            scan_root,
            suffixes=suffixes,
            max_files=max_files,
        )
        ast_count = 0
        embedding_count = 0
        bm25_count = 0
        kg_count = 0
        proof_count = 0
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            digest = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            try:
                relative = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                relative = path.name
            summary = (
                _python_ast_summary(text) if path.suffix.lower() == ".py" else {"ok": True}
            )
            self.put_artefact(
                "ast",
                board_namespace=board,
                artefact_id=f"ast:{relative}",
                path=relative,
                digest=digest,
                payload={
                    "suffix": path.suffix.lower(),
                    "bytes": len(text.encode("utf-8")),
                    **summary,
                },
            )
            ast_count += 1
            tokens = " ".join(
                part
                for part in (
                    relative,
                    " ".join(summary.get("classes") or ()),
                    " ".join(summary.get("functions") or ()),
                    " ".join(summary.get("imports") or ()),
                )
                if part
            )
            vector = _token_embedding(tokens or relative)
            self.put_artefact(
                "vector_index",
                board_namespace=board,
                artefact_id=f"vec:{relative}",
                model="sha256-token-v1",
                dim=len(vector),
                vector=vector,
                text=tokens or relative,
                payload={"path": relative, "digest": digest},
            )
            embedding_count += 1
            self.put_artefact(
                "bm25",
                board_namespace=board,
                artefact_id=f"bm25:{relative}",
                field="source",
                tokens=tokens or relative,
                payload={"path": relative},
            )
            bm25_count += 1
            module = _python_module_name(path, root)
            self.put_artefact(
                "knowledge_graph",
                board_namespace=board,
                artefact_id=f"file:{relative}",
                subject=board,
                predicate="contains_file",
                object=relative,
                payload={"module": module, "digest": digest},
            )
            kg_count += 1
            for imported in summary.get("imports") or ():
                self.put_artefact(
                    "knowledge_graph",
                    board_namespace=board,
                    artefact_id=f"imp:{relative}:{imported}",
                    subject=module,
                    predicate="imports",
                    object=str(imported),
                    payload={"path": relative},
                )
                kg_count += 1
            parse_ok = bool(summary.get("ok", True))
            self.put_artefact(
                "proof_cache",
                board_namespace=board,
                artefact_id=f"parse:{relative}",
                obligation_id="ast.parse",
                status="verified" if parse_ok else "failed",
                payload={
                    "path": relative,
                    "digest": digest,
                    "error": summary.get("error") or "",
                },
            )
            proof_count += 1
        proof_count += self.ingest_proof_cache_files(
            board,
            proof_cache_paths or self._discover_proof_cache_files(root),
        )
        self.aggregate_boards()
        return {
            "schema": BOARD_CONTROL_PLANE_SCHEMA,
            "board_namespace": board,
            "source_root": str(scan_root),
            "file_count": len(files),
            "ast_count": ast_count,
            "embedding_count": embedding_count,
            "bm25_count": bm25_count,
            "knowledge_graph_count": kg_count,
            "proof_cache_count": proof_count,
            "backend": self.backend,
            "ducklake_attached": self.ducklake_attached,
        }

    def _discover_proof_cache_files(self, repo_root: Path) -> list[Path]:
        names = (
            "formal_verification_cache.duckdb",
            "doctor_proof_cache.sqlite3",
            "doctor_proof_cache.duckdb",
        )
        found: list[Path] = []
        search_roots = (
            repo_root / "data" / "agent_supervisor",
            repo_root / "workspace" / "agent-supervisor",
        )
        for directory in search_roots:
            if not directory.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(directory, followlinks=False):
                dirnames[:] = [
                    name
                    for name in dirnames
                    if name not in _CODEBASE_SKIP_DIRS
                ]
                for name in filenames:
                    if name in names:
                        found.append(Path(dirpath) / name)
                        if len(found) >= 16:
                            return found
        return found

    def ingest_proof_cache_files(
        self,
        namespace: str,
        paths: Sequence[str | os.PathLike[str]],
    ) -> int:
        """Copy durable proof-cache receipts into the DuckLake proof table."""

        board = normalize_board_namespace(namespace)
        ingested = 0
        for raw_path in paths:
            path = Path(raw_path)
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".duckdb", ".ddb"}:
                continue
            try:
                cache = DuckDBConnection(path, memory_limit="64MB", threads=1)
            except Exception:
                continue
            try:
                try:
                    rows = cache.execute(
                        "SELECT key_id, key_json, entry_json FROM proof_cache_entries LIMIT 512"
                    ).fetchall()
                except Exception:
                    rows = []
                for row in rows:
                    key_id = str(row[0] or "").strip()
                    if not key_id:
                        continue
                    status = "cached"
                    try:
                        entry = json.loads(str(row[2] or "{}"))
                        if isinstance(entry, Mapping):
                            status = str(
                                entry.get("status") or entry.get("result") or "cached"
                            )
                    except json.JSONDecodeError:
                        entry = {"entry_json": str(row[2] or "")}
                    self.put_artefact(
                        "proof_cache",
                        board_namespace=board,
                        artefact_id=f"cache:{path.name}:{key_id[:80]}",
                        obligation_id=str(row[1] or key_id)[:240],
                        status=status,
                        payload={
                            "source": str(path),
                            "key_id": key_id,
                            "entry": entry if isinstance(entry, Mapping) else {},
                        },
                    )
                    ingested += 1
            finally:
                cache.close()
        return ingested

    def _materialize_board_database(self, namespace: str, path: Path) -> None:
        """Write a queryable per-board DuckDB sibling for Quack/DuckLake attach."""

        board = normalize_board_namespace(namespace)
        tasks = self.list_board_tasks(board)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            sibling = DuckDBConnection(path, memory_limit="128MB", threads=1)
        except Exception:
            return
        try:
            sibling.execute(
                """
                CREATE TABLE IF NOT EXISTS board_tasks (
                    board_namespace VARCHAR,
                    task_id VARCHAR,
                    status VARCHAR,
                    title VARCHAR,
                    depends_on_json VARCHAR,
                    body_json VARCHAR,
                    PRIMARY KEY (board_namespace, task_id)
                )
                """
            )
            sibling.execute("DELETE FROM board_tasks")
            for task in tasks:
                sibling.execute(
                    """
                    INSERT INTO board_tasks VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        board,
                        task["task_id"],
                        task["status"],
                        task["title"],
                        _canonical_json(task.get("depends_on") or []),
                        _canonical_json(task.get("body") or {}),
                    ],
                )
            sibling.execute(
                """
                CREATE TABLE IF NOT EXISTS board_metadata (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR
                )
                """
            )
            sibling.execute(
                """
                INSERT INTO board_metadata VALUES ('board_namespace', ?)
                ON CONFLICT (key) DO UPDATE SET value = excluded.value
                """,
                [board],
            )
        finally:
            sibling.close()


def _open_extension_capable_connection(
    database_path: Path,
    *,
    timeout_seconds: float,
    allow_extension_install: bool,
) -> DuckDBConnection:
    """Open the catalog with a raw DuckDB handle that can LOAD Quack/DuckLake."""

    from .duckdb_state import exclusive_file_lock

    database_path.parent.mkdir(parents=True, exist_ok=True)
    lock_context = exclusive_file_lock(
        database_path.with_name(f".{database_path.name}.lock"),
        timeout_seconds=timeout_seconds,
    )
    lock_context.__enter__()
    try:
        import duckdb
    except ImportError:
        lock_context.__exit__(None, None, None)
        raise
    try:
        raw = duckdb.connect(str(database_path))
        raw.execute("SET threads=1")
        raw.execute("SET memory_limit='128MB'")
        if not allow_extension_install:
            raw.execute("SET autoinstall_known_extensions = false")
            raw.execute("SET autoload_known_extensions = false")
    except BaseException:
        lock_context.__exit__(None, None, None)
        raise
    wrapped = DuckDBConnection.wrap(raw)
    wrapped.path = database_path
    wrapped._lock_context = lock_context
    return wrapped


def open_board_control_plane(
    repo_root: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] | None = None,
    timeout_seconds: float = 30.0,
    allow_extension_install: bool | None = None,
) -> BoardControlPlane:
    """Open (or create) the repo-wide DuckDB + Quack board control plane."""

    configured_policy = str(
        os.environ.get(BOARD_EXTENSION_INSTALL_POLICY_ENV, "") or ""
    ).strip()
    if configured_policy not in {"", BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY}:
        raise BoardControlPlaneError("board extension installation policy is invalid")
    if allow_extension_install is not None and type(allow_extension_install) is not bool:
        raise BoardControlPlaneError("allow_extension_install must be boolean or None")
    installation_allowed = (
        configured_policy != BOARD_EXTENSION_INSTALL_POLICY_LOAD_ONLY
        and allow_extension_install is not False
    )

    catalog_root = Path(root) if root is not None else default_control_plane_root(
        Path(repo_root)
    )
    catalog_root.mkdir(parents=True, exist_ok=True)
    database_path = catalog_root / CONTROL_DATABASE_NAME
    try:
        connection = _open_extension_capable_connection(
            database_path,
            timeout_seconds=timeout_seconds,
            allow_extension_install=installation_allowed,
        )
    except ImportError as exc:
        raise BoardControlPlaneUnavailableError(
            "DuckDB is required for the board control plane"
        ) from exc
    except Exception as exc:
        raise BoardControlPlaneError(
            f"could not open board control plane: {exc}"
        ) from exc

    extension_errors: list[str] = []
    quack_loaded = False
    ducklake_loaded = False
    ducklake_attached = False
    quack_error = _try_load_extension(
        connection,
        "quack",
        allow_install=installation_allowed,
    )
    if quack_error:
        extension_errors.append(f"quack: {quack_error}")
    else:
        quack_loaded = True
    ducklake_error = _try_load_extension(
        connection,
        "ducklake",
        allow_install=installation_allowed,
    )
    if ducklake_error:
        extension_errors.append(f"ducklake: {ducklake_error}")
    else:
        ducklake_loaded = True
        attach_error = _attach_ducklake(connection, catalog_root)
        if attach_error:
            extension_errors.append(f"ducklake_attach: {attach_error}")
        else:
            ducklake_attached = True
    if ducklake_attached and quack_loaded:
        backend = "ducklake+quack"
    elif quack_loaded:
        backend = "duckdb+quack"
    else:
        backend = "hermetic-duckdb"

    plane = BoardControlPlane(
        root=catalog_root,
        database_path=database_path,
        quack_loaded=quack_loaded,
        ducklake_loaded=ducklake_loaded,
        ducklake_attached=ducklake_attached,
        backend=backend,
        extension_errors=tuple(extension_errors),
    )
    plane._connection = connection
    plane._install_schema()
    plane.aggregate_boards()
    return plane


VALIDATION_PYTHON_ENV: Final = "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHON"
VALIDATION_PYTHONPATH_ENV: Final = "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHONPATH"
VALIDATION_PYTHON_MODULES_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_VALIDATION_PYTHON_MODULES"
)
VALIDATION_PLAYWRIGHT_BROWSERS_PATH_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_VALIDATION_PLAYWRIGHT_BROWSERS_PATH"
)
IMPLEMENTATION_PROVIDER_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER"
)
IMPLEMENTATION_FALLBACK_PROVIDER_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER"
)
IMPLEMENTATION_FALLBACK_TRIGGER_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER"
)
GROK_MODEL_ENV: Final = "IPFS_ACCELERATE_AGENT_GROK_MODEL"
CODEX_MODEL_ENV: Final = "IPFS_ACCELERATE_AGENT_CODEX_MODEL"
CODEX_REASONING_EFFORT_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT"
)
DEFAULT_BOARD_GROK_MODEL: Final = "grok-4.6"


def _todo_scheduler_stem(todo_path: str | os.PathLike[str] | None) -> str:
    if todo_path is None:
        return ""
    name = Path(todo_path).name
    for suffix in (".todo.md", ".md", ".duckdb", ".ddb"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def discover_board_scheduler_config(
    repo_root: str | os.PathLike[str],
    todo_path: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Return the board scheduler JSON next to a todo board, if present."""

    stem = _todo_scheduler_stem(todo_path)
    if not stem:
        return None
    candidates = (
        Path(repo_root) / "config" / f"agent_supervisor_{stem}_scheduler.json",
        Path(todo_path).resolve().parent.parent
        / "config"
        / f"agent_supervisor_{stem}_scheduler.json"
        if todo_path is not None
        else None,
    )
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        return resolved
    return None


def apply_board_validation_runtime(
    repo_root: str | os.PathLike[str],
    todo_path: str | os.PathLike[str] | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Bind a board's sealed ``validation_runtime`` when the process lacks one.

    Official ``configured_board_scheduler`` launches already export
    ``IPFS_ACCELERATE_AGENT_VALIDATION_PYTHONPATH`` pointing at the
    non-writable ``/opt/ipfs-accelerate-legal-validation-*`` deployment.
    Operator fallbacks that invoke ``implementation_daemon`` directly skip
    that adapter and then fail closed on host-interpreter drift.  Applying
    the reviewed scheduler document here keeps fallbacks on the same
    approved environment without installing the pin set onto
    ``/usr/bin/python3.12``.
    """

    target = os.environ if environ is None else environ
    if str(target.get(VALIDATION_PYTHONPATH_ENV) or "").strip():
        return {
            "applied": False,
            "reason": "already_configured",
            "pythonpath": str(target.get(VALIDATION_PYTHONPATH_ENV) or ""),
        }
    config_path = discover_board_scheduler_config(repo_root, todo_path)
    if config_path is None:
        return {"applied": False, "reason": "scheduler_config_missing"}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "applied": False,
            "reason": "scheduler_config_unreadable",
            "error": str(exc)[:300],
            "config_path": str(config_path),
        }
    raw = payload.get("validation_runtime") if isinstance(payload, Mapping) else None
    if not isinstance(raw, Mapping):
        return {
            "applied": False,
            "reason": "validation_runtime_absent",
            "config_path": str(config_path),
        }
    executable = str(raw.get("python_executable") or "").strip()
    raw_paths = raw.get("pythonpath_entries")
    if (
        not executable
        or not Path(executable).is_absolute()
        or not isinstance(raw_paths, list)
        or not raw_paths
    ):
        return {
            "applied": False,
            "reason": "validation_runtime_invalid",
            "config_path": str(config_path),
        }
    pythonpath_entries: list[str] = []
    for item in raw_paths:
        text = str(item or "").strip()
        if not text or not Path(text).is_absolute():
            return {
                "applied": False,
                "reason": "validation_runtime_invalid",
                "config_path": str(config_path),
            }
        if not Path(text).is_dir():
            return {
                "applied": False,
                "reason": "validation_pythonpath_unavailable",
                "config_path": str(config_path),
                "pythonpath": text,
            }
        if text not in pythonpath_entries:
            pythonpath_entries.append(text)
    modules = [
        str(item).strip()
        for item in (raw.get("required_modules") or [])
        if str(item).strip()
    ]
    target[VALIDATION_PYTHON_ENV] = executable
    target[VALIDATION_PYTHONPATH_ENV] = os.pathsep.join(pythonpath_entries)
    if modules:
        target[VALIDATION_PYTHON_MODULES_ENV] = ",".join(modules)
    browsers = str(raw.get("playwright_browsers_path") or "").strip()
    if browsers and Path(browsers).is_absolute():
        target[VALIDATION_PLAYWRIGHT_BROWSERS_PATH_ENV] = browsers
    target.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    target.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    return {
        "applied": True,
        "reason": "bound_scheduler_validation_runtime",
        "config_path": str(config_path),
        "python_executable": executable,
        "pythonpath": target[VALIDATION_PYTHONPATH_ENV],
    }


def apply_board_provider_runtime(
    repo_root: str | os.PathLike[str],
    todo_path: str | os.PathLike[str] | None = None,
    *,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Bind the board's reviewed Grok/Codex route when the process lacks one.

    Official ``configured_board_scheduler`` launches already export
    ``IPFS_ACCELERATE_AGENT_GROK_MODEL=grok-4.6``.  Operator fallbacks that
    invoke ``implementation_daemon`` directly inherit no provider env and
    then dispatch the packaged grok-4.5 default.  Applying the scheduler
    ``provider`` block keeps fallbacks on grok-4.6.
    """

    target = os.environ if environ is None else environ
    if str(target.get(GROK_MODEL_ENV) or "").strip():
        return {
            "applied": False,
            "reason": "already_configured",
            "grok_model": str(target.get(GROK_MODEL_ENV) or ""),
        }
    config_path = discover_board_scheduler_config(repo_root, todo_path)
    provider: Mapping[str, Any] | None = None
    if config_path is not None:
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, Mapping) and isinstance(
            payload.get("provider"), Mapping
        ):
            provider = payload["provider"]
    grok_model = ""
    if provider is not None:
        grok_model = str(provider.get("primary_model_id") or "").strip()
        primary = str(provider.get("primary_provider_id") or "").strip()
        fallback = str(provider.get("fallback_provider_id") or "").strip()
        trigger = str(provider.get("fallback_trigger") or "").strip()
        codex_model = str(provider.get("fallback_model_id") or "").strip()
        effort = str(provider.get("fallback_reasoning_effort") or "").strip()
        if primary:
            target[IMPLEMENTATION_PROVIDER_ENV] = primary
        if fallback:
            target[IMPLEMENTATION_FALLBACK_PROVIDER_ENV] = fallback
        if trigger:
            target[IMPLEMENTATION_FALLBACK_TRIGGER_ENV] = trigger
        if codex_model:
            target[CODEX_MODEL_ENV] = codex_model
        if effort:
            target[CODEX_REASONING_EFFORT_ENV] = effort
    if not grok_model:
        grok_model = DEFAULT_BOARD_GROK_MODEL
    target[GROK_MODEL_ENV] = grok_model
    return {
        "applied": True,
        "reason": "bound_scheduler_provider_runtime",
        "config_path": str(config_path) if config_path is not None else "",
        "grok_model": grok_model,
    }


def isolate_board_runtime(
    *,
    repo_root: str | os.PathLike[str],
    board_namespace: str | None = None,
    merge_target_branch: str | None = None,
    todo_path: str | os.PathLike[str] | None = None,
    state_prefix: str | None = None,
    source_kind: str = "",
    ensure_branch: bool = True,
    ensure_worktree: bool = True,
    ingest_todo: bool = True,
    ingest_codebase: bool | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve isolation + open/register the control plane for one board.

    Failures to open DuckDB are returned in the payload rather than raised so
    the supervisor can keep implementing on an isolated branch even when the
    optional catalog is unavailable.
    """

    namespace = infer_board_namespace(
        board_namespace=board_namespace,
        merge_target_branch=merge_target_branch,
        todo_path=todo_path,
        state_prefix=state_prefix,
    )
    validation_runtime = apply_board_validation_runtime(repo_root, todo_path)
    provider_runtime = apply_board_provider_runtime(repo_root, todo_path)
    branch = resolve_board_implementation_branch(merge_target_branch, namespace)
    branch_result: dict[str, Any] = {
        "created": False,
        "branch": branch,
        "reason": "skipped",
    }
    if ensure_branch and (
        is_shared_implementation_branch(merge_target_branch)
        or branch.startswith(IMPLEMENTATION_BRANCH_PREFIX)
    ):
        try:
            branch_result = ensure_board_implementation_branch(
                Path(repo_root),
                branch,
            )
        except BoardControlPlaneError as exc:
            branch_result = {
                "created": False,
                "branch": branch,
                "reason": "unsafe_branch",
                "error": str(exc),
            }
    worktree_result: dict[str, Any] = {
        "created": False,
        "branch": branch,
        "reason": "skipped",
    }
    if ensure_worktree:
        try:
            worktree_result = ensure_board_implementation_worktree(
                Path(repo_root),
                branch,
                namespace,
            )
        except BoardControlPlaneError as exc:
            worktree_result = {
                "created": False,
                "branch": branch,
                "reason": "unsafe_worktree",
                "error": str(exc),
            }
    payload: dict[str, Any] = {
        "schema": BOARD_CONTROL_PLANE_SCHEMA,
        "board_namespace": namespace,
        "implementation_branch": branch,
        "merge_lock_name": board_merge_lock_name(namespace),
        "protected_path_lock_name": board_protected_path_lock_name(namespace),
        "branch_result": branch_result,
        "worktree_result": worktree_result,
        "board_worktree": worktree_result.get("worktree") or "",
        "validation_runtime": validation_runtime,
        "provider_runtime": provider_runtime,
        "control_plane": None,
        "registration": None,
    }
    try:
        plane = open_board_control_plane(repo_root)
    except BoardControlPlaneError as exc:
        payload["control_plane_error"] = str(exc)
        return payload
    try:
        markdown_todo: Path | None = None
        supplied = Path(todo_path) if todo_path is not None else None
        if (
            ingest_todo
            and supplied is not None
            and supplied.is_file()
            and supplied.suffix.lower() in {".md", ".markdown"}
        ):
            markdown_todo = supplied
        if markdown_todo is None:
            markdown_todo = discover_board_todo_path(repo_root, namespace)
        if markdown_todo is not None:
            registration = plane.ingest_markdown_board(
                namespace,
                markdown_todo,
                merge_target_branch=branch,
                extra=extra,
            )
        else:
            registration = plane.register_board(
                namespace,
                source_path=todo_path,
                source_kind=source_kind,
                merge_target_branch=branch,
                extra=extra,
            )
        payload["registration"] = registration
        should_ingest_codebase = (
            bool(ingest_codebase)
            if ingest_codebase is not None
            else os.environ.get(
                "IPFS_ACCELERATE_AGENT_DUCKLAKE_INGEST_CODEBASE", ""
            ).strip()
            in {"1", "true", "yes", "on"}
        )
        if should_ingest_codebase:
            payload["codebase_artefacts"] = plane.ingest_codebase_artefacts(
                repo_root,
                namespace,
            )
        payload["control_plane"] = {
            "root": str(plane.root),
            "database_path": str(plane.database_path),
            "backend": plane.backend,
            "quack_loaded": plane.quack_loaded,
            "ducklake_loaded": plane.ducklake_loaded,
            "ducklake_attached": plane.ducklake_attached,
            "extension_errors": list(plane.extension_errors),
        }
    finally:
        plane.close()
    return payload


__all__ = [
    "ARTEFACT_KINDS",
    "BOARD_CONTROL_PLANE_SCHEMA",
    "BOARD_CONTROL_PLANE_SCHEMA_VERSION",
    "BoardControlPlane",
    "BoardControlPlaneError",
    "BoardControlPlaneUnavailableError",
    "IMPLEMENTATION_BRANCH_PREFIX",
    "SHARED_IMPLEMENTATION_BRANCHES",
    "apply_board_provider_runtime",
    "apply_board_validation_runtime",
    "discover_board_scheduler_config",
    "board_database_path",
    "board_implementation_branch",
    "board_implementation_worktree_path",
    "board_merge_lock_name",
    "board_namespace_digest",
    "board_protected_path_lock_name",
    "control_plane_git_common_dir",
    "default_control_plane_root",
    "default_board_worktree_parent",
    "discover_board_todo_path",
    "ensure_board_implementation_branch",
    "ensure_board_implementation_worktree",
    "infer_board_namespace",
    "is_shared_implementation_branch",
    "isolate_board_runtime",
    "open_board_control_plane",
    "parse_markdown_board_tasks",
    "resolve_board_implementation_branch",
]
