"""Opt-out pytest seal over filesystem mtimes, AST closures, and file hashes.

A passing test is reused only when the hash of the content hashes of every
file in its AST import closure still matches.  mtimes avoid re-reading files
that have not changed.  The seal does not admit task completion and never
opens an extra-gate ``control.duckdb``.

Disable a run with ``IPFS_ACCELERATE_PYTEST_SEAL=0``.  Disable one test with
``@pytest.mark.pytest_seal_opt_out`` or ``@pytest.mark.proof_reuse_disabled``.
"""

from __future__ import annotations

import ast
import hashlib
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

SEAL_ENV = "IPFS_ACCELERATE_PYTEST_SEAL"
SEAL_DUCKDB_ENV = "IPFS_ACCELERATE_PYTEST_SEAL_DUCKDB"
META_INDEX_ENV = "IPFS_ACCELERATE_META_INDEX_DUCKDB"
OPT_OUT_MARKER = "pytest_seal_opt_out"
PROOF_REUSE_OPT_OUT_MARKER = "proof_reuse_disabled"
MAX_CLOSURE_FILES = 800
_OFF = frozenset({"0", "false", "no", "off", "opt-out", "optout"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pytest_ast_file_hash (
    path VARCHAR PRIMARY KEY,
    mtime_ns BIGINT NOT NULL,
    size_bytes BIGINT NOT NULL,
    content_hash VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS pytest_ast_seal (
    nodeid VARCHAR PRIMARY KEY,
    seal VARCHAR NOT NULL,
    file_count INTEGER NOT NULL,
    recorded_at VARCHAR NOT NULL,
    completion_authority BOOLEAN NOT NULL
);
"""


class PytestSealCatalogError(ValueError):
    """The seal catalog path is not usable."""


class PytestSealOracle(Protocol):
    """Injected decision: matching seals return true and nothing else does."""

    def reuse(self, nodeid: str, seal: str) -> bool:
        """Return true only when this node already passed under ``seal``."""

    def remember(self, nodeid: str, seal: str, *, file_count: int) -> None:
        """Store a passing seal.  Must not grant completion authority."""

    def cached_file_hash(self, path: str) -> tuple[int, int, str] | None:
        """Return ``(mtime_ns, size, content_hash)`` when a file was hashed."""

    def remember_file_hash(
        self, path: str, *, mtime_ns: int, size_bytes: int, content_hash: str
    ) -> None:
        """Remember one file hash so an unchanged mtime skips a re-read."""


@dataclass(frozen=True)
class AstFileSeal:
    """One AST-closure file: freshness plus its content hash."""

    path: str
    mtime_ns: int
    size_bytes: int
    content_hash: str


@dataclass(frozen=True)
class PytestAstSeal:
    """Hash of the content hashes for one test's AST closure."""

    digest: str
    files: tuple[AstFileSeal, ...]
    complete: bool

    @property
    def hash_of_hashes(self) -> str:
        return self.digest


def seal_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Sealing is on unless the environment explicitly opts out."""

    raw = str((environ if environ is not None else os.environ).get(SEAL_ENV, "")).strip().lower()
    return raw not in _OFF


def item_opted_out(item: Any) -> bool:
    """A marker opts one test out without disabling the rest of the run."""

    for name in (OPT_OUT_MARKER, PROOF_REUSE_OPT_OUT_MARKER):
        try:
            if item.get_closest_marker(name) is not None:
                return True
        except Exception:
            continue
    return False


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_of_hashes(files: Iterable[tuple[str, str]]) -> str:
    """Seal sorted ``path:content_hash`` lines.  mtimes are not part of the seal."""

    lines = [f"{path}:{content_hash}" for path, content_hash in sorted(files)]
    return _sha256_bytes("\n".join(lines).encode("utf-8"))


def _repo_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return current


def _module_file(repo: Path, module: str) -> Path | None:
    if not module:
        return None
    relative = Path(*module.split("."))
    candidate = repo / relative.with_suffix(".py")
    package = repo / relative / "__init__.py"
    if candidate.is_file():
        return candidate
    if package.is_file():
        return package
    return None


def _package_parts(repo: Path, path: Path) -> tuple[str, ...]:
    relative = path.resolve().relative_to(repo.resolve())
    if path.name == "__init__.py":
        return relative.parent.parts
    return relative.with_suffix("").parts


def _imported_modules(repo: Path, path: Path, tree: ast.AST) -> tuple[str, ...]:
    modules: list[str] = []
    package = _package_parts(repo, path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names if alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package if path.name == "__init__.py" else package[:-1]
                base_parts = base_parts[: len(base_parts) - (node.level - 1)]
                base = ".".join(base_parts)
                if node.module and base:
                    modules.append(f"{base}.{node.module}")
                elif node.module:
                    modules.append(node.module)
                elif base:
                    modules.append(base)
            elif node.module:
                modules.append(node.module)
    return tuple(module for module in modules if module)


def _conftest_chain(repo: Path, start: Path) -> tuple[Path, ...]:
    found: list[Path] = []
    current = start.parent
    root = repo.resolve()
    while True:
        candidate = current / "conftest.py"
        if candidate.is_file():
            found.append(candidate.resolve())
        if current.resolve() == root or root not in current.parents and current.resolve() != root:
            break
        if current == current.parent:
            break
        current = current.parent
    return tuple(found)


def ast_closure(repo: Path, test_file: Path, *, max_files: int = MAX_CLOSURE_FILES) -> tuple[Path, ...] | None:
    """Return the local files named by the test's AST imports, or None if unbounded."""

    root = repo.resolve()
    pending = [test_file.resolve(), *_conftest_chain(root, test_file)]
    seen: set[Path] = set()
    ordered: list[Path] = []
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        if root not in path.parents and path != root:
            continue
        if not path.is_file() or path.suffix != ".py":
            continue
        ordered.append(path)
        if len(ordered) > max_files:
            return None
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError):
            return None
        for module in _imported_modules(root, path, tree):
            resolved = _module_file(root, module)
            if resolved is not None and resolved.resolve() not in seen:
                pending.append(resolved.resolve())
    return tuple(sorted(ordered))


def seal_files(
    files: Iterable[Path],
    oracle: PytestSealOracle,
) -> PytestAstSeal | None:
    """Hash each AST file, reusing a stored hash when the mtime is unchanged."""

    sealed: list[AstFileSeal] = []
    try:
        paths = tuple(files)
    except TypeError:
        return None
    if not paths:
        return None
    for path in paths:
        try:
            stat = path.stat()
            mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
            size = int(stat.st_size)
            cached = oracle.cached_file_hash(str(path))
            if cached is not None and cached[0] == mtime_ns and cached[1] == size:
                content_hash = cached[2]
            else:
                content_hash = _sha256_file(path)
                oracle.remember_file_hash(
                    str(path),
                    mtime_ns=mtime_ns,
                    size_bytes=size,
                    content_hash=content_hash,
                )
        except OSError:
            return None
        sealed.append(
            AstFileSeal(
                path=str(path),
                mtime_ns=mtime_ns,
                size_bytes=size,
                content_hash=content_hash,
            )
        )
    digest = hash_of_hashes((item.path, item.content_hash) for item in sealed)
    return PytestAstSeal(digest=digest, files=tuple(sealed), complete=True)


class MemorySealOracle:
    """In-process oracle.  A matching seal returns true; nothing is authoritative."""

    def __init__(self) -> None:
        self._seals: dict[str, tuple[str, int]] = {}
        self._files: dict[str, tuple[int, int, str]] = {}
        self.completion_authority = False

    def reuse(self, nodeid: str, seal: str) -> bool:
        stored = self._seals.get(nodeid)
        return bool(stored and stored[0] == seal)

    def remember(self, nodeid: str, seal: str, *, file_count: int) -> None:
        self._seals[nodeid] = (seal, int(file_count))

    def cached_file_hash(self, path: str) -> tuple[int, int, str] | None:
        return self._files.get(path)

    def remember_file_hash(
        self, path: str, *, mtime_ns: int, size_bytes: int, content_hash: str
    ) -> None:
        self._files[path] = (int(mtime_ns), int(size_bytes), str(content_hash))


class DuckDbQuackSealOracle:
    """DuckDB seal catalog.  Refuses extra-gate ``control.duckdb``."""

    def __init__(self, path: Path) -> None:
        if path.name == "control.duckdb":
            raise PytestSealCatalogError(
                "pytest seal refuses extra-gate exclusive control.duckdb"
            )
        self.path = path
        self.completion_authority = False
        self._ensure()

    def _connect(self):
        import duckdb

        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(str(self.path), config={"threads": 1})
        connection.execute(_SCHEMA)
        return connection

    def _ensure(self) -> None:
        connection = self._connect()
        connection.close()

    def reuse(self, nodeid: str, seal: str) -> bool:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT seal, completion_authority
                FROM pytest_ast_seal
                WHERE nodeid = ?
                """,
                [nodeid],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return False
        return str(row[0]) == seal and row[1] is False

    def remember(self, nodeid: str, seal: str, *, file_count: int) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO pytest_ast_seal (
                    nodeid, seal, file_count, recorded_at, completion_authority
                ) VALUES (?, ?, ?, ?, FALSE)
                ON CONFLICT (nodeid) DO UPDATE SET
                    seal = excluded.seal,
                    file_count = excluded.file_count,
                    recorded_at = excluded.recorded_at,
                    completion_authority = FALSE
                """,
                [nodeid, seal, int(file_count), datetime.now(UTC).isoformat()],
            )
        finally:
            connection.close()

    def cached_file_hash(self, path: str) -> tuple[int, int, str] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT mtime_ns, size_bytes, content_hash
                FROM pytest_ast_file_hash
                WHERE path = ?
                """,
                [path],
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return (int(row[0]), int(row[1]), str(row[2]))

    def remember_file_hash(
        self, path: str, *, mtime_ns: int, size_bytes: int, content_hash: str
    ) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO pytest_ast_file_hash (
                    path, mtime_ns, size_bytes, content_hash
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT (path) DO UPDATE SET
                    mtime_ns = excluded.mtime_ns,
                    size_bytes = excluded.size_bytes,
                    content_hash = excluded.content_hash
                """,
                [path, int(mtime_ns), int(size_bytes), content_hash],
            )
        finally:
            connection.close()


def _oracle_for(config: Any) -> PytestSealOracle:
    injected = getattr(config, "_pytest_ast_seal_oracle", None)
    if injected is not None:
        return injected
    env_path = str(os.environ.get(SEAL_DUCKDB_ENV) or "").strip()
    if env_path:
        path = Path(env_path)
    else:
        meta = str(os.environ.get(META_INDEX_ENV) or "").strip()
        if meta:
            path = Path(meta).with_name("pytest_ast_seal.duckdb")
        else:
            root = Path(getattr(config, "rootpath", Path.cwd()))
            path = root / ".pytest_cache" / "pytest_ast_seal.duckdb"
    if path.name == "control.duckdb":
        raise PytestSealCatalogError(
            "pytest seal refuses extra-gate exclusive control.duckdb"
        )
    return DuckDbQuackSealOracle(path)


def _repo_for(config: Any) -> Path:
    root = getattr(config, "rootpath", None)
    if root is None:
        root = Path.cwd()
    return _repo_root(Path(root))


def _item_path(item: Any) -> Path | None:
    raw = getattr(item, "path", None) or getattr(item, "fspath", None)
    if raw is None:
        return None
    path = Path(os.fspath(raw))
    if path.suffix != ".py" or not path.is_file():
        return None
    return path


_CONFIG_KEY = "_pytest_ast_seal_state"


def pytest_configure(config: Any) -> None:
    """Install the seal once.  Opt-out and catalog failures fail open to running tests."""

    if getattr(config, _CONFIG_KEY, None) is not None:
        return
    try:
        config.addinivalue_line(
            "markers",
            "pytest_seal_opt_out(reason=None): run this test even when its AST seal matches",
        )
    except Exception:
        pass
    if not seal_enabled():
        setattr(config, _CONFIG_KEY, {"enabled": False})
        return
    try:
        oracle = _oracle_for(config)
        repo = _repo_for(config)
    except Exception:
        setattr(config, _CONFIG_KEY, {"enabled": False})
        return
    setattr(
        config,
        _CONFIG_KEY,
        {"enabled": True, "oracle": oracle, "repo": repo, "seals": {}, "wrote": False},
    )


def pytest_collection_modifyitems(config: Any, items: Iterable[Any]) -> None:
    """Skip tests whose injected oracle says the AST seal still matches."""

    state = getattr(config, _CONFIG_KEY, None)
    if not isinstance(state, dict) or not state.get("enabled"):
        return
    oracle: PytestSealOracle = state["oracle"]
    repo: Path = state["repo"]
    closures: dict[Path, tuple[Path, ...] | None] = {}
    try:
        import pytest
    except Exception:
        return
    for item in items:
        if item_opted_out(item):
            continue
        path = _item_path(item)
        if path is None:
            continue
        if path not in closures:
            closures[path] = ast_closure(repo, path)
        files = closures[path]
        if not files:
            continue
        sealed = seal_files(files, oracle)
        if sealed is None or not sealed.complete:
            continue
        nodeid = str(getattr(item, "nodeid", "") or "")
        if not nodeid:
            continue
        state["seals"][nodeid] = sealed
        try:
            matched = bool(oracle.reuse(nodeid, sealed.digest))
        except Exception:
            matched = False
        if matched:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "pytest AST seal matches; "
                        "opt out with IPFS_ACCELERATE_PYTEST_SEAL=0"
                    )
                )
            )


def _remember_passing_report(item: Any, report: Any) -> None:
    """Store a seal only after the call phase really passed."""

    state = getattr(getattr(item, "config", None), _CONFIG_KEY, None)
    if not isinstance(state, dict) or not state.get("enabled") or item_opted_out(item):
        return
    if getattr(report, "when", "") != "call" or not getattr(report, "passed", False):
        return
    sealed = state["seals"].get(str(getattr(item, "nodeid", "") or ""))
    if sealed is None:
        return
    try:
        state["oracle"].remember(
            str(item.nodeid),
            sealed.digest,
            file_count=len(sealed.files),
        )
        state["wrote"] = True
    except Exception:
        return


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """Project one observational AST seal into the meta-index.  Not completion."""

    del exitstatus
    config = getattr(session, "config", None)
    state = getattr(config, _CONFIG_KEY, None)
    if not isinstance(state, dict) or not state.get("wrote"):
        return
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        mirror_work_record(
            catalog_kind="ast",
            record_kind="pytest_ast_seal",
            record_ref="pytest-ast-seal",
            subject_kind="record_cid",
            subject_ref="pytest-ast-seal",
        )
    except Exception:
        return


try:
    import pytest
except Exception:  # pragma: no cover - unit tests may import before pytest
    pass
else:

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(item: Any, call: Any):
        """Remember a real pass.  Skips and failures do not become seals."""

        del call
        outcome = yield
        try:
            report = outcome.get_result()
        except Exception:
            return
        _remember_passing_report(item, report)
