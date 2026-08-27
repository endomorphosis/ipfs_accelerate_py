#!/usr/bin/env python3
"""Deterministically materialize the owner-exact VRIF-030 benchmark artifacts.

The Portal baseline is an explicit input, never inferred from a branch name or
from an existing manifest.  Check and dry-run modes are read-only.  Write mode
may replace only the two generated benchmark artifacts and the two bounded
baseline identity literals in the declared benchmark test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.benchmark import (  # noqa: E402
    MANIFEST_SCHEMA,
    build_frozen_benchmark_contract,
    sha256_identity,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (  # noqa: E402
    PROGRAM_ID,
    ResidualIntelligenceError,
    ResidualTaskFamily,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (  # noqa: E402
    ControlPlaneContractError,
    content_identity,
)

MANIFEST_PATH: Final = Path(
    "benchmarks/agent_supervisor/residual_intelligence/manifest.json"
)
CASES_PATH: Final = Path(
    "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl"
)
TEST_PATH: Final = Path("test/api/residual_intelligence/test_benchmark.py")
ADMISSION_PATH: Final = Path(
    "benchmarks/agent_supervisor/residual_intelligence/"
    "synthetic_training_admission.json"
)
SPLIT_PATH: Final = Path(
    "benchmarks/agent_supervisor/residual_intelligence/" "synthetic_split_manifest.json"
)
OBJECTIVE_PATHS: Final = (
    Path("docs/architecture/agent_supervisor_residual_intelligence.objectives.md"),
    Path("docs/architecture/agent_supervisor_residual_intelligence.todo.md"),
)
OPERATION_PATH: Final = Path(
    "ipfs_accelerate_py/agent_supervisor/control/control_plane.py"
)
PROVIDER_PATH: Final = Path(
    "config/agent_supervisor_residual_intelligence_scheduler.json"
)
INVENTORY_PATH: Final = Path(
    "docs/architecture/residual_intelligence_inventory/"
    "residual_model_call_inventory.json"
)
VALIDATION_COMMAND: Final = (
    "python3 -m pytest -q test/api/residual_intelligence/test_benchmark.py"
)
MAX_INPUT_BYTES: Final = 8 * 1024 * 1024
GIT_EXECUTABLE: Final = "/usr/bin/git"
_GIT_OBJECT_RE: Final = re.compile(r"[0-9a-f]{40}\Z")
_MARKER_BEGIN: Final = b"# BEGIN VRIF-030 PORTAL BASELINE (materializer-owned)"
_MARKER_END: Final = b"# END VRIF-030 PORTAL BASELINE (materializer-owned)"
_MARKER_RE: Final = re.compile(
    rb"(?m)^# BEGIN VRIF-030 PORTAL BASELINE \(materializer-owned\)\n"
    rb'VRIF_PORTAL_BASELINE_COMMIT = "([0-9a-f]{40})"\n'
    rb'VRIF_PORTAL_BASELINE_TREE = "([0-9a-f]{40})"\n'
    rb"# END VRIF-030 PORTAL BASELINE \(materializer-owned\)$"
)
_MUTABLE_PATHS: Final = frozenset(
    {MANIFEST_PATH.as_posix(), CASES_PATH.as_posix(), TEST_PATH.as_posix()}
)
_INPUT_PATHS: Final = (
    *OBJECTIVE_PATHS,
    OPERATION_PATH,
    PROVIDER_PATH,
    ADMISSION_PATH,
    SPLIT_PATH,
    INVENTORY_PATH,
    TEST_PATH,
)


class MaterializationError(RuntimeError):
    """The requested materialization cannot be derived unambiguously."""


def _git(repo_root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            [
                GIT_EXECUTABLE,
                "-c",
                "core.fsmonitor=false",
                "-c",
                f"core.hooksPath={os.devnull}",
                "-c",
                "diff.external=",
                *args,
            ],
            cwd=repo_root,
            capture_output=True,
            check=False,
            env={
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "HOME": "/nonexistent",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
            },
            shell=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MaterializationError(
            "Git is unavailable for VRIF-030 materialization"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise MaterializationError(
            f"Git command failed ({' '.join(args)}): {detail or 'no diagnostic'}"
        )
    return bytes(completed.stdout)


def _git_text(repo_root: Path, *args: str) -> str:
    try:
        return _git(repo_root, *args).decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise MaterializationError("Git returned a non-ASCII object identity") from exc


def _repository_root(value: Path | str) -> Path:
    unresolved = Path(value)
    try:
        requested_metadata = unresolved.lstat()
        requested = unresolved.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MaterializationError(
            f"repository root is unavailable: {unresolved}"
        ) from exc
    if stat.S_ISLNK(requested_metadata.st_mode) or not stat.S_ISDIR(
        requested_metadata.st_mode
    ):
        raise MaterializationError(f"repository root is not a directory: {unresolved}")
    try:
        observed = Path(
            _git(requested, "rev-parse", "--show-toplevel").decode("utf-8").strip()
        ).resolve()
    except UnicodeDecodeError as exc:
        raise MaterializationError("Git repository root is not UTF-8") from exc
    if observed != requested:
        raise MaterializationError(
            f"repository root must be the exact Git worktree root: {requested}"
        )
    return requested


def _assert_no_executable_repository_git_config(repo_root: Path) -> None:
    """Reject repository/worktree Git configuration that can execute code."""

    try:
        completed = subprocess.run(
            [GIT_EXECUTABLE, "-C", str(repo_root), "config", "--null", "--list"],
            capture_output=True,
            check=False,
            env={
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "HOME": "/nonexistent",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
            },
            shell=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MaterializationError("Git configuration is unavailable") from exc
    raw = bytes(completed.stdout or b"")
    if completed.returncode != 0 or len(raw) > 1024 * 1024:
        raise MaterializationError("Git configuration cannot be bounded")
    for record in raw.split(b"\0"):
        if not record:
            continue
        raw_key, separator, _value = record.partition(b"\n")
        if not separator:
            raise MaterializationError("Git configuration is malformed")
        key = raw_key.decode("utf-8", errors="replace").lower()
        executable = bool(
            key in {"core.fsmonitor", "core.hookspath", "diff.external"}
            or (
                key.startswith("diff.")
                and key.rsplit(".", 1)[-1] in {"command", "textconv"}
            )
            or (
                key.startswith("filter.")
                and key.rsplit(".", 1)[-1] in {"clean", "smudge", "process"}
            )
        )
        if executable:
            raise MaterializationError(
                f"repository Git configuration may execute code: {key}"
            )


def _assert_no_replacement_refs(repo_root: Path) -> None:
    refs = _git(
        repo_root,
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace/",
    )
    if refs.strip():
        raise MaterializationError("Git replacement refs are forbidden")


def _regular_path_beneath(repo_root: Path, relative: Path) -> Path:
    """Return a regular in-repository path with no symlink component."""

    path_text = relative.as_posix()
    if (
        not relative.parts
        or relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
        or "\\" in path_text
        or Path(path_text).as_posix() != path_text
    ):
        raise MaterializationError(f"owner path is unsafe: {path_text!r}")
    try:
        root = repo_root.resolve(strict=True)
        root_metadata = root.lstat()
    except (OSError, RuntimeError) as exc:
        raise MaterializationError("repository root is unavailable") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise MaterializationError("repository root is not a regular directory")

    current = root
    for index, component in enumerate(relative.parts):
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise MaterializationError(f"owner path is unavailable: {path_text}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise MaterializationError(
                f"owner path is not a regular symlink-free path: {path_text}"
            )
        if index < len(relative.parts) - 1:
            if not stat.S_ISDIR(metadata.st_mode):
                raise MaterializationError(
                    f"owner path parent is not a directory: {path_text}"
                )
        elif not stat.S_ISREG(metadata.st_mode):
            raise MaterializationError(f"owner path is not a regular file: {path_text}")
    try:
        current.resolve(strict=True).relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MaterializationError(
            f"owner path escapes the repository: {path_text}"
        ) from exc
    return current


def _nul_paths(raw: bytes, *, noun: str) -> set[str]:
    result: set[str] = set()
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            path = item.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MaterializationError(f"{noun} contains a non-UTF-8 path") from exc
        if not path or path.startswith("/") or "\\" in path or ".." in Path(path).parts:
            raise MaterializationError(f"{noun} contains an unsafe path: {path!r}")
        result.add(path)
    return result


def _resolve_baseline(repo_root: Path, baseline_commit: str) -> tuple[str, str]:
    if _GIT_OBJECT_RE.fullmatch(baseline_commit) is None:
        raise MaterializationError(
            "baseline commit must be an explicit lowercase 40-hex identity"
        )
    resolved = _git_text(
        repo_root, "rev-parse", "--verify", f"{baseline_commit}^{{commit}}"
    )
    if resolved != baseline_commit:
        raise MaterializationError(
            "baseline commit did not resolve to its exact identity"
        )
    tree = _git_text(repo_root, "rev-parse", "--verify", f"{baseline_commit}^{{tree}}")
    if _GIT_OBJECT_RE.fullmatch(tree) is None:
        raise MaterializationError("baseline tree did not resolve to a 40-hex identity")
    head = _git_text(repo_root, "rev-parse", "--verify", "HEAD^{commit}")
    if _GIT_OBJECT_RE.fullmatch(head) is None:
        raise MaterializationError("candidate HEAD is not a commit")
    try:
        _git(repo_root, "merge-base", "--is-ancestor", baseline_commit, head)
    except MaterializationError as exc:
        raise MaterializationError(
            "baseline commit must be an ancestor of the candidate HEAD"
        ) from exc
    return baseline_commit, tree


def _verify_candidate_scope(repo_root: Path, baseline_commit: str) -> None:
    try:
        _git(
            repo_root,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--quiet",
            "--diff-filter=U",
            "--",
        )
    except MaterializationError as exc:
        raise MaterializationError(
            "candidate workspace contains unmerged paths"
        ) from exc
    changed = _nul_paths(
        _git(
            repo_root,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "--name-only",
            "-z",
            baseline_commit,
            "--",
        ),
        noun="candidate diff",
    )
    untracked = _nul_paths(
        _git(repo_root, "ls-files", "--others", "--exclude-standard", "-z"),
        noun="candidate untracked set",
    )
    unauthorized = sorted((changed | untracked) - _MUTABLE_PATHS)
    if unauthorized:
        raise MaterializationError(
            "candidate workspace is dirty outside the VRIF-030 outputs: "
            + ", ".join(unauthorized)
        )


def _read_regular_tracked_blob(repo_root: Path, relative: Path) -> bytes:
    path_text = relative.as_posix()
    stage = _git(repo_root, "ls-files", "--stage", "-z", "--", path_text)
    try:
        stage_text = stage.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MaterializationError(
            f"tracked metadata is not UTF-8: {path_text}"
        ) from exc
    match = re.fullmatch(r"(100644|100755) [0-9a-f]{40} 0\t([^\0]+)\0", stage_text)
    if match is None or match.group(2) != path_text:
        raise MaterializationError(
            f"owner input is absent, conflicted, or not a regular tracked file: {path_text}"
        )
    absolute = _regular_path_beneath(repo_root, relative)
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise MaterializationError(f"owner input is unavailable: {path_text}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_INPUT_BYTES:
        raise MaterializationError(
            f"owner input is not a bounded regular file: {path_text}"
        )
    try:
        payload = absolute.read_bytes()
    except OSError as exc:
        raise MaterializationError(f"owner input cannot be read: {path_text}") from exc
    if len(payload) != metadata.st_size:
        raise MaterializationError(f"owner input changed during read: {path_text}")
    return payload


def _strict_json_object(raw: bytes, *, noun: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MaterializationError(f"{noun} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"noncanonical JSON constant {value!r}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MaterializationError(f"{noun} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"{noun} must be a JSON object")
    return value


def _rewrite_test_marker(raw: bytes, *, commit: str, tree: str) -> bytes:
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MaterializationError("benchmark test is not UTF-8") from exc
    if raw.count(_MARKER_BEGIN) != 1 or raw.count(_MARKER_END) != 1:
        raise MaterializationError(
            "benchmark test must contain exactly one delimited Portal baseline marker"
        )
    matches = tuple(_MARKER_RE.finditer(raw))
    if len(matches) != 1:
        raise MaterializationError(
            "benchmark test Portal baseline marker is malformed or duplicated"
        )
    marker = matches[0]
    rewritten = bytearray(raw)
    rewritten[marker.start(1) : marker.end(1)] = commit.encode("ascii")
    rewritten[marker.start(2) : marker.end(2)] = tree.encode("ascii")
    return bytes(rewritten)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MaterializationError(
            "owner benchmark contains noncanonical JSON"
        ) from exc


def build_materialization(
    repo_root: Path | str,
    *,
    baseline_commit: str,
) -> dict[str, Any]:
    """Return exact candidate bytes without mutating the repository."""

    root = _repository_root(repo_root)
    _assert_no_executable_repository_git_config(root)
    _assert_no_replacement_refs(root)
    commit, tree = _resolve_baseline(root, baseline_commit)
    _verify_candidate_scope(root, commit)
    blobs = {path: _read_regular_tracked_blob(root, path) for path in _INPUT_PATHS}
    current_outputs = {
        path: (
            blobs[path] if path == TEST_PATH else _read_regular_tracked_blob(root, path)
        )
        for path in (TEST_PATH, CASES_PATH, MANIFEST_PATH)
    }
    final_test = _rewrite_test_marker(blobs[TEST_PATH], commit=commit, tree=tree)

    admission = _strict_json_object(blobs[ADMISSION_PATH], noun="training admission")
    split = _strict_json_object(blobs[SPLIT_PATH], noun="split manifest")
    admission_body = dict(admission)
    admission_id = admission_body.pop("admission_id", None)
    try:
        expected_admission_id = content_identity(admission_body)
    except ControlPlaneContractError as exc:
        raise MaterializationError("training admission body is not canonical") from exc
    if type(admission_id) is not str or admission_id != expected_admission_id:
        raise MaterializationError("training admission identity does not verify")

    objective_artifacts = {
        path.as_posix(): sha256_identity(blobs[path]) for path in OBJECTIVE_PATHS
    }
    base_bindings = {
        "repository_states": sha256_identity({"commit": commit, "tree": tree}),
        "objective_revisions": sha256_identity(
            {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "residual-benchmark-objective-revisions@1"
                ),
                "artifacts": objective_artifacts,
            }
        ),
        "operation_catalog": sha256_identity(blobs[OPERATION_PATH]),
        "provider_policy": sha256_identity(blobs[PROVIDER_PATH]),
        "tokenizer": sha256_identity(
            {
                "admission_id": admission_id,
                "disposition": "no_learned_tokenizer_admitted",
            }
        ),
        "model_versions": sha256_identity(
            {
                "inventory_blob_identity": sha256_identity(blobs[INVENTORY_PATH]),
                "disposition": "training_unavailable",
            }
        ),
        "validation_policy": sha256_identity(
            {
                "argv": [[VALIDATION_COMMAND]],
                "test_blob_identity": sha256_identity(final_test),
            }
        ),
    }
    task_families = [family.value for family in ResidualTaskFamily]
    try:
        contract = build_frozen_benchmark_contract(
            task_families=task_families,
            source_commit=commit,
            source_tree=tree,
            split_root=str(split.get("split_root") or ""),
            base_bindings=base_bindings,
        )
    except ResidualIntelligenceError as exc:
        raise MaterializationError("owner benchmark reconstruction failed") from exc
    if len(contract.get("cases", ())) != 96:
        raise MaterializationError("owner benchmark did not produce exactly 96 cases")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "program_identifier": PROGRAM_ID,
        "status": "staged_not_qualified",
        "owner_task": "VRIF-030",
        "source_revision": commit,
        "partitions": contract["partitions"],
        "required_case_kinds": contract["case_kinds"],
        "task_families": task_families,
        "training_admission": "training_unavailable",
        "weights_committed": False,
        "large_corpus_committed": False,
        "promotion_evidence": False,
        "benchmark_freeze": contract["benchmark_freeze"],
    }
    manifest_bytes = _canonical_json(manifest) + b"\n"
    cases_bytes = b"".join(_canonical_json(case) + b"\n" for case in contract["cases"])
    expected = {
        TEST_PATH.as_posix(): final_test,
        CASES_PATH.as_posix(): cases_bytes,
        MANIFEST_PATH.as_posix(): manifest_bytes,
    }
    changed_paths = [
        path
        for path, payload in expected.items()
        if current_outputs[Path(path)] != payload
    ]
    return {
        "repository_root": root,
        "baseline_commit": commit,
        "baseline_tree": tree,
        "expected": expected,
        "changed_paths": changed_paths,
        "case_count": len(contract["cases"]),
        "case_root": contract["benchmark_freeze"]["case_root"],
        "binding_set_id": contract["benchmark_freeze"]["binding_set_id"],
        "freeze_id": contract["benchmark_freeze"]["freeze_id"],
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise MaterializationError(f"output is unavailable: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise MaterializationError(f"output is not a regular file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, stat.S_IMODE(metadata.st_mode))
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_materialization(materialization: Mapping[str, Any]) -> tuple[str, ...]:
    """Atomically replace only changed declared outputs, manifest last."""

    root = materialization.get("repository_root")
    expected = materialization.get("expected")
    changed = materialization.get("changed_paths")
    if (
        not isinstance(root, Path)
        or not isinstance(expected, Mapping)
        or not isinstance(changed, list)
    ):
        raise MaterializationError("materialization result is malformed")
    if set(expected) != _MUTABLE_PATHS or any(
        path not in _MUTABLE_PATHS for path in changed
    ):
        raise MaterializationError(
            "materialization result exceeds VRIF-030 output scope"
        )
    ordered = (TEST_PATH.as_posix(), CASES_PATH.as_posix(), MANIFEST_PATH.as_posix())
    for relative in ordered:
        if relative in changed:
            payload = expected.get(relative)
            if not isinstance(payload, bytes):
                raise MaterializationError(
                    f"materialization payload is invalid: {relative}"
                )
            output_path = _regular_path_beneath(root, Path(relative))
            _atomic_write(output_path, payload)
    return tuple(relative for relative in ordered if relative in changed)


def _summary(materialization: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    expected = materialization["expected"]
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/vrif-frozen-benchmark-materialization@1",
        "mode": mode,
        "baseline_commit": materialization["baseline_commit"],
        "baseline_tree": materialization["baseline_tree"],
        "changed_paths": list(materialization["changed_paths"]),
        "case_count": materialization["case_count"],
        "case_root": materialization["case_root"],
        "binding_set_id": materialization["binding_set_id"],
        "freeze_id": materialization["freeze_id"],
        "output_identities": {
            path: "sha256:" + hashlib.sha256(payload).hexdigest()
            for path, payload in sorted(expected.items())
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-commit",
        required=True,
        help="Exact lowercase 40-hex Portal baseline commit.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Exact Git worktree root (default: this script's repository).",
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--check", action="store_true", help="Fail if canonical bytes differ."
    )
    modes.add_argument(
        "--dry-run",
        action="store_true",
        help="Report deterministic changes without writing.",
    )
    modes.add_argument(
        "--write", action="store_true", help="Atomically write declared outputs."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    mode = "write" if arguments.write else "check" if arguments.check else "dry-run"
    try:
        materialization = build_materialization(
            arguments.repo_root,
            baseline_commit=arguments.baseline_commit,
        )
        if arguments.write:
            write_materialization(materialization)
        summary = _summary(materialization, mode=mode)
    except (MaterializationError, OSError) as exc:
        print(f"VRIF-030 materialization rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    if arguments.check and materialization["changed_paths"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
