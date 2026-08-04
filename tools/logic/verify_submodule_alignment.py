#!/usr/bin/env python3
"""Read-only alignment check for the ``ipfs_datasets_py`` submodule.

The checker deliberately uses only local Git metadata.  In particular, it
does not fetch, checkout, reset, clean, stage, or otherwise update any
repository.  ``origin/main`` therefore means the locally available remote
tracking ref; callers that require a fresh network view must fetch before
running this command.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


INTERFACE = "LogicSubmoduleAlignment@1"
DEFAULT_SUBMODULE_PATH = Path("ipfs_datasets_py")
DEFAULT_REQUIRED_LOGIC_MODULES = (
    "ipfs_datasets_py.logic",
    "ipfs_datasets_py.logic.ir_core",
    "ipfs_datasets_py.logic.backends",
)
_GIT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class AlignmentDiagnostic:
    """An actionable reason that alignment could not be established."""

    code: str
    message: str
    remediation: str


@dataclass(frozen=True)
class CheckoutObservation:
    """Read-only observations for an embedded or sibling Git checkout."""

    path: str | None
    available: bool
    head: str | None
    origin_main: str | None
    clean: bool | None
    status: tuple[str, ...] = ()


@dataclass(frozen=True)
class SubmoduleAlignmentReport:
    """Complete ``LogicSubmoduleAlignment@1`` observation and decision."""

    interface: str
    aligned: bool
    parent_path: str
    parent_commit: str | None
    submodule_path: str
    gitlink: str | None
    embedded: CheckoutObservation
    sibling: CheckoutObservation
    required_logic_modules: dict[str, bool]
    diagnostics: tuple[AlignmentDiagnostic, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class _GitResult:
    returncode: int
    stdout: str
    stderr: str


def _git(repository: Path, *arguments: str) -> _GitResult:
    """Run one bounded, read-only Git query without optional index locking."""

    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return _GitResult(127, "", str(error))
    return _GitResult(
        completed.returncode,
        completed.stdout,
        completed.stderr,
    )


def _commit(repository: Path, reference: str) -> str | None:
    result = _git(repository, "rev-parse", "--verify", f"{reference}^{{commit}}")
    if result.returncode != 0:
        return None
    value = result.stdout.strip().lower()
    if len(value) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in value
    ):
        return None
    return value


def _is_git_repository(repository: Path) -> bool:
    result = _git(repository, "rev-parse", "--is-inside-work-tree")
    return result.returncode == 0 and result.stdout.strip() == "true"


def _gitlink(parent: Path, parent_commit: str, relative_path: str) -> str | None:
    result = _git(parent, "ls-tree", "-z", parent_commit, "--", relative_path)
    if result.returncode != 0:
        return None
    for entry in result.stdout.rstrip("\0").split("\0"):
        metadata, separator, observed_path = entry.partition("\t")
        fields = metadata.split()
        if (
            separator
            and observed_path == relative_path
            and len(fields) == 3
            and fields[0] == "160000"
            and fields[1] == "commit"
        ):
            value = fields[2].lower()
            if len(value) in {40, 64} and all(
                character in "0123456789abcdef" for character in value
            ):
                return value
    return None


def _status(repository: Path) -> tuple[bool | None, tuple[str, ...]]:
    result = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if result.returncode != 0:
        return None, ()
    entries = tuple(line for line in result.stdout.splitlines() if line)
    return not entries, entries


def _observe_checkout(path: Path | None) -> CheckoutObservation:
    if path is None:
        return CheckoutObservation(
            path=None,
            available=False,
            head=None,
            origin_main=None,
            clean=None,
        )
    resolved = path.expanduser().resolve()
    if not resolved.is_dir() or not _is_git_repository(resolved):
        return CheckoutObservation(
            path=str(resolved),
            available=False,
            head=None,
            origin_main=None,
            clean=None,
        )
    clean, status = _status(resolved)
    return CheckoutObservation(
        path=str(resolved),
        available=True,
        head=_commit(resolved, "HEAD"),
        origin_main=_commit(resolved, "refs/remotes/origin/main"),
        clean=clean,
        status=status,
    )


def _valid_relative_path(path: Path) -> bool:
    return (
        not path.is_absolute()
        and bool(path.parts)
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _module_available(checkout: Path, module_name: str) -> bool:
    """Check module source availability without importing or executing it."""

    parts = module_name.split(".")
    if not parts or any(not part.isidentifier() for part in parts):
        return False
    module_path = checkout.joinpath(*parts)
    module_file = module_path.with_suffix(".py")
    if module_file.is_file():
        return True
    if not module_path.is_dir():
        return False
    if (module_path / "__init__.py").is_file():
        return True
    # PEP 420 namespace packages need no __init__.py.  Requiring at least one
    # Python child prevents an unrelated empty directory from satisfying the
    # availability check.
    return any(
        child.is_file() and child.suffix == ".py"
        for child in module_path.iterdir()
    )


def discover_sibling_repository(
    parent_repo: Path,
    *,
    submodule_name: str = DEFAULT_SUBMODULE_PATH.name,
    embedded_path: Path | None = None,
) -> Path | None:
    """Find a sibling checkout beside the primary parent worktree, if present."""

    # Supervisor worktrees are nested below a coordination repository whose
    # top level contains the independently checked-out datasets sibling.
    # Prefer the nearest such ancestor.  Shared Git common directories can
    # contain unrelated worktrees, so their neighbors are only fallbacks.
    candidates = [
        ancestor / submodule_name for ancestor in parent_repo.parents
    ]
    worktrees = _git(parent_repo, "worktree", "list", "--porcelain")
    if worktrees.returncode == 0:
        for line in worktrees.stdout.splitlines():
            prefix = "worktree "
            if line.startswith(prefix):
                candidates.append(Path(line[len(prefix) :]).parent / submodule_name)

    embedded_resolved = embedded_path.resolve() if embedded_path is not None else None
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen or resolved == embedded_resolved:
            continue
        seen.add(resolved)
        if resolved.is_dir() and _is_git_repository(resolved):
            return resolved
    return None


def verify_submodule_alignment(
    parent_repo: str | Path,
    *,
    submodule_path: str | Path = DEFAULT_SUBMODULE_PATH,
    sibling_repo: str | Path | None = None,
    discover_sibling: bool = True,
    required_modules: Sequence[str] = DEFAULT_REQUIRED_LOGIC_MODULES,
) -> SubmoduleAlignmentReport:
    """Observe and verify parent, embedded, remote-main, and sibling alignment."""

    parent = Path(parent_repo).expanduser().resolve()
    relative = Path(submodule_path)
    diagnostics: list[AlignmentDiagnostic] = []

    if not _valid_relative_path(relative):
        diagnostics.append(
            AlignmentDiagnostic(
                "invalid_submodule_path",
                f"submodule path must be a safe relative path: {relative}",
                "Pass a repository-relative submodule path without '.' or '..'.",
            )
        )
        relative = DEFAULT_SUBMODULE_PATH

    parent_commit = _commit(parent, "HEAD") if _is_git_repository(parent) else None
    if parent_commit is None:
        diagnostics.append(
            AlignmentDiagnostic(
                "parent_repository_unavailable",
                f"parent checkout is not a readable Git worktree: {parent}",
                "Run the checker from an initialized parent repository checkout.",
            )
        )

    relative_text = relative.as_posix()
    gitlink = (
        _gitlink(parent, parent_commit, relative_text)
        if parent_commit is not None
        else None
    )
    if parent_commit is not None and gitlink is None:
        diagnostics.append(
            AlignmentDiagnostic(
                "gitlink_unavailable",
                f"{relative_text} is not a submodule gitlink in parent HEAD",
                f"Commit the intended {relative_text} gitlink in the parent branch.",
            )
        )

    embedded_path = parent / relative
    embedded = _observe_checkout(embedded_path)
    if not embedded.available:
        diagnostics.append(
            AlignmentDiagnostic(
                "embedded_checkout_unavailable",
                f"embedded submodule checkout is unavailable: {embedded.path}",
                f"Initialize {relative_text} at the parent-recorded gitlink.",
            )
        )
    else:
        if embedded.head is None:
            diagnostics.append(
                AlignmentDiagnostic(
                    "embedded_head_unavailable",
                    "embedded checkout has no readable HEAD commit",
                    "Restore the embedded checkout to a valid commit.",
                )
            )
        if embedded.origin_main is None:
            diagnostics.append(
                AlignmentDiagnostic(
                    "embedded_origin_main_unavailable",
                    "embedded checkout has no local refs/remotes/origin/main",
                    "Fetch origin/main explicitly before running this read-only check.",
                )
            )
        if embedded.clean is not True:
            diagnostics.append(
                AlignmentDiagnostic(
                    "embedded_checkout_dirty",
                    (
                        "embedded checkout contains tracked, untracked, "
                        "or nested-submodule changes"
                    ),
                    "Commit, stash, or remove the reported embedded checkout changes.",
                )
            )

    if gitlink is not None and embedded.head is not None and gitlink != embedded.head:
        diagnostics.append(
            AlignmentDiagnostic(
                "gitlink_embedded_head_mismatch",
                f"parent gitlink {gitlink} differs from embedded HEAD {embedded.head}",
                (
                    "Checkout the parent-recorded gitlink, or intentionally "
                    "update and commit the parent gitlink."
                ),
            )
        )
    if (
        gitlink is not None
        and embedded.origin_main is not None
        and gitlink != embedded.origin_main
    ):
        diagnostics.append(
            AlignmentDiagnostic(
                "gitlink_origin_main_mismatch",
                (
                    f"parent gitlink {gitlink} differs from embedded "
                    f"origin/main {embedded.origin_main}"
                ),
                (
                    "Publish the intended datasets commit and update the parent "
                    "gitlink, or restore the expected local origin/main ref."
                ),
            )
        )

    explicit_sibling = Path(sibling_repo) if sibling_repo is not None else None
    sibling_path = explicit_sibling
    if sibling_path is None and discover_sibling:
        sibling_path = discover_sibling_repository(
            parent,
            submodule_name=relative.name,
            embedded_path=embedded_path,
        )
    sibling = _observe_checkout(sibling_path)
    if explicit_sibling is not None and not sibling.available:
        diagnostics.append(
            AlignmentDiagnostic(
                "sibling_checkout_unavailable",
                f"requested sibling checkout is unavailable: {sibling.path}",
                (
                    "Pass an initialized sibling datasets checkout, or omit "
                    "--sibling to use discovery."
                ),
            )
        )
    if sibling.available:
        if sibling.head is None:
            diagnostics.append(
                AlignmentDiagnostic(
                    "sibling_head_unavailable",
                    "sibling checkout has no readable HEAD commit",
                    "Restore the sibling checkout to a valid commit.",
                )
            )
        if sibling.origin_main is None:
            diagnostics.append(
                AlignmentDiagnostic(
                    "sibling_origin_main_unavailable",
                    "sibling checkout has no local refs/remotes/origin/main",
                    "Fetch origin/main explicitly in the sibling checkout.",
                )
            )
        if sibling.clean is not True:
            diagnostics.append(
                AlignmentDiagnostic(
                    "sibling_checkout_dirty",
                    (
                        "sibling checkout contains tracked, untracked, "
                        "or nested-submodule changes"
                    ),
                    "Commit, stash, or remove the reported sibling checkout changes.",
                )
            )
        if gitlink is not None and sibling.head is not None and gitlink != sibling.head:
            diagnostics.append(
                AlignmentDiagnostic(
                    "gitlink_sibling_head_mismatch",
                    f"parent gitlink {gitlink} differs from sibling HEAD {sibling.head}",
                    (
                        "Align the sibling checkout with the published commit "
                        "recorded by the parent gitlink."
                    ),
                )
            )
        if (
            gitlink is not None
            and sibling.origin_main is not None
            and gitlink != sibling.origin_main
        ):
            diagnostics.append(
                AlignmentDiagnostic(
                    "gitlink_sibling_origin_main_mismatch",
                    (
                        f"parent gitlink {gitlink} differs from sibling "
                        f"origin/main {sibling.origin_main}"
                    ),
                    (
                        "Fetch or publish the intended sibling origin/main, "
                        "then align the parent gitlink."
                    ),
                )
            )

    module_availability = {
        module_name: embedded.available
        and _module_available(embedded_path, module_name)
        for module_name in dict.fromkeys(required_modules)
    }
    for module_name, available in module_availability.items():
        if not available:
            diagnostics.append(
                AlignmentDiagnostic(
                    "required_logic_module_unavailable",
                    f"required logic module source is unavailable: {module_name}",
                    "Use a published datasets revision that contains the required logic module.",
                )
            )

    return SubmoduleAlignmentReport(
        interface=INTERFACE,
        aligned=not diagnostics,
        parent_path=str(parent),
        parent_commit=parent_commit,
        submodule_path=relative_text,
        gitlink=gitlink,
        embedded=embedded,
        sibling=sibling,
        required_logic_modules=module_availability,
        diagnostics=tuple(diagnostics),
    )


def _display(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _display_status(observation: CheckoutObservation) -> str:
    if not observation.available:
        return "unavailable"
    return ", ".join(observation.status) or "clean"


def render_text(report: SubmoduleAlignmentReport) -> str:
    """Render every required observation plus actionable diagnostics."""

    lines = [
        f"{report.interface}: {'PASS' if report.aligned else 'FAIL'}",
        f"parent path: {report.parent_path}",
        f"parent commit: {_display(report.parent_commit)}",
        f"gitlink ({report.submodule_path}): {_display(report.gitlink)}",
        f"embedded path: {_display(report.embedded.path)}",
        f"embedded HEAD: {_display(report.embedded.head)}",
        f"embedded origin/main: {_display(report.embedded.origin_main)}",
        f"embedded clean: {_display(report.embedded.clean)}",
        f"embedded status: {_display_status(report.embedded)}",
        f"sibling path: {_display(report.sibling.path)}",
        f"sibling HEAD: {_display(report.sibling.head)}",
        f"sibling origin/main: {_display(report.sibling.origin_main)}",
        f"sibling clean: {_display(report.sibling.clean)}",
        f"sibling status: {_display_status(report.sibling)}",
    ]
    lines.extend(
        f"required module {module_name}: {_display(available)}"
        for module_name, available in report.required_logic_modules.items()
    )
    for diagnostic in report.diagnostics:
        lines.append(f"[{diagnostic.code}] {diagnostic.message}")
        lines.append(f"  action: {diagnostic.remediation}")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify ipfs_datasets_py gitlink, checkout, local origin/main, "
            "sibling, cleanliness, and required logic-module alignment without mutation."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="parent repository root (default: root containing this tool)",
    )
    parser.add_argument(
        "--submodule",
        type=Path,
        default=DEFAULT_SUBMODULE_PATH,
        help="parent-relative datasets submodule path",
    )
    parser.add_argument(
        "--sibling",
        type=Path,
        help="explicit sibling datasets checkout (otherwise auto-discovered)",
    )
    parser.add_argument(
        "--skip-sibling",
        action="store_true",
        help="disable optional sibling-checkout discovery",
    )
    parser.add_argument(
        "--required-module",
        action="append",
        dest="required_modules",
        help="required dotted module source; repeat to replace the default set",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the structured LogicSubmoduleAlignment@1 JSON report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.  Exit zero only when every available check aligns."""

    arguments = _parser().parse_args(argv)
    report = verify_submodule_alignment(
        arguments.repo_root,
        submodule_path=arguments.submodule,
        sibling_repo=arguments.sibling,
        discover_sibling=not arguments.skip_sibling,
        required_modules=(
            arguments.required_modules
            if arguments.required_modules is not None
            else DEFAULT_REQUIRED_LOGIC_MODULES
        ),
    )
    if arguments.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 0 if report.aligned else 1


if __name__ == "__main__":
    sys.exit(main())
