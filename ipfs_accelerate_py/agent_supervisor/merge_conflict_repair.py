"""Deterministic merge-conflict repairs for generated supervisor artifacts."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MarkdownMergeResult:
    text: str
    section_count: int
    duplicate_variant_count: int


def _run_git(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def _safe_relative_path(value: str) -> bool:
    if not value or value.startswith("/") or "\0" in value:
        return False
    return ".." not in Path(value).parts


def _repo_relative(repo_root: Path, path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return ""
    normalized = candidate.as_posix().strip("/")
    return normalized if _safe_relative_path(normalized) else ""


def _normalize_allowed_paths(repo_root: Path, paths: Iterable[str | Path]) -> set[str]:
    allowed: set[str] = set()
    for path in paths:
        relative = _repo_relative(repo_root, path)
        if relative:
            allowed.add(relative)
    return allowed


def _normalize_allowed_dirs(repo_root: Path, dirs: Iterable[str | Path]) -> tuple[str, ...]:
    allowed: list[str] = []
    for path in dirs:
        relative = _repo_relative(repo_root, path)
        if relative:
            allowed.append(relative.rstrip("/") + "/")
    return tuple(dict.fromkeys(allowed))


def _path_allowed(relative: str, *, allowed_paths: set[str], allowed_dirs: tuple[str, ...]) -> bool:
    if not _safe_relative_path(relative):
        return False
    if not relative.endswith((".md", ".markdown")):
        return False
    return relative in allowed_paths or any(relative.startswith(prefix) for prefix in allowed_dirs)


def unmerged_paths(repo_root: Path) -> list[str]:
    result = _run_git(repo_root, ["diff", "--name-only", "--diff-filter=U"])
    if result.returncode != 0:
        return []
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def conflict_stage_blob(repo_root: Path, relative: str, *, stage: int) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f":{stage}:{relative}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _decode_blob(blob: bytes | None) -> str:
    return blob.decode("utf-8", errors="surrogateescape") if blob is not None else ""


def _split_markdown_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    lines = text.splitlines(keepends=True)
    prefix: list[str] = []
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if current_heading:
                sections.append((current_heading, "".join(current)))
            elif current:
                prefix = current
            current_heading = " ".join(line.strip().split())
            current = [line]
            continue
        current.append(line)

    if current_heading:
        sections.append((current_heading, "".join(current)))
    elif current:
        prefix = current
    return "".join(prefix), sections


def _status_rank(block: str) -> int:
    lowered = block.lower()
    if "- status: completed" in lowered:
        return 4
    if "- status: active" in lowered:
        return 3
    if "- status: todo" in lowered:
        return 2
    if "- status: blocked" in lowered:
        return 1
    return 0


def _select_section_variant(existing: str, candidate: str) -> tuple[str, bool]:
    if existing == candidate:
        return existing, False
    if existing and existing in candidate:
        return candidate, True
    if candidate and candidate in existing:
        return existing, True
    existing_rank = _status_rank(existing)
    candidate_rank = _status_rank(candidate)
    if candidate_rank > existing_rank:
        return candidate, True
    if candidate_rank < existing_rank:
        return existing, True
    return (candidate if len(candidate) > len(existing) else existing), True


def merge_append_only_markdown_sections(
    *,
    base_text: str,
    ours_text: str,
    theirs_text: str,
) -> MarkdownMergeResult | None:
    """Union h2 markdown sections from an append-only generated document."""

    base_prefix, base_sections = _split_markdown_sections(base_text)
    ours_prefix, ours_sections = _split_markdown_sections(ours_text)
    theirs_prefix, theirs_sections = _split_markdown_sections(theirs_text)
    if not (base_sections or ours_sections or theirs_sections):
        return None

    prefix = ours_prefix or theirs_prefix or base_prefix
    ordered: dict[str, str] = {}
    duplicate_variant_count = 0
    for _source, sections in (
        ("base", base_sections),
        ("ours", ours_sections),
        ("theirs", theirs_sections),
    ):
        for heading, block in sections:
            if heading not in ordered:
                ordered[heading] = block
                continue
            selected, changed = _select_section_variant(ordered[heading], block)
            ordered[heading] = selected
            duplicate_variant_count += int(changed)

    body = "\n\n".join(block.strip("\n") for block in ordered.values()).rstrip()
    text = (prefix.rstrip() + "\n\n" + body if prefix.strip() else body).rstrip() + "\n"
    return MarkdownMergeResult(
        text=text,
        section_count=len(ordered),
        duplicate_variant_count=duplicate_variant_count,
    )


def resolve_append_only_markdown_conflicts(
    *,
    repo_root: Path,
    allowed_paths: Iterable[str | Path] = (),
    allowed_dirs: Iterable[str | Path] = (),
) -> list[dict[str, object]]:
    """Resolve configured generated markdown conflicts by unioning h2 sections."""

    repo_root = repo_root.resolve()
    normalized_paths = _normalize_allowed_paths(repo_root, allowed_paths)
    normalized_dirs = _normalize_allowed_dirs(repo_root, allowed_dirs)
    results: list[dict[str, object]] = []
    for relative in unmerged_paths(repo_root):
        if not _path_allowed(relative, allowed_paths=normalized_paths, allowed_dirs=normalized_dirs):
            continue
        base = _decode_blob(conflict_stage_blob(repo_root, relative, stage=1))
        ours = _decode_blob(conflict_stage_blob(repo_root, relative, stage=2))
        theirs = _decode_blob(conflict_stage_blob(repo_root, relative, stage=3))
        merged = merge_append_only_markdown_sections(
            base_text=base,
            ours_text=ours,
            theirs_text=theirs,
        )
        if merged is None:
            results.append({"path": relative, "resolved": False, "reason": "no_markdown_sections"})
            continue
        target = repo_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(merged.text, encoding="utf-8", errors="surrogateescape")
        add = _run_git(repo_root, ["add", "--", relative])
        results.append(
            {
                "path": relative,
                "resolved": add.returncode == 0,
                "reason": "append_only_markdown_sections_merged"
                if add.returncode == 0
                else "git_add_failed",
                "returncode": add.returncode,
                "stdout": add.stdout[-4000:],
                "stderr": add.stderr[-4000:],
                "section_count": merged.section_count,
                "duplicate_variant_count": merged.duplicate_variant_count,
            }
        )
    return results
