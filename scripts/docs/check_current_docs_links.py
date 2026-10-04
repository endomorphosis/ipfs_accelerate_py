#!/usr/bin/env python3
"""Fail-closed relative-link checker for the maintained documentation surface.

This is intentionally **allowlisted**, not a full-tree crawl. Historical,
plan, archive, and generated trees are out of scope so the gate stays
actionable and offline-safe.

Checks:
  - Markdown links ``[text](target)`` in allowlisted files
  - Relative targets resolve to an existing path under the repository root
  - Optional ``#fragment`` anchors resolve to a heading/id when the target is
    a Markdown file under the repo (best-effort GitHub-style slug)

Skips:
  - Absolute URLs (http/https/ftp/mailto/tel/data/javascript)
  - Empty targets and pure whitespace
  - Fenced code blocks (examples may cite fictional paths)
  - Inline code spans
  - Autolinks and bare URLs outside Markdown link syntax

Exit codes:
  0 — all allowlisted links resolve
  1 — one or more broken links or missing allowlist roots
  2 — usage / internal error
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Maintained navigation + Current/Reference surfaces named in
# docs/development/DOCUMENTATION_CURRENT_STATE.md and docs/INDEX.md.
# Keep this list explicit so scope growth is reviewable.
ALLOWLIST: tuple[str, ...] = (
    # Top-level entrypoints
    "README.md",
    "docs/README.md",
    "docs/INDEX.md",
    # Governance / current-state
    "docs/development/DOCUMENTATION_CURRENT_STATE.md",
    "docs/development/DOCUMENTATION_LIFECYCLE.md",
    "docs/development/DOCUMENTATION_MAINTENANCE.md",
    "docs/development/DOCUMENTATION_MANIFEST.md",
    "docs/development/testing.md",
    # Getting started / product journeys
    "docs/guides/getting-started/README.md",
    "docs/guides/getting-started/installation.md",
    "docs/guides/QUICKSTART.md",
    "docs/guides/cli/README_CLI.md",
    "docs/guides/MCP_SETUP_GUIDE.md",
    "docs/guides/AGENT_SUPERVISOR_GUIDE.md",
    "docs/guides/deployment/README.md",
    "docs/guides/hardware/overview.md",
    "docs/guides/p2p/README.md",
    "docs/guides/troubleshooting/faq.md",
    "docs/api/overview.md",
    "docs/MCP_SERVER.md",
    # Architecture (Current product + supervisor hubs)
    "docs/architecture/README.md",
    "docs/architecture/overview.md",
    "docs/architecture/SYSTEM_CONTEXT.md",
    "docs/architecture/INFERENCE_RUNTIME.md",
    "docs/architecture/MODEL_SERVICE_ROUTING.md",
    "docs/architecture/MCP_RUNTIME.md",
    "docs/architecture/DISTRIBUTED_RUNTIME.md",
    "docs/architecture/INTEGRATION_BOUNDARIES.md",
    "docs/architecture/GUIDE_CONVENTIONS.md",
    "docs/architecture/GLOSSARY.md",
    "docs/architecture/AI_SERVICE_CATALOG.md",
    "docs/architecture/AGENT_SUPERVISOR_PHILOSOPHY.md",
    "docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md",
    "docs/architecture/agent_supervisor/README.md",
    "docs/architecture/agent_supervisor/CONTROL_PLANE.md",
    "docs/architecture/agent_supervisor/PLANNING_AND_ASSURANCE.md",
    "docs/architecture/agent_supervisor/EXECUTION_AND_RECOVERY.md",
    "docs/architecture/agent_supervisor/PROMPT_FIRST_RUNTIME.md",
    "docs/architecture/agent_supervisor/DEVELOPER_GUIDE.md",
    "docs/architecture/agent_supervisor/FOR_AGENTS.md",
    "docs/architecture/agent_supervisor/FOR_CONTRIBUTORS.md",
    "docs/architecture/agent_supervisor/PACKAGE_MAP.md",
    "docs/architecture/agent_supervisor/PROGRAMS.md",
    "docs/architecture/decisions/README.md",
    # Reference feature surface revalidated with the docs refresh
    "docs/features/hf-model-server/README.md",
    "docs/NESTED_PACKAGES.md",
)

FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
LINK_RE = re.compile(r"(?<!!)\[(?P<label>[^\]]*)\]\((?P<target>[^)]+)\)")
ABS_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
HEADING_RE = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*#*\s*$", re.MULTILINE)
HTML_ID_RE = re.compile(
    r'<(?:h[1-6]|a|span|div)[^>]*\bid=["\'](?P<id>[^"\']+)["\']',
    re.IGNORECASE,
)
MD_SUFFIXES = {".md", ".markdown", ".mdx"}


def strip_markup_noise(text: str) -> str:
    text = FENCE_RE.sub("", text)
    text = INLINE_CODE_RE.sub("", text)
    return text


def github_slug(title: str) -> str:
    """Approximate GitHub / CommonMark heading anchors."""
    value = unicodedata.normalize("NFKD", title)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value


def collect_markdown_anchors(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return set()
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for match in HEADING_RE.finditer(text):
        title = match.group("title").strip()
        explicit = re.search(r"\{#([A-Za-z0-9._:-]+)\}\s*$", title)
        if explicit:
            slug = explicit.group(1)
        else:
            slug = github_slug(title)
        if not slug:
            continue
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    for match in HTML_ID_RE.finditer(text):
        anchors.add(match.group("id"))
    return anchors


def split_target(raw: str) -> tuple[str, str | None]:
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    if " " in target:
        first, rest = target.split(None, 1)
        if rest[:1] in {'"', "'"}:
            target = first
    if "#" in target:
        path_part, frag = target.split("#", 1)
        return path_part, frag
    return target, None


def is_external(path_part: str, fragment: str | None) -> bool:
    probe = path_part if path_part else f"#{fragment or ''}"
    if not probe or probe == "#":
        return True
    if path_part.startswith("//"):
        return True
    if path_part and ABS_SCHEME_RE.match(path_part):
        return True
    return False


def resolve_link(source: Path, path_part: str) -> Path | None:
    if path_part == "":
        return source
    if path_part.startswith("/"):
        return None
    candidate = (source.parent / path_part).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return candidate


def rel_display(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def check_file(path: Path, *, check_anchors: bool) -> list[str]:
    errors: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{rel_display(path)}: cannot read file: {exc}"]

    text = strip_markup_noise(raw)
    for match in LINK_RE.finditer(text):
        target_raw = match.group("target").strip()
        path_part, fragment = split_target(target_raw)
        if is_external(path_part, fragment):
            continue

        if path_part == "" and fragment is not None:
            if check_anchors:
                anchors = collect_markdown_anchors(path)
                if fragment and fragment not in anchors:
                    errors.append(
                        f"{rel_display(path)}: missing anchor #{fragment} (self)"
                    )
            continue

        resolved = resolve_link(path, path_part)
        if resolved is None:
            errors.append(
                f"{rel_display(path)}: link escapes repository or is "
                f"site-absolute: {target_raw!r}"
            )
            continue
        if not resolved.exists():
            errors.append(
                f"{rel_display(path)}: missing target {target_raw!r} "
                f"-> {rel_display(resolved)}"
            )
            continue
        if (
            fragment
            and check_anchors
            and resolved.is_file()
            and resolved.suffix.lower() in MD_SUFFIXES
        ):
            anchors = collect_markdown_anchors(resolved)
            if fragment not in anchors:
                errors.append(
                    f"{rel_display(path)}: missing anchor #{fragment} in "
                    f"{rel_display(resolved)}"
                )
    return errors


def expand_allowlist(entries: tuple[str, ...]) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    missing: list[str] = []
    for entry in entries:
        path = ROOT / entry
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.md") if p.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            missing.append(entry)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-anchors",
        action="store_true",
        help="Only verify path existence; skip heading fragment checks",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print allowlisted files and exit 0",
    )
    args = parser.parse_args(argv)

    files, missing = expand_allowlist(ALLOWLIST)
    if missing:
        print("Allowlist paths missing from tree:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 1

    if args.list:
        for path in files:
            print(rel_display(path))
        return 0

    errors: list[str] = []
    for path in files:
        errors.extend(check_file(path, check_anchors=not args.no_anchors))

    if errors:
        print(
            f"Broken relative links in allowlisted docs ({len(errors)}):",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        return 1

    mode = "anchors checked" if not args.no_anchors else "paths only"
    print(
        f"OK: {len(files)} allowlisted documentation files have resolvable "
        f"relative links ({mode})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(2)
