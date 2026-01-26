#!/usr/bin/env python3
"""Bulk asyncio -> anyio migration helper.

Goal
----
Do mechanical, low-risk conversions in bulk (sleep/run/locks/events/to_thread)
while leaving higher-level concurrency constructs in place with TODO markers.

Safety
------
- Default is DRY RUN (no file writes)
- Optional unified diffs via --show-diff
- Optional backups via --backup (stores copies under .migrate_backups/)

Examples
--------
Dry run on core package:
  .venv/bin/python tools/bulk_asyncio_to_anyio.py ipfs_accelerate_py

Apply changes with backups:
  .venv/bin/python tools/bulk_asyncio_to_anyio.py ipfs_accelerate_py --apply --backup

Apply repo-wide but skip tests:
  .venv/bin/python tools/bulk_asyncio_to_anyio.py . --apply --backup --exclude 'test/**'

Notes
-----
This tool intentionally does NOT try to auto-convert:
- asyncio.create_task / gather -> anyio task groups
- asyncio.Queue -> anyio memory object streams
- event loop management (get_event_loop/new_event_loop/set_event_loop)
Those are left as-is with inline TODO markers so the code stays valid.
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXCLUDES = [
    ".git/**",
    ".venv/**",
    "**/__pycache__/**",
    "**/node_modules/**",
    "build/**",
    "dist/**",
]


TODO_TAG = "TODO(anyio-migrate)"
UNSAFE_ASYNCIO_CALLS = [
    "create_task",
    "gather",
    "wait_for",
    "Queue",
    "get_event_loop",
    "new_event_loop",
    "set_event_loop",
]


@dataclass
class FileResult:
    path: Path
    changed: bool
    reason: str
    warnings: list[str]
    diff: str | None = None


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return str(path)


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def _ensure_anyio_import(text: str) -> str:
    if re.search(r"^\s*import\s+anyio\b", text, re.MULTILINE):
        return text
    if re.search(r"^\s*from\s+anyio\b", text, re.MULTILINE):
        return text

    lines = text.splitlines(True)

    insert_at = 0
    # Skip shebang
    if lines and lines[0].startswith("#!"):
        insert_at = 1

    # Skip encoding cookie
    if insert_at < len(lines) and re.match(r"^#.*coding[:=]", lines[insert_at]):
        insert_at += 1

    # Skip module docstring if present
    if insert_at < len(lines) and re.match(r"^\s*(\"\"\"|''')", lines[insert_at]):
        quote = "\"\"\"" if "\"\"\"" in lines[insert_at] else "'''"
        insert_at += 1
        while insert_at < len(lines) and quote not in lines[insert_at]:
            insert_at += 1
        if insert_at < len(lines):
            insert_at += 1

    # Insert after leading blank/comment lines
    while insert_at < len(lines) and (lines[insert_at].strip() == "" or lines[insert_at].lstrip().startswith("#")):
        insert_at += 1

    lines.insert(insert_at, "import anyio\n")
    return "".join(lines)


def _ensure_inspect_import(text: str) -> str:
    if re.search(r"^\s*import\s+inspect\b", text, re.MULTILINE):
        return text

    lines = text.splitlines(True)

    # Insert after the last import line in the initial import block.
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
        elif insert_at > 0 and line.strip() and not line.lstrip().startswith("#"):
            break

    lines.insert(insert_at, "import inspect\n")
    return "".join(lines)


def migrate_text(text: str, *, mark_unsafe: bool) -> tuple[str, dict[str, int], list[str]]:
    """Return (new_text, counters, warnings).

    By default this only applies *mechanical* conversions that are very likely
    to be correct without understanding control flow.

    If mark_unsafe=True, it will optionally annotate some unsafe asyncio usage
    sites by appending an end-of-line TODO marker when it can do so safely.
    """

    counters: dict[str, int] = {
        "anyio_added": 0,
        "sleep": 0,
        "run": 0,
        "event": 0,
        "lock": 0,
        "to_thread": 0,
        "iscoro": 0,
        "unsafe_found": 0,
        "unsafe_marked": 0,
    }
    warnings: list[str] = []

    original = text

    # Mechanical replacements that keep code valid.
    text, n = re.subn(r"\basyncio\.sleep\(", "anyio.sleep(", text)
    counters["sleep"] += n

    text, n = re.subn(r"\basyncio\.run\(", "anyio.run(", text)
    counters["run"] += n

    text, n = re.subn(r"\basyncio\.Event\(\)", "anyio.Event()", text)
    counters["event"] += n

    text, n = re.subn(r"\basyncio\.Lock\(\)", "anyio.Lock()", text)
    counters["lock"] += n

    text, n = re.subn(r"\basyncio\.to_thread\(", "anyio.to_thread.run_sync(", text)
    counters["to_thread"] += n

    # iscoroutinefunction -> inspect.iscoroutinefunction
    text, n = re.subn(r"\basyncio\.iscoroutinefunction\(", "inspect.iscoroutinefunction(", text)
    counters["iscoro"] += n

    # Detect higher-level constructs that need manual conversion.
    unsafe_hits: dict[str, int] = {}
    for call in UNSAFE_ASYNCIO_CALLS:
        n_call = len(re.findall(rf"\basyncio\.{re.escape(call)}\b", text))
        if n_call:
            unsafe_hits[call] = n_call
            counters["unsafe_found"] += n_call

    if unsafe_hits:
        warnings.append(
            "Unsafe asyncio constructs present (manual anyio conversion needed): "
            + ", ".join(f"{k}={v}" for k, v in sorted(unsafe_hits.items()))
        )

    if mark_unsafe and unsafe_hits:
        # Conservative marking: only append TODO to lines that:
        # - contain an unsafe asyncio call
        # - do not already contain TODO_TAG
        # - do not already contain a comment (#), to avoid mangling existing comments
        # This keeps code valid and avoids the whitespace/syntax issues from modifying call sites.
        lines = text.splitlines(True)
        out: list[str] = []
        for line in lines:
            if TODO_TAG in line or "#" in line:
                out.append(line)
                continue
            if any(f"asyncio.{call}" in line for call in unsafe_hits.keys()):
                stripped_nl = "\n" if line.endswith("\n") else ""
                base = line[:-1] if stripped_nl else line
                out.append(f"{base}  # {TODO_TAG}: convert to anyio pattern{stripped_nl}")
                counters["unsafe_marked"] += 1
            else:
                out.append(line)
        text = "".join(out)

    # If we introduced anyio usage, ensure import anyio exists.
    anyio_used = any(
        counters[k] > 0 for k in ["sleep", "run", "event", "lock", "to_thread"]
    )

    if anyio_used:
        before = text
        text = _ensure_anyio_import(text)
        if text != before:
            counters["anyio_added"] += 1

    if counters["iscoro"] > 0:
        text = _ensure_inspect_import(text)

    # If we replaced import asyncio -> import anyio, we might break remaining asyncio uses.
    # We do NOT touch imports automatically; we only add anyio.

    if text != original:
        # Quick sanity warnings.
        if "# TODO: Replace with" in text:
            warnings.append("Found legacy TODO markers from older migration scripts")

    return text, counters, warnings


def migrate_file(
    path: Path,
    root: Path,
    *,
    apply: bool,
    backup: bool,
    backup_root: Path | None,
    show_diff: bool,
    mark_unsafe: bool,
) -> FileResult:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return FileResult(path=path, changed=False, reason=f"read-error: {e}", warnings=[str(e)])

    if "asyncio" not in raw:
        return FileResult(path=path, changed=False, reason="no-asyncio", warnings=[])

    new_text, counters, warnings = migrate_text(raw, mark_unsafe=mark_unsafe)

    if new_text == raw:
        return FileResult(path=path, changed=False, reason="no-change", warnings=warnings)

    diff_text = None
    if show_diff:
        diff_text = "".join(
            difflib.unified_diff(
                raw.splitlines(True),
                new_text.splitlines(True),
                fromfile=f"a/{_rel(root, path)}",
                tofile=f"b/{_rel(root, path)}",
            )
        )

    if apply:
        if backup:
            if backup_root is None:
                backup_root = root / ".migrate_backups"
            dst = backup_root / _rel(root, path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(raw, encoding="utf-8")

        path.write_text(new_text, encoding="utf-8")

    return FileResult(
        path=path,
        changed=True,
        reason=(
            f"updated (anyio_added={counters['anyio_added']}, sleep={counters['sleep']}, run={counters['run']}, "
            f"lock={counters['lock']}, event={counters['event']}, to_thread={counters['to_thread']}, "
            f"unsafe_found={counters['unsafe_found']}, unsafe_marked={counters['unsafe_marked']})"
        ),
        warnings=warnings,
        diff=diff_text,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Bulk asyncio->anyio mechanical migration")
    parser.add_argument("paths", nargs="+", help="File(s) or directory(ies) to process")
    parser.add_argument("--apply", action="store_true", help="Write changes to disk")
    parser.add_argument("--backup", action="store_true", help="Store backups under .migrate_backups/")
    parser.add_argument("--backup-root", default=None, help="Override backup root directory")
    parser.add_argument("--show-diff", action="store_true", help="Print unified diffs for changed files")
    parser.add_argument(
        "--mark-unsafe",
        action="store_true",
        help=f"Conservatively append end-of-line '{TODO_TAG}' markers where possible for unsafe asyncio constructs",
    )
    parser.add_argument("--exclude", action="append", default=[], help="Glob exclude pattern (repeatable)")

    args = parser.parse_args(argv)

    root = Path(os.getcwd()).resolve()
    excludes = DEFAULT_EXCLUDES + args.exclude

    backup_root = Path(args.backup_root).resolve() if args.backup_root else None

    changed: list[FileResult] = []
    processed = 0
    skipped = 0

    def iter_files(p: Path):
        if p.is_file() and p.suffix == ".py":
            yield p
            return
        if p.is_dir():
            yield from p.rglob("*.py")

    for raw_path in args.paths:
        p = Path(raw_path).resolve()
        if not p.exists():
            print(f"[skip] missing: {raw_path}", file=sys.stderr)
            continue

        for file_path in iter_files(p):
            rel = _rel(root, file_path)
            if _matches_any(rel, excludes):
                skipped += 1
                continue

            processed += 1
            res = migrate_file(
                file_path,
                root,
                apply=args.apply,
                backup=args.backup,
                backup_root=backup_root,
                show_diff=args.show_diff,
                mark_unsafe=args.mark_unsafe,
            )
            if res.changed:
                changed.append(res)
                print(f"[change] {rel}: {res.reason}")
                if res.diff:
                    print(res.diff)
            else:
                # Keep output quiet for no-change files.
                pass

    print("\nSummary")
    print(f"- processed: {processed}")
    print(f"- changed:   {len(changed)}")
    print(f"- skipped:   {skipped}")

    if not args.apply:
        print("- mode:      DRY RUN (use --apply to write)")
    else:
        print("- mode:      APPLY")
        if args.backup:
            print(f"- backups:   enabled ({backup_root or (root / '.migrate_backups')})")

    # Non-zero exit if we changed nothing? keep 0.
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
