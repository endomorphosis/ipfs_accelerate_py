"""Make generated Markdown context references portable before task dispatch."""
from pathlib import Path


def localize_context(repo: Path) -> int:
    """Replace references to this seed checkout with workspace-relative paths.

    Run before the generated board is committed or registered with the daemon.
    No task IDs, outputs, acceptance rules, or external paths are changed.
    """
    count = 0
    prefix = str(repo.resolve()) + "/"
    files = [repo / "tasks.todo.md", repo / "objectives.md"]
    files += list((repo / ".runtime").rglob("*.md"))
    for path in files:
        if not path.is_file():
            continue
        before = path.read_text()
        after = before.replace(prefix, "")
        if after != before:
            path.write_text(after)
            count += 1
    return count
