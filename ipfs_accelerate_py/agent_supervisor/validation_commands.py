"""Validation command helpers for supervisor-managed todo boards."""

from __future__ import annotations


def split_validation_commands(value: str) -> list[str]:
    """Split semicolon-separated shell commands without splitting quoted code."""

    text = str(value or "")
    commands: list[str] = []
    current: list[str] = []
    in_single_quote = False
    in_double_quote = False
    escaped = False

    def flush() -> None:
        command = "".join(current).strip()
        if command:
            commands.append(command)
        current.clear()

    for char in text:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\" and not in_single_quote:
            current.append(char)
            escaped = True
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current.append(char)
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            current.append(char)
            continue
        if char == ";" and not in_single_quote and not in_double_quote:
            flush()
            continue
        current.append(char)

    flush()
    return commands
