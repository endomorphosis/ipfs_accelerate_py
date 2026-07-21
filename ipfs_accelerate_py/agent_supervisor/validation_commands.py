"""Validation command helpers for supervisor-managed todo boards."""

from __future__ import annotations

import re

INLINE_CODE_COMMAND_RE = re.compile(r"^`+(?P<command>[^`]+?)`+\s*\.?\s*$", re.DOTALL)


def normalize_validation_command_text(value: str) -> str:
    """Return a shell command with markdown-only inline-code wrapping removed."""

    command = str(value or "").strip()
    match = INLINE_CODE_COMMAND_RE.fullmatch(command)
    if match:
        return match.group("command").strip()
    return command


def split_validation_commands(value: str) -> list[str]:
    """Split semicolon-separated shell commands without splitting quoted code."""

    text = normalize_validation_command_text(value)
    commands: list[str] = []
    current: list[str] = []
    in_single_quote = False
    in_double_quote = False
    escaped = False

    def flush() -> None:
        command = normalize_validation_command_text("".join(current))
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
