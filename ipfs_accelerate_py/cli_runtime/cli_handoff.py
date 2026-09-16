"""Bounded, redacted context packets for migrating work between CLI tools."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Optional

MAX_HANDOFF_CHARS = 24000
FULL_HANDOFF_CHARS = 80000
COMPACT_HANDOFF_CHARS = 8000
_MAX_TURNS = 40
_FULL_MAX_TURNS = 120
_COMPACT_RECENT_TURNS = 8
_FILE_TOKEN = re.compile(
    r"(?:(?<![A-Za-z0-9._-])(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]{1,8}|[\w.-]+\.(?:py|ts|tsx|js|jsx|go|rs|md|toml|json|yml|yaml|txt|sh|lean))\b"
)

HISTORY_NONE = "none"
HISTORY_FULL = "full"
HISTORY_COMPACT = "compact"
HISTORY_MODES = frozenset({HISTORY_NONE, HISTORY_FULL, HISTORY_COMPACT})

_SECRET_ASSIGN = re.compile(
    r"(?i)([A-Za-z0-9_]*(?:api[_-]?key|token|secret|password)|authorization|bearer)\s*[:=]\s*\S+"
)
_KEYISH = re.compile(
    r"\b(?:sk|xai|ghp|gho|github_pat|AIza)[-_][A-Za-z0-9_\-]{8,}\b"
)


def redact_cli_context(text: str) -> str:
    """Strip credential-shaped substrings from a handoff blob."""
    raw = str(text or "")
    raw = _SECRET_ASSIGN.sub(r"\1=[redacted]", raw)
    raw = _KEYISH.sub("[redacted]", raw)
    return raw


def _join_turns(turns: Sequence[tuple[str, str]]) -> str:
    lines = []
    for role, text in turns:
        role_s = str(role or "message").strip() or "message"
        body = str(text or "").strip()
        if body:
            lines.append(f"{role_s}: {body}")
    return "\n".join(lines)


def bound_cli_context(text: str, *, max_chars: int = MAX_HANDOFF_CHARS) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_chars:
        return raw
    return raw[-max_chars:]


def apply_cli_handoff(
    prompt: str,
    handoff: str,
    *,
    from_provider: str = "",
) -> str:
    """Prefix a user prompt with migrated CLI context (once)."""
    body = str(prompt or "")
    context = str(handoff or "").strip()
    if not context:
        return body
    if "Prior work was on" in body and context[:80] in body:
        return body
    header = "Prior work was on another CLI tool."
    if from_provider:
        header = f"Prior work was on {from_provider}."
    return (
        f"{header} Continue from this context:\n\n"
        f"{context}\n\n---\n\n{body}"
    ).strip()


def _turns_from_mapping(payload: Mapping[str, Any]) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        for item in messages[-_FULL_MAX_TURNS:]:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role") or item.get("type") or "message").strip() or "message"
            text = item.get("text") or item.get("content") or item.get("message")
            if isinstance(text, list):
                parts = []
                for chunk in text:
                    if isinstance(chunk, Mapping):
                        chunk_text = chunk.get("text") or chunk.get("content")
                        if isinstance(chunk_text, str) and chunk_text.strip():
                            parts.append(chunk_text.strip())
                    elif isinstance(chunk, str) and chunk.strip():
                        parts.append(chunk.strip())
                text = "\n".join(parts)
            if isinstance(text, str) and text.strip():
                turns.append((role, text.strip()))
        return turns
    return turns


def _turns_from_muse_jsonl(path: Path) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return turns
    for line in lines:
        if not line.strip().startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        ptype = str(event.get("payload_type") or event.get("type") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        if "input.user" in ptype:
            text = payload.get("prompt") or payload.get("text")
            if isinstance(text, str) and text.strip():
                turns.append(("user", text.strip()))
        elif ptype in {"run.output.delta", "run.terminal.completed"} or "assistant" in ptype:
            text = payload.get("text")
            if isinstance(text, str) and text.strip() and not text.strip().startswith("echo:"):
                if turns and turns[-1][0] == "assistant":
                    turns[-1] = ("assistant", text.strip())
                else:
                    turns.append(("assistant", text.strip()))
        elif "tool" in ptype.lower() or "side_effect" in ptype.lower():
            name = str(payload.get("name") or payload.get("tool") or "tool")
            turns.append(("tool", name[:200]))
    return turns[-_FULL_MAX_TURNS:]


def extract_muse_session_context(
    native_session_id: str,
    *,
    muse_bin: str = "",
    timeout_seconds: float = 30.0,
    home: Optional[Path] = None,
) -> str:
    """Export a Muse session via `muse export --redacted`, else read local JSONL."""
    sid = str(native_session_id or "").strip()
    if not sid:
        return ""
    binary = muse_bin or shutil.which("muse") or str(Path.home() / ".local" / "bin" / "muse")
    looks_like_id = len(sid) >= 32 and all(ch.isalnum() or ch == "-" for ch in sid)
    if looks_like_id and binary and Path(binary).is_file():
        out = tempfile.NamedTemporaryFile(
            suffix=".json", prefix="muse-handoff-", delete=False
        )
        out_path = out.name
        out.close()
        try:
            proc = subprocess.run(
                [binary, "export", "--session", sid, "--redacted", "--out", out_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if getattr(proc, "returncode", 1) in {0, None} and Path(out_path).is_file():
                try:
                    payload = json.loads(
                        Path(out_path).read_text(encoding="utf-8", errors="replace")
                    )
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, Mapping):
                    turns = _turns_from_mapping(payload)
                    if turns:
                        return _join_turns(turns)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass
    root = _home(home) / ".local" / "share" / "muse" / "sessions"
    if not root.is_dir():
        return ""
    try:
        matches = list(root.rglob(f"{sid}/session.jsonl"))
    except OSError:
        matches = []
    if not matches:
        return ""
    return _join_turns(_turns_from_muse_jsonl(matches[-1]))


def _home(home: Optional[Path] = None) -> Path:
    if home is not None:
        return Path(home)
    return Path.home()


def _provider_session_roots(provider: str, *, home: Optional[Path] = None) -> list[Path]:
    root = _home(home)
    key = str(provider or "").strip().lower().replace("-", "_")
    mapping = {
        "muse_code": [root / ".local" / "share" / "muse" / "sessions"],
        "muse": [root / ".local" / "share" / "muse" / "sessions"],
        "claude_code": [root / ".claude" / "projects"],
        "codex_cli": [root / ".codex" / "sessions", root / ".codex"],
        "goose_cli": [
            root / ".local" / "share" / "goose" / "sessions",
            root / ".config" / "goose" / "sessions",
            root / ".goose" / "sessions",
        ],
        "grok_cli": [root / ".grok" / "sessions", root / ".grok"],
        "copilot_cli": [
            root / ".copilot" / "session-state",
            root / ".copilot",
            root / ".github-copilot",
        ],
        "gemini_cli": [root / ".gemini" / "tmp", root / ".gemini"],
        "mistral_vibe": [
            root / ".vibe" / "sessions",
            root / ".mistral-vibe",
            root / ".config" / "mistral-vibe",
        ],
    }
    return [path for path in mapping.get(key, [])]


def _find_session_files(roots: Sequence[Path], session_id: str) -> list[Path]:
    sid = str(session_id or "").strip()
    if not sid:
        return []
    found: list[Path] = []
    seen: set[str] = set()
    suffixes = {".jsonl", ".json", ".md"}
    for root in roots:
        try:
            if not root.is_dir():
                continue
        except OSError:
            continue
        candidates = [
            root / f"{sid}.jsonl",
            root / f"{sid}.json",
            root / sid / "session.jsonl",
            root / sid / "rollout.jsonl",
            root / sid / "transcript.jsonl",
        ]
        try:
            for path in root.rglob(f"*{sid}*"):
                if len(candidates) > 80:
                    break
                try:
                    if path.is_file() and path.suffix.lower() in suffixes:
                        candidates.append(path)
                    elif path.is_dir():
                        for name in ("session.jsonl", "rollout.jsonl", "transcript.jsonl"):
                            inner = path / name
                            if inner.is_file():
                                candidates.append(inner)
                except OSError:
                    continue
        except OSError:
            pass
        for path in candidates:
            try:
                resolved = str(path.resolve())
            except OSError:
                continue
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            found.append(path)
    return found


def _text_from_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("text", "content", "prompt", "message", "body"):
            inner = value.get(key)
            text = _text_from_content(inner)
            if text:
                return text
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [_text_from_content(item) for item in value]
        return "\n".join(part for part in parts if part)
    return ""


def _turns_from_jsonl_generic(path: Path) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return turns
    stripped = raw.strip()
    if stripped.startswith("{") and "\n" not in stripped[:200] and path.suffix == ".json":
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping):
            mapped = _turns_from_mapping(payload)
            if mapped:
                return mapped
    for line in raw.splitlines():
        if not line.strip().startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        etype = str(event.get("type") or event.get("payload_type") or event.get("kind") or "")
        role = str(event.get("role") or "").lower()
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        message = event.get("message") if isinstance(event.get("message"), Mapping) else {}
        nested_role = str(message.get("role") or payload.get("role") or role).lower()
        if any(token in etype.lower() for token in ("input.user",)) or nested_role in {
            "user",
            "human",
        } or etype.lower() in {"user", "human"}:
            text = (
                _text_from_content(payload.get("prompt"))
                or _text_from_content(payload.get("text"))
                or _text_from_content(message.get("content"))
                or _text_from_content(event.get("content"))
                or _text_from_content(event.get("text"))
            )
            if text:
                turns.append(("user", text))
            continue
        if "tool" in etype.lower() or nested_role == "tool":
            name = str(
                payload.get("name")
                or message.get("name")
                or event.get("name")
                or etype
            )[:200]
            if name:
                turns.append(("tool", name))
            continue
        if nested_role in {"assistant", "agent", "model"} or any(
            token in etype.lower()
            for token in (
                "assistant",
                "agent_message",
                "run.output",
                "run.terminal.completed",
                "complete",
            )
        ):
            text = (
                _text_from_content(payload.get("text"))
                or _text_from_content(message.get("content"))
                or _text_from_content(event.get("content"))
                or _text_from_content(event.get("text"))
            )
            if text and not text.startswith("echo:"):
                if "delta" in etype.lower() and turns and turns[-1][0] == "assistant":
                    turns[-1] = ("assistant", text)
                else:
                    turns.append(("assistant", text))
    return turns[-_FULL_MAX_TURNS:]


def extract_native_cli_turns(
    provider: str,
    native_session_id: str,
    *,
    home: Optional[Path] = None,
) -> list[tuple[str, str]]:
    """Structured user/assistant/tool turns from a native CLI session."""
    key = str(provider or "").strip().lower().replace("-", "_")
    sid = str(native_session_id or "").strip()
    if not sid:
        return []
    if key in {"muse_code", "muse"}:
        exported = extract_muse_session_context(sid, home=home)
        if exported:
            parsed: list[tuple[str, str]] = []
            for line in exported.splitlines():
                if ": " in line[:24]:
                    role, text = line.split(": ", 1)
                    parsed.append((role.strip() or "message", text))
            if parsed:
                return parsed
    files = _find_session_files(_provider_session_roots(key, home=home), sid)
    for path in reversed(files):
        turns = _turns_from_jsonl_generic(path)
        if not turns:
            turns = _turns_from_muse_jsonl(path)
        if turns:
            return list(turns)
    return []


def extract_native_cli_context(
    provider: str,
    native_session_id: str,
    *,
    home: Optional[Path] = None,
) -> str:
    """Best-effort local transcript for one native CLI session id.

    Every catalogued CLI provider is searched. Native ids are not portable,
    but the extracted turns can be handed to any other CLI as a prompt prefix.
    """
    return _join_turns(extract_native_cli_turns(provider, native_session_id, home=home))


def format_full_history(
    turns: Sequence[tuple[str, str]],
    *,
    from_provider: str = "",
) -> str:
    """Port native history as a near-complete turn log."""
    clipped = list(turns)[-_FULL_MAX_TURNS:]
    header = "Native history"
    if from_provider:
        header = f"Native history from {from_provider}"
    body = _join_turns(clipped)
    return bound_cli_context(f"{header} ({len(clipped)} turns):\n\n{body}", max_chars=FULL_HANDOFF_CHARS)


def format_compact_history(
    turns: Sequence[tuple[str, str]],
    *,
    from_provider: str = "",
) -> str:
    """Compact native history: goal, files, last few turns."""
    items = [(str(role), str(text).strip()) for role, text in turns if str(text).strip()]
    if not items:
        return ""
    first_user = next((text for role, text in items if role in {"user", "human"}), "")
    files: list[str] = []
    seen_files: set[str] = set()
    for _role, text in items:
        for match in _FILE_TOKEN.findall(text):
            token = match.strip()
            if token and token not in seen_files:
                seen_files.add(token)
                files.append(token)
            if len(files) >= 12:
                break
        if len(files) >= 12:
            break
    recent = items[-_COMPACT_RECENT_TURNS:]
    dropped = max(0, len(items) - len(recent))
    lines = [
        f"Compacted native history from {from_provider or 'cli'} ({len(items)} turns, dropped {dropped}):",
        "",
    ]
    if first_user:
        lines.append(f"Goal: {first_user[:500]}")
        lines.append("")
    if files:
        lines.append("Files: " + ", ".join(files))
        lines.append("")
    lines.append("Recent:")
    lines.append(_join_turns(recent))
    return bound_cli_context("\n".join(lines), max_chars=COMPACT_HANDOFF_CHARS)


def collect_cli_handoff_context(
    *,
    from_provider: str,
    native_session_id: str = "",
    handoff: str = "",
    home: Optional[Path] = None,
    history: str = HISTORY_FULL,
) -> str:
    """Merge caller notes with native history in ``full``, ``compact``, or ``none`` mode."""
    mode = str(history or HISTORY_FULL).strip().lower()
    if mode not in HISTORY_MODES:
        raise ValueError(f"history must be one of {sorted(HISTORY_MODES)}")
    parts: list[str] = []
    explicit = str(handoff or "").strip()
    if explicit:
        parts.append(explicit)
    if mode != HISTORY_NONE:
        turns = extract_native_cli_turns(
            from_provider, native_session_id, home=home
        )
        if turns:
            if mode == HISTORY_COMPACT:
                formatted = format_compact_history(turns, from_provider=from_provider)
            else:
                formatted = format_full_history(turns, from_provider=from_provider)
            if formatted and formatted not in explicit:
                parts.append(formatted)
    blob = redact_cli_context("\n\n".join(parts))
    max_chars = COMPACT_HANDOFF_CHARS if mode == HISTORY_COMPACT else (
        FULL_HANDOFF_CHARS if mode == HISTORY_FULL else MAX_HANDOFF_CHARS
    )
    return bound_cli_context(blob, max_chars=max_chars)


__all__ = [
    "COMPACT_HANDOFF_CHARS",
    "FULL_HANDOFF_CHARS",
    "HISTORY_COMPACT",
    "HISTORY_FULL",
    "HISTORY_MODES",
    "HISTORY_NONE",
    "MAX_HANDOFF_CHARS",
    "apply_cli_handoff",
    "bound_cli_context",
    "collect_cli_handoff_context",
    "extract_muse_session_context",
    "extract_native_cli_context",
    "extract_native_cli_turns",
    "format_compact_history",
    "format_full_history",
    "redact_cli_context",
]
