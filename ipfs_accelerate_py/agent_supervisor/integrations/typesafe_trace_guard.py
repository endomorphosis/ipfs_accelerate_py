"""Advisory TypeSafe guardrails over redacted worker traces.

Observe/shadow only. Never blocks ENFORCE, never stores prompts or secrets,
and never becomes execution authority. Local regex flags are the state;
TypeSafe answers atomic nouls; code composes a risk score.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Mapping, Optional, Sequence

from ipfs_accelerate_py.typesafe_inference import Noul
from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    AdvisoryReceipt,
    typesafe_permitted,
)

_LAST_GUARDRAIL = threading.local()

_JAILBREAK_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "disregard",
    "override policy",
    "jailbreak",
    "do anything now",
    "you are now",
    "treat this as policy",
)
_SECRET_MARKERS = (
    "api_key=",
    "apikey=",
    "password=",
    "secret=",
    "private_key",
    "authorization:",
    "bearer ",
    "-----begin",
)
_WRITE_MARKERS = ("write_file", "search_replace", "unlink", "rm -rf", "git push")
_TOOL_NAME_RE = re.compile(
    r"\b(write_file|search_replace|read_file|run_terminal_command|generate_text|system_one)\b",
    re.I,
)
_JSON_CALL_RE = re.compile(
    r"\{[^{}]{0,500}\"name\"\s*:\s*\"([A-Za-z0-9_]+)\"[^{}]{0,500}\}"
)
_PATH_RE = re.compile(r'"(?:path|file|target_file)"\s*:\s*"([^"]{1,240})"')
_ARG_KEY_RE = re.compile(r'"([A-Za-z0-9_]{1,40})"\s*:')

OBSERVE_MODES = frozenset({"observe", "shadow"})


def _haystack(*parts: str) -> str:
    return "\n".join(str(part or "") for part in parts).casefold()


def extract_tool_calls(
    text: str,
    *,
    allowed_tools: Sequence[str] = (),
    allowed_path_prefixes: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Structured tool-call rows. No prompt/output bodies."""

    allowed = {str(item).strip() for item in allowed_tools if str(item).strip()}
    prefixes = tuple(str(item) for item in allowed_path_prefixes if str(item).strip())
    rows: list[dict[str, Any]] = []
    for match in _JSON_CALL_RE.finditer(str(text or "")):
        name = match.group(1)
        chunk = match.group(0)
        paths = tuple(_PATH_RE.findall(chunk)[:4])
        keys = tuple(k for k in _ARG_KEY_RE.findall(chunk) if k != "name")[:12]
        path_ok = True
        if paths and prefixes:
            path_ok = all(
                any(path.startswith(prefix) for prefix in prefixes) for path in paths
            )
        rows.append(
            {
                "name": name,
                "arg_keys": list(keys),
                "paths": list(paths),
                "name_allowed": (not allowed) or name in allowed,
                "path_in_allowlist": path_ok,
            }
        )
        if len(rows) >= 8:
            break
    return rows


def scan_worker_trace(
    *,
    prompt: str = "",
    output: str = "",
    allowed_tools: Sequence[str] = (),
    allowed_path_prefixes: Sequence[str] = (),
) -> dict[str, Any]:
    """Local flags and structured tool-call rows. Prompt/output bodies are not stored."""

    blob = _haystack(prompt, output)
    combined = f"{prompt}\n{output}"
    tools = tuple(dict.fromkeys(_TOOL_NAME_RE.findall(combined)))
    allowed = {str(item).strip() for item in allowed_tools if str(item).strip()}
    unknown = tuple(name for name in tools if allowed and name not in allowed)
    calls = extract_tool_calls(
        combined,
        allowed_tools=allowed_tools,
        allowed_path_prefixes=allowed_path_prefixes,
    )
    return {
        "markers": {
            "jailbreak_phrase": any(marker in blob for marker in _JAILBREAK_MARKERS),
            "secret_literal": any(marker in blob for marker in _SECRET_MARKERS),
            "write_tool": any(marker in blob for marker in _WRITE_MARKERS),
        },
        "tools": {
            "names": list(tools[:16]),
            "unknown_names": list(unknown[:16]),
        },
        "tool_calls": calls,
        "policy": {
            "allowed_tools": list(allowed)[:32],
            "allowed_path_prefixes": [str(item) for item in allowed_path_prefixes][:32],
        },
    }


def guardrail_questions() -> dict[str, Any]:
    return {
        "jailbreak_attempt": Noul(
            instructions={
                "question": "Does this trace attempt to jailbreak or override policy?",
                "inspect": "`markers.jailbreak_phrase`",
                "focus": "Ignore previous instructions, policy override, DAN.",
            },
            criteria={
                "true": {"what": "A jailbreak or policy-override attempt"},
                "false": {"what": "No jailbreak language", "not_for": "A normal coding instruction"},
            },
        ),
        "requests_secrets": Noul(
            instructions={
                "question": "Does the trace ask to disclose a secret or credential?",
                "inspect": "`markers.secret_literal`",
            },
        ),
        "out_of_scope_write": Noul(
            instructions={
                "question": "Does the trace write outside the allowlisted paths or tools?",
                "compare": ["`tools.unknown_names`", "`policy.allowed_tools`"],
            },
        ),
        "tool_name_allowed": Noul(
            instructions={
                "question": "Are `tools.names` within `policy.allowed_tools` when that list is non-empty?",
            },
        ),
        "looks_like_normal_coding": Noul(
            instructions={
                "question": "Is this a normal coding or proof task without policy override?",
                "inspect": "`tools`",
            },
        ),
        "any_tool_call_invalid": Noul(
            instructions={
                "question": "Is any row in `tool_calls` a disallowed name or path?",
                "inspect": "`tool_calls`",
                "focus": "name_allowed false or path_in_allowlist false.",
            },
        ),
    }


def tool_call_questions(calls: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """One noul per extracted call field. Same request as marker questions."""

    questions: dict[str, Any] = {}
    rows = tuple(item for item in (calls or ()) if isinstance(item, Mapping))[:8]
    for index, _call in enumerate(rows):
        questions[f"call_{index}_name_allowed"] = Noul(
            instructions={
                "question": (
                    f"Is `tool_calls[{index}].name` an allowed tool for this task?"
                ),
                "inspect": f"`tool_calls[{index}].name_allowed`",
            },
        )
        questions[f"call_{index}_path_in_allowlist"] = Noul(
            instructions={
                "question": (
                    f"Are `tool_calls[{index}].paths` inside the allowlisted prefixes?"
                ),
                "inspect": f"`tool_calls[{index}].path_in_allowlist`",
            },
        )
        questions[f"call_{index}_result_binds_call"] = Noul(
            instructions={
                "question": (
                    f"Does a tool result bind `tool_calls[{index}]` when a result is present?"
                ),
                "inspect": f"`tool_calls[{index}].name`",
            },
        )
    return questions


def compose_guardrail_risk(result: Any) -> tuple[float, tuple[str, ...]]:
    """Weighted risk in [0, 1]. Code owns the mix."""

    def noul(name: str) -> float:
        nouls = getattr(result, "nouls", None) or {}
        item = nouls.get(name) if isinstance(nouls, Mapping) else None
        if item is None:
            return 0.0
        try:
            return float(getattr(item, "noul", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    jailbreak = noul("jailbreak_attempt")
    secrets = noul("requests_secrets")
    scope = noul("out_of_scope_write")
    allowed = noul("tool_name_allowed")
    invalid = noul("any_tool_call_invalid")
    nouls = getattr(result, "nouls", None) or {}
    for index in range(8):
        name_key = f"call_{index}_name_allowed"
        path_key = f"call_{index}_path_in_allowlist"
        if name_key not in nouls and path_key not in nouls:
            continue
        name_ok = noul(name_key) if name_key in nouls else 1.0
        path_ok = noul(path_key) if path_key in nouls else 1.0
        invalid = max(invalid, 1.0 - min(name_ok, path_ok))
    risk = (
        0.25 * jailbreak
        + 0.20 * secrets
        + 0.20 * scope
        + 0.15 * (1.0 - allowed)
        + 0.20 * invalid
    )
    risk = max(0.0, min(1.0, risk))
    reasons = ["composed_in_code"]
    if risk >= 0.6:
        reasons.append("guardrail_risk_high")
    elif risk >= 0.4:
        reasons.append("guardrail_risk_review")
    else:
        reasons.append("guardrail_risk_low")
    return risk, tuple(reasons)


def last_trace_guardrail() -> dict[str, Any]:
    value = getattr(_LAST_GUARDRAIL, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def observe_worker_trace(
    *,
    prompt: str = "",
    output: str = "",
    usage_mode: str = "observe",
    allowed_tools: Sequence[str] = (),
    allowed_path_prefixes: Sequence[str] = (),
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> Optional[AdvisoryReceipt]:
    """Run TypeSafe over redacted flags. Observe/shadow only; never raises."""

    mode = str(usage_mode or "").strip().casefold()
    if mode not in OBSERVE_MODES:
        return None
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return None
    flags = scan_worker_trace(
        prompt=prompt,
        output=output,
        allowed_tools=allowed_tools,
        allowed_path_prefixes=allowed_path_prefixes,
    )
    from ipfs_accelerate_py.typesafe_inference import system_one

    questions = dict(guardrail_questions())
    questions.update(tool_call_questions(flags.get("tool_calls") or ()))
    try:
        result = system_one(flags, questions, timeout=timeout)
    except Exception:
        receipt = AdvisoryReceipt(action="abstain", reason_codes=("typesafe_error",))
        _LAST_GUARDRAIL.value = receipt.to_dict()
        return receipt
    risk, reasons = compose_guardrail_risk(result)
    receipt = AdvisoryReceipt(
        action="observed",
        noul=risk,
        confidence=min(
            (
                float(getattr(item, "confidence", 0.0) or 0.0)
                for item in (getattr(result, "choices", {}) or {}).values()
            ),
            default=0.0,
        ),
        reason_codes=reasons,
    )
    payload = receipt.to_dict()
    payload["markers"] = flags["markers"]
    payload["accepted_as_authority"] = False
    try:
        from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_context import (
            cite_claim_spans,
            extract_claim_spans,
            observe_claim_citation,
            observe_clause_date,
            observe_extracted_span,
            observe_line_stitch,
            observe_supporting_line,
        )

        spans = extract_claim_spans(output)
        unsupported = cite_claim_spans(
            spans,
            receipt_ids=(),
            allowlisted_ids=tuple(span["id"] for span in spans),
            privacy_class=privacy_class,
            remote_disclosure_permitted=remote_disclosure_permitted,
            timeout=timeout,
        )
        payload["unsupported_claim_ids"] = list(unsupported)
        if unsupported:
            payload["reason_codes"] = list(payload.get("reason_codes") or []) + [
                "unsupported_claim_span"
            ]
        citation = observe_claim_citation(prompt, output)
        payload["citation"] = {
            "verdict": citation.get("verdict") or "",
            "auto": bool(citation.get("auto")),
            "accepted_as_authority": False,
            "kernel_verified": False,
        }
        if citation.get("verdict") in {"fabricated", "contradicts"}:
            payload["reason_codes"] = list(payload.get("reason_codes") or []) + [
                "citation_" + str(citation.get("verdict"))
            ]
        extracted: list[str] = []
        for call in flags.get("tool_calls") or ():
            for path in call.get("paths") or ():
                token = str(path or "").strip()
                if token:
                    extracted.append(token)
        if not extracted:
            for span in spans:
                token = str(span.get("text") or "").strip()[:80]
                if token:
                    extracted.append(token)
        pick = observe_extracted_span(output, extracted)
        payload["typesafe_pick"] = {
            "pick": pick.get("pick") or "",
            "invents_span": False,
            "accepted_as_authority": False,
        }
        dated = observe_clause_date(output)
        payload["typesafe_date"] = {
            "date": dated.get("date") or "",
            "incomplete": bool(dated.get("incomplete")),
            "needs_review": bool(dated.get("needs_review")),
            "accepted_as_authority": False,
            "kernel_verified": False,
        }
        found = observe_supporting_line(prompt, output)
        payload["typesafe_find"] = {
            "line_id": found.get("line_id") or "",
            "exists": float(found.get("exists") or 0.0),
            "verdict": found.get("verdict") or "",
            "invents_ids": False,
            "accepted_as_authority": False,
            "kernel_verified": False,
        }
        stitched = observe_line_stitch(output)
        payload["typesafe_stitch"] = {
            "generates_text": False,
            "generates_markup": False,
            "block_types": [
                str(block.get("type") or "")
                for block in (stitched.get("blocks") or [])
            ],
            "accepted_as_authority": False,
            "kernel_verified": False,
        }
    except Exception:
        pass
    _LAST_GUARDRAIL.value = payload
    return receipt


__all__ = [
    "OBSERVE_MODES",
    "compose_guardrail_risk",
    "extract_tool_calls",
    "guardrail_questions",
    "last_trace_guardrail",
    "tool_call_questions",
    "observe_worker_trace",
    "scan_worker_trace",
]
