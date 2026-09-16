"""Parse Leanstral chat output into TypeSafe System One solver inputs.

Leanstral is an untrusted proposal generator. This module:

* keeps only the first terminated assistant turn;
* never mines ``thought`` preambles for proof text;
* builds typed noul/choice/score questions for TypeSafe;
* treats TypeSafe as a structured solver over the parsed draft + goal.

TypeSafe answers are advisory. They are not kernel proof.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .typesafe_inference import (
    Choice,
    Noul,
    Score,
    SystemOneResult,
    typesafe_configured,
)

IM_END = "<|im_end|>"
IM_START = "<|im_start|>"
_THOUGHT_OPEN = re.compile(r"<\|im_start\|>thought>?", re.I)
_START_LINE = re.compile(r"^<\|im_start\|>([A-Za-z0-9_-]*)\s*\n?", re.I)
_FENCE = re.compile(r"^```(?:lean4?|text)?\s*\n(.*)\n```\s*$", re.S | re.I)
_FORBIDDEN = re.compile(
    r"\b(sorry|admit|sorryAx|axiom|constant|opaque|unsafe|import|open|"
    r"set_option|theorem|def|instance|macro|syntax|elab|run_tac|"
    r"native_decide|IO|System|Lean|eval|include|attribute)\b"
)

DEFAULT_LEANSTRAL_ENDPOINTS: tuple[str, ...] = (
    "http://127.0.0.1:8080/v1",
    "http://172.17.0.1:8080/v1",
)

PROOF_PROMPT = (
    "Return only a Lean 4 proof body starting with by for the declaration below. "
    "Do not change the declaration or add premises. Do not use imports, sorry, "
    "admit, axioms, unsafe features, native_decide, or executable metaprogramming. "
    "Use only elementary core Lean tactics. If the statement is not provable from "
    "its explicit parameters, return exactly ABSTAIN. Names such as add and zero "
    "denote arbitrary parameters, not arithmetic operations.\n\n{declaration} :="
)


@dataclass(frozen=True)
class ParsedLeanstralOutput:
    kind: str
    body: str = ""
    language: str = ""
    had_thought: bool = False
    first_turn: str = ""
    raw: str = ""

    @property
    def is_abstain(self) -> bool:
        return self.kind == "abstain"

    @property
    def is_proof_body(self) -> bool:
        return self.kind == "proof_body"


@dataclass(frozen=True)
class LeanstralGoal:
    goal_id: str
    declaration: str
    expected_provable: Optional[bool] = None
    expected_solver_status: str = ""


@dataclass
class SolverVerdict:
    disposition: str
    claim_status: str
    candidate_quality: float
    is_proof_body: float
    is_abstain: float
    uses_forbidden: float
    well_formed: float
    confidence: float
    parsed: ParsedLeanstralOutput
    typesafe_state: Mapping[str, Any] = field(default_factory=dict)
    typesafe_answers: Mapping[str, Any] = field(default_factory=dict)
    advisory_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "claim_status": self.claim_status,
            "candidate_quality": self.candidate_quality,
            "is_proof_body": self.is_proof_body,
            "is_abstain": self.is_abstain,
            "uses_forbidden": self.uses_forbidden,
            "well_formed": self.well_formed,
            "confidence": self.confidence,
            "kind": self.parsed.kind,
            "body": self.parsed.body,
            "advisory_only": True,
        }


def parse_leanstral_output(content: str) -> ParsedLeanstralOutput:
    """Extract a candidate from one Leanstral chat completion.

    Framing removal only. Does not repair proofs or read thought preambles.
    """

    raw = str(content or "")
    had_thought = bool(_THOUGHT_OPEN.search(raw))
    remainder = raw
    if remainder.lstrip().startswith(IM_END):
        remainder = remainder.lstrip()[len(IM_END) :].lstrip()
    if IM_END in remainder:
        first_turn = remainder.split(IM_END, 1)[0]
    else:
        first_turn = remainder
    first_turn = first_turn.strip()
    if _THOUGHT_OPEN.match(first_turn):
        return ParsedLeanstralOutput(
            kind="incomplete",
            had_thought=True,
            first_turn=first_turn,
            raw=raw,
        )
    language = ""
    start = _START_LINE.match(first_turn)
    if start:
        language = str(start.group(1) or "").strip()
        if language.casefold() in {"thought", "thought>"}:
            return ParsedLeanstralOutput(
                kind="incomplete",
                language=language,
                had_thought=True,
                first_turn=first_turn,
                raw=raw,
            )
        first_turn = first_turn[start.end() :].strip()
    fence = _FENCE.fullmatch(first_turn)
    if fence:
        first_turn = fence.group(1).strip()
        language = language or "lean4"
    if first_turn == "ABSTAIN":
        return ParsedLeanstralOutput(
            kind="abstain",
            language=language,
            had_thought=had_thought,
            first_turn=first_turn,
            raw=raw,
        )
    if first_turn.startswith("by"):
        return ParsedLeanstralOutput(
            kind="proof_body",
            body=first_turn,
            language=language or "lean4",
            had_thought=had_thought,
            first_turn=first_turn,
            raw=raw,
        )
    if not first_turn:
        return ParsedLeanstralOutput(
            kind="incomplete",
            had_thought=had_thought,
            first_turn=first_turn,
            raw=raw,
        )
    return ParsedLeanstralOutput(
        kind="malformed",
        body=first_turn,
        language=language,
        had_thought=had_thought,
        first_turn=first_turn,
        raw=raw,
    )


def draft_flags(parsed: ParsedLeanstralOutput) -> dict[str, bool]:
    body = parsed.body or parsed.first_turn
    return {
        "starts_with_by": body.lstrip().startswith("by"),
        "exact_abstain": parsed.kind == "abstain",
        "has_comments": ("--" in body) or ("/-" in body),
        "has_forbidden_identifier": bool(_FORBIDDEN.search(body)),
        "has_thought": parsed.had_thought,
        "empty": not str(body or "").strip() and parsed.kind != "abstain",
    }


def typesafe_state(goal: LeanstralGoal, parsed: ParsedLeanstralOutput) -> dict[str, Any]:
    """State for System One. Includes the parsed draft, never thought text."""

    return {
        "goal": {
            "id": goal.goal_id,
            "declaration": goal.declaration,
            "expected_provable": goal.expected_provable,
        },
        "draft": {
            "kind": parsed.kind,
            "body": parsed.body,
            "language": parsed.language,
            "flags": draft_flags(parsed),
        },
    }


def solver_questions() -> dict[str, Any]:
    """Atomic TypeSafe questions over the same draft. Combined in code."""

    return {
        "is_abstain": Noul(
            instructions={
                "question": "Did the model abstain instead of proposing a proof?",
                "inspect": "`draft.kind`",
                "focus": "Exact ABSTAIN, not a tactic script that mentions ABSTAIN.",
            },
            criteria={
                "true": {
                    "what": "`draft.kind` is abstain or `draft.body` is exactly ABSTAIN",
                    "not_for": "A by-proof that contains the identifier ABSTAIN",
                },
                "false": {"what": "The draft proposes tactics or other proof text"},
            },
        ),
        "is_proof_body": Noul(
            instructions={
                "question": "Is `draft.body` a Lean 4 tactic proof starting with by?",
                "inspect": "`draft.body`",
            },
        ),
        "well_formed": Noul(
            instructions={
                "question": "Is `draft.body` a tactic script only?",
                "inspect": "`draft.body`",
                "focus": "No extra prose, chat turns, comments, or thought text.",
            },
        ),
        "uses_forbidden": Noul(
            instructions={
                "question": "Does `draft.body` use a forbidden construct?",
                "inspect": "`draft.body`",
                "focus": "sorry, admit, axioms, imports, comments, or metaprogramming.",
            },
        ),
        "disposition": Choice(
            instructions={
                "question": "What should code do with this untrusted Leanstral draft?",
                "inspect": "`draft`",
            },
            criteria={
                "accept_candidate": {
                    "what": "Well-formed elementary proof body; send to the kernel",
                    "not_for": "Abstain, malformed, or forbidden constructs",
                },
                "abstain": {
                    "what": "The model declined; there is no candidate",
                    "not_for": "A by-proof, even a bad one",
                },
                "reject": {
                    "what": "Malformed, forbidden, incomplete, or not a proof",
                    "not_for": "A clean elementary tactic script",
                },
                "needs_kernel": {
                    "what": "Looks like a proof; only the kernel may accept it",
                    "not_for": "Exact ABSTAIN or empty thought-only output",
                },
            },
        ),
        "claim_status": Choice(
            instructions={
                "question": "SMT-LIB check-sat of the negation of `goal.declaration`?",
                "inspect": "`goal.declaration`",
                "focus": "Ignore whether `draft.body` would typecheck.",
            },
            criteria={
                "unsat": {
                    "what": "The claim holds from the stated parameters",
                    "not_for": "A countermodel exists",
                },
                "sat": {
                    "what": "A countermodel exists from the stated parameters",
                    "not_for": "The claim is valid",
                },
                "unknown": {
                    "what": "Not a first-order/decidable claim from these parameters",
                    "not_for": "A clear sat or unsat FOL/SMT claim",
                },
            },
        ),
        "candidate_quality": Score(
            instructions={
                "question": "How close is `draft.body` to a kernel-checkable elementary proof?",
                "inspect": "`draft.body`",
            },
            criteria=["unusable", "partial", "kernel-ready"],
        ),
    }


def compose_draft_quality(*, is_proof_body: float, well_formed: float, uses_forbidden: float) -> float:
    """Weighted draft quality in [0, 1]. Code owns the weights."""

    return max(
        0.0,
        min(
            1.0,
            0.4 * float(is_proof_body)
            + 0.4 * float(well_formed)
            + 0.2 * (1.0 - float(uses_forbidden)),
        ),
    )


def typesafe_request(goal: LeanstralGoal, parsed: ParsedLeanstralOutput) -> dict[str, Any]:
    from .typesafe_inference import serialize_questions

    return {
        "state": typesafe_state(goal, parsed),
        "questions": serialize_questions(solver_questions()),
    }


def _choice(result: SystemOneResult, name: str, default: str = "") -> str:
    answer = result.choices.get(name)
    if answer is None:
        return default
    return str(answer.choice or default)


def _noul(result: SystemOneResult, name: str) -> float:
    answer = result.nouls.get(name)
    if answer is None:
        return 0.0
    return float(answer.noul)


def _score(result: SystemOneResult, name: str) -> float:
    answer = result.scores.get(name)
    if answer is None:
        return 0.0
    return float(answer.score)


def _confidence(result: SystemOneResult) -> float:
    values: list[float] = []
    for answer in result.choices.values():
        values.append(float(answer.confidence))
    for answer in result.scores.values():
        values.append(float(answer.confidence))
    return min(values) if values else 0.0


def verdict_from_result(
    parsed: ParsedLeanstralOutput,
    result: SystemOneResult,
    *,
    state: Optional[Mapping[str, Any]] = None,
) -> SolverVerdict:
    return SolverVerdict(
        disposition=_choice(result, "disposition", "reject"),
        claim_status=_choice(result, "claim_status", "unknown"),
        candidate_quality=_score(result, "candidate_quality"),
        is_proof_body=_noul(result, "is_proof_body"),
        is_abstain=_noul(result, "is_abstain"),
        uses_forbidden=_noul(result, "uses_forbidden"),
        well_formed=_noul(result, "well_formed"),
        confidence=_confidence(result),
        parsed=parsed,
        typesafe_state=dict(state or {}),
        typesafe_answers=dict(result.answers),
    )


def solve_with_typesafe(
    goal: LeanstralGoal,
    content: str,
    *,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> SolverVerdict:
    """Parse Leanstral output and ask TypeSafe for a structured solver verdict."""

    from .typesafe_inference import system_one

    parsed = parse_leanstral_output(content)
    state = typesafe_state(goal, parsed)
    result = system_one(
        state,
        solver_questions(),
        model=model,
        timeout=timeout,
    )
    return verdict_from_result(parsed, result, state=state)


def discover_leanstral_base_url(
    *,
    environ: Optional[Mapping[str, str]] = None,
    timeout: float = 2.0,
) -> str:
    env = os.environ if environ is None else environ
    explicit = (
        str(env.get("LEANSTRAL_BASE_URL") or env.get("LEANSTRAL_ENDPOINT") or "").strip()
        or str(env.get("IPFS_ACCELERATE_LLAMA_CPP_BASE_URL") or "").strip()
    )
    candidates = []
    if explicit:
        candidates.append(explicit.rstrip("/"))
    candidates.extend(DEFAULT_LEANSTRAL_ENDPOINTS)
    seen: set[str] = set()
    for raw in candidates:
        base = raw.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        if base in seen:
            continue
        seen.add(base)
        try:
            req = urllib.request.Request(base + "/models", method="GET")
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            rows = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(rows, list) and any(
                "leanstral" in str((row or {}).get("id") or "").casefold() for row in rows
            ):
                return base
        except Exception:
            continue
    return ""


def leanstral_chat(
    prompt: str,
    *,
    base_url: str = "",
    model: str = "leanstral_local",
    max_tokens: int = 256,
    timeout: float = 90.0,
    temperature: float = 0.0,
    seed: int = 104729,
) -> dict[str, Any]:
    root = (base_url or discover_leanstral_base_url()).rstrip("/")
    if not root:
        raise RuntimeError("no Leanstral HTTP endpoint is reachable")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    req = urllib.request.Request(
        root + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"Leanstral HTTP {exc.code}: {detail or exc.reason}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Leanstral returned invalid JSON")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Leanstral response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise RuntimeError("Leanstral response missing string content")
    return {
        "content": content,
        "model": data.get("model"),
        "usage": data.get("usage"),
        "finish_reason": choices[0].get("finish_reason") if isinstance(choices[0], dict) else None,
        "base_url": root,
    }


def propose_and_solve(
    goal: LeanstralGoal,
    *,
    leanstral_base_url: str = "",
    max_tokens: int = 256,
    leanstral_timeout: float = 90.0,
    typesafe_timeout: Optional[float] = None,
    call_typesafe: Optional[bool] = None,
) -> dict[str, Any]:
    """Call Leanstral, parse the draft, optionally ask TypeSafe to solve."""

    prompt = PROOF_PROMPT.format(declaration=goal.declaration)
    chat = leanstral_chat(
        prompt,
        base_url=leanstral_base_url,
        max_tokens=max_tokens,
        timeout=leanstral_timeout,
    )
    parsed = parse_leanstral_output(str(chat["content"]))
    request = typesafe_request(goal, parsed)
    payload: dict[str, Any] = {
        "goal_id": goal.goal_id,
        "parsed": {
            "kind": parsed.kind,
            "body": parsed.body,
            "language": parsed.language,
            "had_thought": parsed.had_thought,
        },
        "typesafe_request": request,
        "leanstral_usage": chat.get("usage"),
        "leanstral_finish_reason": chat.get("finish_reason"),
        "advisory_only": True,
    }
    should_call = typesafe_configured() if call_typesafe is None else bool(call_typesafe)
    if should_call:
        verdict = solve_with_typesafe(
            goal,
            str(chat["content"]),
            timeout=typesafe_timeout,
        )
        payload["verdict"] = verdict.to_dict()
    else:
        payload["verdict"] = None
        payload["typesafe_skipped"] = "TYPESAFE_API_KEY is not set"
    return payload


__all__ = [
    "LeanstralGoal",
    "ParsedLeanstralOutput",
    "PROOF_PROMPT",
    "SolverVerdict",
    "discover_leanstral_base_url",
    "draft_flags",
    "leanstral_chat",
    "parse_leanstral_output",
    "propose_and_solve",
    "solve_with_typesafe",
    "compose_draft_quality",
    "solver_questions",
    "typesafe_request",
    "typesafe_state",
    "verdict_from_result",
]
