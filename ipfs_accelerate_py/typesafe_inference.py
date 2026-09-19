"""TypeSafe System One inference client.

TypeSafe is not a chat-completions API. Send ``state`` plus typed
``noul`` / ``choice`` / ``score`` questions and get structured answers.

HTTP: POST https://api.typesafe.ai/v1/systemone
Docs: https://docs.typesafe.ai/sdk/python/usage

Environment (never stored in allocation records):
- ``TYPESAFE_API_KEY`` (required; aliases ``ipfs_accelerate_py_TYPESAFE_API_KEY``,
  ``IPFS_ACCELERATE_PY_TYPESAFE_API_KEY``)
- ``TYPESAFE_BASE_URL`` (default ``https://api.typesafe.ai``)
- ``TYPESAFE_DEFAULT_MODEL`` (default ``jev-latest``)
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT = 30.0
SYSTEMONE_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"
PROVIDER_NAME = "typesafe"

API_KEY_ENV_NAMES: tuple[str, ...] = (
    "TYPESAFE_API_KEY",
    "ipfs_accelerate_py_TYPESAFE_API_KEY",
    "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
    "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
)
BASE_URL_ENV_NAMES: tuple[str, ...] = (
    "TYPESAFE_BASE_URL",
    "ipfs_accelerate_py_TYPESAFE_BASE_URL",
    "IPFS_ACCELERATE_PY_TYPESAFE_BASE_URL",
)
MODEL_ENV_NAMES: tuple[str, ...] = (
    "TYPESAFE_DEFAULT_MODEL",
    "ipfs_accelerate_py_TYPESAFE_MODEL",
    "IPFS_ACCELERATE_PY_TYPESAFE_MODEL",
)

JsonValue = Union[None, bool, int, float, str, list[Any], dict[str, Any]]
QuestionLike = Union["Noul", "Choice", "Score", Mapping[str, Any]]

_LAST_OBSERVATION = threading.local()


class TypeSafeInferenceError(RuntimeError):
    """HTTP or validation failure from System One. Never includes the API key."""

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        request_id: str = "",
    ) -> None:
        super().__init__(message)
        self.status = status
        self.status_code = status
        self.request_id = str(request_id or "")


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    backoff_max: float = 2.0
    timeout: float = DEFAULT_TIMEOUT
    retry_statuses: tuple[int, ...] = (429, 529)


@dataclass
class Noul:
    """Yes/no question. Answer is a probability in [0, 1]."""

    instructions: JsonValue
    criteria: Optional[Mapping[str, JsonValue]] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            payload["criteria"] = dict(self.criteria)
        return payload


def noul_yes_no(
    *,
    true_what: str,
    false_what: str,
    true_examples: Sequence[str] = (),
    false_examples: Sequence[str] = (),
) -> dict[str, Any]:
    """Structured true/false Noul criteria (Advanced: structure)."""

    return {
        "true": {"what": true_what, "examples": list(true_examples)},
        "false": {"what": false_what, "examples": list(false_examples)},
    }


@dataclass
class Choice:
    """Pick one labeled option. Answer includes the winner and probabilities."""

    instructions: JsonValue
    criteria: Mapping[str, Optional[JsonValue]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass
class Score:
    """Rate state against an ordered rubric. Answer can land between levels."""

    instructions: JsonValue
    criteria: Sequence[JsonValue] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": list(self.criteria),
        }


@dataclass(frozen=True)
class NoulAnswer:
    noul: float
    type: str = "noul"
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: Mapping[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    type: str = "choice"
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    legend: Mapping[str, str] = field(default_factory=dict)
    probabilities: Mapping[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    type: str = "score"
    raw: Mapping[str, Any] = field(default_factory=dict)


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


@dataclass
class SystemOneResult:
    model: str
    answers: dict[str, dict[str, Any]]
    usage: dict[str, int]
    raw: dict[str, Any] = field(default_factory=dict)
    nouls: dict[str, NoulAnswer] = field(default_factory=dict)
    choices: dict[str, ChoiceAnswer] = field(default_factory=dict)
    scores: dict[str, ScoreAnswer] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "answers": dict(self.answers),
            "usage": dict(self.usage),
        }

    def to_json(self) -> str:
        return json.dumps(self.answers, separators=(",", ":"), ensure_ascii=False)


def _coalesce_env(*names: str, environ: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if environ is None else environ
    for name in names:
        value = env.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def resolve_typesafe_api_key(environ: Optional[Mapping[str, str]] = None) -> str:
    return _coalesce_env(*API_KEY_ENV_NAMES, environ=environ)


def resolve_typesafe_base_url(environ: Optional[Mapping[str, str]] = None) -> str:
    return (
        _coalesce_env(*BASE_URL_ENV_NAMES, environ=environ) or DEFAULT_BASE_URL
    ).rstrip("/")


def resolve_typesafe_model(environ: Optional[Mapping[str, str]] = None) -> str:
    return _coalesce_env(*MODEL_ENV_NAMES, environ=environ) or DEFAULT_MODEL


def typesafe_configured(environ: Optional[Mapping[str, str]] = None) -> bool:
    return bool(resolve_typesafe_api_key(environ=environ))


def _redact(text: str, secret: str) -> str:
    if not text:
        return ""
    redacted = str(text)
    if secret:
        redacted = redacted.replace(secret, "[redacted]")
    for token in ("Bearer ", "bearer "):
        if token in redacted:
            parts = redacted.split(token, 1)
            rest = parts[1]
            cut = rest.split(None, 1)
            rest_tail = (" " + cut[1]) if len(cut) > 1 else ""
            redacted = parts[0] + token + "[redacted]" + rest_tail
    return redacted


def question_to_dict(question: QuestionLike) -> dict[str, Any]:
    if isinstance(question, (Noul, Choice, Score)):
        return question.to_dict()
    if isinstance(question, Mapping):
        payload = dict(question)
        kind = str(payload.get("type") or "").strip().lower()
        if not kind:
            raise TypeSafeInferenceError("question dict requires a type of noul, choice, or score")
        payload["type"] = kind
        if "instructions" not in payload:
            raise TypeSafeInferenceError("question dict requires instructions")
        return payload
    to_dict = getattr(question, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeSafeInferenceError(
        "questions must be Noul, Choice, Score, or dicts with type and instructions"
    )


def serialize_questions(questions: Mapping[str, QuestionLike]) -> dict[str, dict[str, Any]]:
    if not isinstance(questions, Mapping) or not questions:
        raise TypeSafeInferenceError("questions must be a non-empty mapping")
    return {str(key): question_to_dict(value) for key, value in questions.items()}


def _parse_answer(payload: Mapping[str, Any]) -> Optional[Answer]:
    kind = str(payload.get("type") or "").strip().lower()
    if kind == "noul":
        try:
            value = float(payload.get("noul"))
        except (TypeError, ValueError):
            return None
        return NoulAnswer(noul=value, raw=dict(payload))
    if kind == "choice":
        probs_raw = payload.get("probabilities") if isinstance(payload.get("probabilities"), Mapping) else {}
        probabilities = {}
        for key, value in dict(probs_raw).items():
            try:
                probabilities[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        try:
            confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return ChoiceAnswer(
            choice=str(payload.get("choice") or ""),
            probabilities=probabilities,
            confidence=confidence,
            raw=dict(payload),
        )
    if kind == "score":
        legend_raw = payload.get("legend") if isinstance(payload.get("legend"), Mapping) else {}
        probs_raw = payload.get("probabilities") if isinstance(payload.get("probabilities"), Mapping) else {}
        probabilities = {}
        for key, value in dict(probs_raw).items():
            try:
                probabilities[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        try:
            score = float(payload.get("score"))
        except (TypeError, ValueError):
            return None
        try:
            confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return ScoreAnswer(
            score=score,
            legend={str(key): str(value) for key, value in dict(legend_raw).items()},
            probabilities=probabilities,
            confidence=confidence,
            raw=dict(payload),
        )
    return None


def _parse_result(data: Mapping[str, Any]) -> SystemOneResult:
    answers_raw = data.get("answers") if isinstance(data.get("answers"), Mapping) else {}
    answers = {str(key): dict(value) for key, value in dict(answers_raw).items() if isinstance(value, Mapping)}
    usage_raw = data.get("usage") if isinstance(data.get("usage"), Mapping) else {}
    try:
        input_tokens = int(usage_raw.get("input_tokens") or 0)
    except (TypeError, ValueError):
        input_tokens = 0
    try:
        output_tokens = int(usage_raw.get("output_tokens") or 0)
    except (TypeError, ValueError):
        output_tokens = 0
    nouls: dict[str, NoulAnswer] = {}
    choices: dict[str, ChoiceAnswer] = {}
    scores: dict[str, ScoreAnswer] = {}
    for key, payload in answers.items():
        parsed = _parse_answer(payload)
        if isinstance(parsed, NoulAnswer):
            nouls[key] = parsed
        elif isinstance(parsed, ChoiceAnswer):
            choices[key] = parsed
        elif isinstance(parsed, ScoreAnswer):
            scores[key] = parsed
    return SystemOneResult(
        model=str(data.get("model") or ""),
        answers=answers,
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
        raw=dict(data),
        nouls=nouls,
        choices=choices,
        scores=scores,
    )


def _set_last_observation(payload: Mapping[str, Any]) -> None:
    _LAST_OBSERVATION.value = {
        "model": str(payload.get("model") or "")[:128],
        "input_tokens": int(payload.get("input_tokens") or 0),
        "output_tokens": int(payload.get("output_tokens") or 0),
        "total_tokens": int(payload.get("total_tokens") or 0),
        "status_code": payload.get("status_code"),
        "provider": PROVIDER_NAME,
    }


def get_last_typesafe_observation() -> dict[str, Any]:
    """Usage-only snapshot of the last System One call. Never includes secrets or state."""
    value = getattr(_LAST_OBSERVATION, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def _request_id_from_headers(headers: Any) -> str:
    if headers is None:
        return ""
    for name in ("x-request-id", "X-Request-Id", "request-id"):
        try:
            value = headers.get(name)
        except Exception:
            value = None
        if value:
            return str(value)
    return ""


def _http_json(
    *,
    url: str,
    api_key: str,
    payload: Optional[Mapping[str, Any]] = None,
    method: str = "POST",
    timeout: float,
    retry: RetryPolicy,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    attempts = max(1, int(retry.max_retries) + 1)
    last_error: Optional[BaseException] = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
                status = getattr(response, "status", 200)
                request_id = _request_id_from_headers(getattr(response, "headers", None))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            request_id = _request_id_from_headers(getattr(exc, "headers", None))
            message = _redact(
                f"TypeSafe HTTP {exc.code}: {detail or exc.reason}",
                api_key,
            )
            last_error = TypeSafeInferenceError(
                message,
                status=int(exc.code),
                request_id=request_id,
            )
            if int(exc.code) in retry.retry_statuses and attempt + 1 < attempts:
                delay = min(retry.backoff_max, 0.2 * (2**attempt))
                time.sleep(delay)
                continue
            raise last_error from exc
        except Exception as exc:
            last_error = TypeSafeInferenceError(
                _redact(f"TypeSafe request failed: {exc}", api_key)
            )
            if attempt + 1 < attempts:
                delay = min(retry.backoff_max, 0.2 * (2**attempt))
                time.sleep(delay)
                continue
            raise last_error from exc
        try:
            data = json.loads(raw)
        except Exception as exc:
            raise TypeSafeInferenceError(
                "TypeSafe returned invalid JSON",
                status=int(status) if status else None,
                request_id=request_id,
            ) from exc
        if not isinstance(data, dict):
            raise TypeSafeInferenceError(
                "TypeSafe returned invalid JSON",
                status=int(status) if status else None,
                request_id=request_id,
            )
        data.setdefault("_request_id", request_id)
        return data
    raise last_error or TypeSafeInferenceError("TypeSafe request failed")


def system_one(
    state: JsonValue,
    questions: Mapping[str, QuestionLike],
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    extra_body: Optional[Mapping[str, Any]] = None,
    retry: Optional[RetryPolicy] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> SystemOneResult:
    """Evaluate ``state`` against typed System One questions."""

    key = str(api_key or "").strip() or resolve_typesafe_api_key(environ=environ)
    if not key:
        raise TypeSafeInferenceError(
            "TYPESAFE_API_KEY is required for TypeSafe System One inference"
        )
    root = str(base_url or "").strip() or resolve_typesafe_base_url(environ=environ)
    model_name = str(model or "").strip() or resolve_typesafe_model(environ=environ)
    policy = retry or RetryPolicy()
    wait = float(timeout if timeout is not None else policy.timeout or DEFAULT_TIMEOUT)
    payload: dict[str, Any] = {
        "state": state,
        "model": model_name,
        "questions": serialize_questions(questions),
    }
    if extra_body:
        payload.update(dict(extra_body))
    url = f"{root.rstrip('/')}{SYSTEMONE_PATH}"
    data = _http_json(
        url=url,
        api_key=key,
        payload=payload,
        timeout=wait,
        retry=policy,
    )
    result = _parse_result(data)
    _set_last_observation(
        {
            "model": result.model or model_name,
            "input_tokens": result.usage.get("input_tokens") or 0,
            "output_tokens": result.usage.get("output_tokens") or 0,
            "total_tokens": int(result.usage.get("input_tokens") or 0)
            + int(result.usage.get("output_tokens") or 0),
            "status_code": 200,
        }
    )
    return result


def list_typesafe_models(
    *,
    live: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 4.0,
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[str, ...]:
    """Return known System One models. Live listing is fail-soft."""

    defaults = (DEFAULT_MODEL, "jev")
    if not live:
        return defaults
    key = str(api_key or "").strip() or resolve_typesafe_api_key(environ=environ)
    if not key:
        return defaults
    root = str(base_url or "").strip() or resolve_typesafe_base_url(environ=environ)
    try:
        data = _http_json(
            url=f"{root.rstrip('/')}{MODELS_PATH}",
            api_key=key,
            payload=None,
            method="GET",
            timeout=max(0.5, float(timeout)),
            retry=RetryPolicy(max_retries=0, timeout=timeout),
        )
    except Exception:
        return defaults
    rows = data.get("data") if isinstance(data.get("data"), list) else data.get("models")
    names: list[str] = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, str) and row.strip():
                names.append(row.strip())
            elif isinstance(row, Mapping):
                ident = str(row.get("id") or row.get("name") or "").strip()
                if ident:
                    names.append(ident)
    return tuple(names[:32]) if names else defaults


class TypeSafeClient:
    """Synchronous System One client matching the TypeSafe Python SDK usage."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        retry: Optional[RetryPolicy] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.api_key = str(api_key or "").strip() or resolve_typesafe_api_key(environ=environ)
        self.base_url = str(base_url or "").strip() or resolve_typesafe_base_url(environ=environ)
        self.model = str(model or "").strip() or resolve_typesafe_model(environ=environ)
        self.timeout = float(timeout if timeout is not None else DEFAULT_TIMEOUT)
        self.retry = retry or RetryPolicy(timeout=self.timeout)

    def system_one(
        self,
        state: JsonValue,
        questions: Mapping[str, QuestionLike],
        *,
        model: Optional[str] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        retry: Optional[RetryPolicy] = None,
        timeout: Optional[float] = None,
    ) -> SystemOneResult:
        return system_one(
            state,
            questions,
            model=model or self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout if timeout is None else float(timeout),
            extra_body=extra_body,
            retry=retry or self.retry,
        )

    def close(self) -> None:
        return None

    def __enter__(self) -> "TypeSafeClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


__all__ = [
    "API_KEY_ENV_NAMES",
    "Choice",
    "ChoiceAnswer",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "Noul",
    "NoulAnswer",
    "noul_yes_no",
    "PROVIDER_NAME",
    "RetryPolicy",
    "Score",
    "ScoreAnswer",
    "SystemOneResult",
    "TypeSafeClient",
    "TypeSafeInferenceError",
    "get_last_typesafe_observation",
    "list_typesafe_models",
    "question_to_dict",
    "resolve_typesafe_api_key",
    "resolve_typesafe_base_url",
    "resolve_typesafe_model",
    "serialize_questions",
    "system_one",
    "typesafe_configured",
]
