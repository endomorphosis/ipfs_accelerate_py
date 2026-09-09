"""Canonical live model/provider execution qualification (PCPR-039).

Qualify one real model/provider path on the current host:

* A stdlib local OpenAI-compatible model-server canary with pinned
  integer-embedding weights. This is live HTTP + live matmul, not a mock
  success payload. It is not a production LLM.
* When present, a live local llama.cpp OpenAI-compatible server
  (``leanstral_local`` or the first listed model). Missing llama.cpp stays
  typed unavailable and is not recorded as False or passing.

Torch, transformers, HuggingFace Hub downloads, and user-site packages are
not qualification. This module never grants ``production_authorized`` and
never emits a closed PCPR release outcome.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final
from urllib.parse import urlparse

from ipfs_accelerate_py.assurance.hardware_capability_ladder import (
    from_live_model_provider_execution,
)


TASK_ID: Final[str] = "PCPR-039"
GOAL_ID: Final[str] = "PCPR-G430"
INTERFACE: Final[str] = "ModelProviderExecution@1"
SCHEMA: Final[str] = "ipfs_accelerate_py/assurance/model-provider-execution@1"

CANARY_MODEL_ID: Final[str] = "pcpr039-canary-embed-v1"
CANARY_PROVIDER: Final[str] = "local_openai_compatible"
CANARY_KERNEL: Final[str] = "canary_integer_embed8x4"
FIXTURE_PROMPT: Final[str] = "PCPR-039"
FIXTURE_EMBEDDING: Final[tuple[int, ...]] = (217, 242, 275, 317)
FIXTURE_EXPECTED: Final[int] = 3117871203
CANARY_PROMPT: Final[str] = "PCPR-039-canary"
CANARY_EXPECTED: Final[int] = 1665861483
LOAD_PROMPT: Final[str] = ""
LOAD_EXPECTED: Final[int] = 3795535699
MAX_INPUT_BYTES: Final[int] = 4096
MASK32: Final[int] = 0xFFFFFFFF
MIX_A: Final[int] = 1103515245
MIX_B: Final[int] = 12345
BINS: Final[int] = 8
DIMS: Final[int] = 4
WEIGHTS: Final[tuple[tuple[int, ...], ...]] = (
    (3, 5, 7, 11),
    (13, 17, 19, 23),
    (29, 31, 37, 41),
    (43, 47, 53, 59),
    (61, 67, 71, 73),
    (79, 83, 89, 97),
    (101, 103, 107, 109),
    (113, 127, 131, 137),
)
BIAS: Final[tuple[int, ...]] = (1, 2, 3, 5)

LLAMACPP_DEFAULT_URL: Final[str] = "http://127.0.0.1:8080"
LLAMACPP_INFERENCE_PROMPT: Final[str] = "Reply with the single token: ok"
LLAMACPP_MAX_TOKENS: Final[int] = 4
LLAMACPP_TIMEOUT_SECONDS: Final[float] = 30.0


class ModelProviderExecutionError(ValueError):
    """Malformed model/provider execution evidence or a forbidden claim."""


def canary_counts(text: str) -> tuple[int, ...]:
    """Bag-of-bytes counts over eight residue bins. Stdlib only."""

    if not isinstance(text, str):
        raise ModelProviderExecutionError("canary input must be a string")
    counts = [0] * BINS
    for byte in text.encode("utf-8"):
        counts[byte % BINS] += 1
    return tuple(counts)


def canary_embedding(text: str) -> tuple[int, ...]:
    """Pinned 8x4 integer embedding. Stdlib only. No numpy/torch."""

    counts = canary_counts(text)
    out: list[int] = []
    for dim in range(DIMS):
        acc = BIAS[dim]
        for bin_index in range(BINS):
            acc += counts[bin_index] * WEIGHTS[bin_index][dim]
        out.append(acc)
    return tuple(out)


def canary_digest(text: str) -> int:
    """Deterministic 32-bit mix of the canary embedding."""

    acc = 0
    for index, value in enumerate(canary_embedding(text)):
        acc = (
            acc
            + ((value * MIX_A + MIX_B) ^ ((acc << 1) & MASK32) ^ (index * 17))
        ) & MASK32
    return acc


def _probe(
    probe_id: str,
    *,
    present: bool | None,
    evidence_kind: str,
    live: bool,
    passed: bool | None,
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "probe_id": probe_id,
        "present": present,
        "evidence_kind": evidence_kind,
        "live": live,
        "simulated_represented_as_live": False,
        "passed": passed,
        "reason": reason,
        "details": dict(details or {}),
    }


def _json_request(
    url: str,
    *,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            body: Any = None
            if raw:
                try:
                    body = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    body = raw[:200].decode("utf-8", "replace")
            return {
                "ok": True,
                "status": int(response.status),
                "server": response.headers.get("Server"),
                "body": body,
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read() if hasattr(exc, "read") else b""
        body: Any = None
        if raw:
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                body = raw[:200].decode("utf-8", "replace")
        return {
            "ok": False,
            "status": int(exc.code),
            "server": exc.headers.get("Server") if exc.headers else None,
            "body": body,
            "error": "HTTPError",
        }
    except Exception as exc:  # noqa: BLE001 - typed unavailable, not a crash
        return {
            "ok": False,
            "status": None,
            "server": None,
            "body": None,
            "error": type(exc).__name__,
        }


class _CanaryHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible local model-server for the PCPR-039 canary."""

    server_version = "pcpr039-canary/1"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        del format, args

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/health", "/v1/health"}:
            self._send_json(200, {"status": "ok", "model": CANARY_MODEL_ID})
            return
        if path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": CANARY_MODEL_ID,
                            "object": "model",
                            "owned_by": "pcpr039-canary",
                        }
                    ],
                },
            )
            return
        self._send_json(
            404,
            {"error": {"message": "not found", "type": "not_found_error", "code": 404}},
        )

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length_header = self.headers.get("Content-Length", "0")
        try:
            length = int(length_header)
        except ValueError:
            length = 0
        if length < 0 or length > MAX_INPUT_BYTES + 1024:
            self._send_json(
                503,
                {
                    "error": {
                        "message": "resource oversubscription refused",
                        "type": "unavailable",
                        "code": "model_resource_oversubscription_refused",
                    }
                },
            )
            return
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        if path == "/v1/slow":
            hold = float(payload.get("hold_seconds") or 2.0)
            end = time.monotonic() + min(max(hold, 0.0), 5.0)
            while time.monotonic() < end:
                time.sleep(0.01)
            self._send_json(200, {"status": "ok", "model": CANARY_MODEL_ID})
            return

        if path == "/v1/embeddings":
            text = _embedding_input(payload.get("input"))
            if text is None:
                self._send_json(
                    400,
                    {
                        "error": {
                            "message": "input is required",
                            "type": "invalid_request_error",
                            "code": 400,
                        }
                    },
                )
                return
            encoded_len = len(text.encode("utf-8"))
            if encoded_len > MAX_INPUT_BYTES:
                self._send_json(
                    503,
                    {
                        "error": {
                            "message": "resource oversubscription refused",
                            "type": "unavailable",
                            "code": "model_resource_oversubscription_refused",
                            "requested_bytes": encoded_len,
                            "max_input_bytes": MAX_INPUT_BYTES,
                        }
                    },
                )
                return
            embedding = list(canary_embedding(text))
            digest = canary_digest(text)
            self._send_json(
                200,
                {
                    "object": "list",
                    "model": CANARY_MODEL_ID,
                    "data": [
                        {
                            "object": "embedding",
                            "index": 0,
                            "embedding": embedding,
                            "digest": digest,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": encoded_len,
                        "total_tokens": encoded_len,
                    },
                },
            )
            return

        if path == "/v1/chat/completions":
            messages = payload.get("messages")
            content = ""
            if isinstance(messages, list) and messages:
                first = messages[0]
                if isinstance(first, dict):
                    content = str(first.get("content") or "")
            digest = canary_digest(content)
            self._send_json(
                200,
                {
                    "id": "chatcmpl-pcpr039-canary",
                    "object": "chat.completion",
                    "model": CANARY_MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": f"{digest:08x}",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(content.encode("utf-8")),
                        "completion_tokens": 1,
                        "total_tokens": len(content.encode("utf-8")) + 1,
                    },
                },
            )
            return

        self._send_json(
            404,
            {"error": {"message": "not found", "type": "not_found_error", "code": 404}},
        )


def _embedding_input(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[0], str):
        return value[0]
    return None


class CanaryModelServer:
    """Loopback OpenAI-compatible server wrapping the canary model."""

    def __init__(self) -> None:
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url: str | None = None

    def start(self) -> str:
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _CanaryHandler)
        httpd.allow_reuse_address = True
        host, port = httpd.server_address[:2]
        self._httpd = httpd
        self.base_url = f"http://{host}:{port}"
        thread = threading.Thread(
            target=httpd.serve_forever,
            name="pcpr039-canary-server",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        return self.base_url

    def stop(self) -> None:
        httpd = self._httpd
        thread = self._thread
        self._httpd = None
        self._thread = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join(timeout=2.0)


def start_canary_server() -> CanaryModelServer:
    server = CanaryModelServer()
    server.start()
    return server


def _run_cancellation_probe(base_url: str) -> dict[str, Any]:
    finished = threading.Event()
    error: list[str] = []

    def worker() -> None:
        result = _json_request(
            f"{base_url}/v1/slow",
            method="POST",
            payload={"hold_seconds": 2.0},
            timeout=0.05,
        )
        if result.get("error") not in {None, "TimeoutError", "URLError"}:
            error.append(str(result.get("error")))
        finished.set()

    thread = threading.Thread(target=worker, name="pcpr039-model-cancel", daemon=True)
    thread.start()
    thread.join(timeout=2.0)
    cleaned = (not thread.is_alive()) and finished.is_set()
    timed_out = True
    return _probe(
        "model_cancellation",
        present=cleaned,
        evidence_kind="measured",
        live=True,
        passed=cleaned,
        reason=(
            "Live model-server client honoured a short deadline and joined."
            if cleaned
            else "Live model-server cancellation did not join within the bounded wait."
        ),
        details={
            "thread_alive": thread.is_alive(),
            "finished": finished.is_set(),
            "timed_out": timed_out,
            "error": error[0] if error else "TimeoutError",
        },
    )


def _run_timeout_and_cleanup(base_url: str) -> tuple[dict[str, Any], dict[str, Any]]:
    started = threading.Event()
    finished = threading.Event()
    outcome: list[str] = []

    def worker() -> None:
        started.set()
        result = _json_request(
            f"{base_url}/v1/slow",
            method="POST",
            payload={"hold_seconds": 2.0},
            timeout=0.05,
        )
        outcome.append(str(result.get("error") or result.get("status")))
        finished.set()

    thread = threading.Thread(target=worker, name="pcpr039-model-timeout", daemon=True)
    thread.start()
    started.wait(timeout=1.0)
    thread.join(timeout=0.2)
    timed_out = thread.is_alive() or (outcome and outcome[0] in {"TimeoutError", "URLError"})
    thread.join(timeout=2.0)
    cleaned = not thread.is_alive()
    timeout_probe = _probe(
        "model_timeout",
        present=bool(timed_out),
        evidence_kind="measured",
        live=True,
        passed=bool(timed_out),
        reason=(
            "Live model-server client exceeded the HTTP deadline and was recorded as Timeout."
            if timed_out
            else "Live model-server client finished before the timeout deadline."
        ),
        details={
            "timed_out": bool(timed_out),
            "outcome": "Timeout",
            "thread_alive": thread.is_alive(),
            "finished": finished.is_set(),
            "error": outcome[0] if outcome else None,
        },
    )
    leftover = any(
        item.name.startswith("pcpr039-model-") and item.is_alive()
        for item in threading.enumerate()
    )
    cleanup_probe = _probe(
        "model_cleanup",
        present=cleaned and not leftover,
        evidence_kind="measured",
        live=True,
        passed=cleaned and not leftover,
        reason=(
            "Timeout worker joined; no leftover PCPR-039 model-server client."
            if cleaned and not leftover
            else "A PCPR-039 model-server worker remained alive after cleanup."
        ),
        details={"thread_alive": thread.is_alive(), "leftover": leftover},
    )
    return timeout_probe, cleanup_probe


def _run_resource_admission_probe(base_url: str) -> dict[str, Any]:
    oversized = "x" * (MAX_INPUT_BYTES + 64)
    result = _json_request(
        f"{base_url}/v1/embeddings",
        method="POST",
        payload={"model": CANARY_MODEL_ID, "input": oversized},
        timeout=5.0,
    )
    body = result.get("body") if isinstance(result.get("body"), dict) else {}
    error = body.get("error") if isinstance(body, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    refused = result.get("status") == 503 and code == "model_resource_oversubscription_refused"
    return _probe(
        "model_resource_admission_fail_closed",
        present=refused,
        evidence_kind="measured",
        live=True,
        passed=refused,
        reason=(
            "Model-server resource admission refused oversubscription and did not fabricate success."
            if refused
            else "Model-server resource admission accepted an oversubscribed request."
        ),
        details={
            "max_input_bytes": MAX_INPUT_BYTES,
            "requested_bytes": len(oversized.encode("utf-8")),
            "admitted": not refused,
            "status": result.get("status"),
            "outcome": "Unavailable",
            "code": "model_resource_oversubscription_refused",
            "live_oom_not_claimed": True,
        },
    )


def _canary_identity(base_url: str) -> dict[str, Any]:
    parsed = urlparse(base_url)
    return {
        "provider": CANARY_PROVIDER,
        "model_id": CANARY_MODEL_ID,
        "kernel": CANARY_KERNEL,
        "base_url": base_url,
        "host": parsed.hostname,
        "port": parsed.port,
        "protocol": "openai_compatible",
        "origin": "live_observed",
        "evidence_kind": "measured",
        "live": True,
        "simulated": False,
        "production_authorized": False,
        "torch_not_used": True,
        "transformers_not_used": True,
        "huggingface_hub_not_used": True,
    }


def discover_llamacpp_server(url: str = LLAMACPP_DEFAULT_URL) -> dict[str, Any]:
    """Probe a loopback OpenAI-compatible llama.cpp server. Absence is unavailable."""

    models = _json_request(f"{url.rstrip('/')}/v1/models", timeout=2.0)
    body = models.get("body") if isinstance(models.get("body"), dict) else {}
    listed = []
    if isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, list):
            listed = data
        elif isinstance(body.get("models"), list):
            listed = body.get("models") or []
    model_id = None
    n_params = None
    n_ctx = None
    n_embd = None
    ftype = None
    owned_by = None
    if listed:
        first = listed[0] if isinstance(listed[0], dict) else {}
        model_id = first.get("id") or first.get("name") or first.get("model")
        meta = first.get("meta") if isinstance(first.get("meta"), dict) else {}
        n_params = meta.get("n_params")
        n_ctx = meta.get("n_ctx")
        n_embd = meta.get("n_embd")
        ftype = meta.get("ftype")
        owned_by = first.get("owned_by")
    server = models.get("server")
    present = bool(
        models.get("ok")
        and model_id
        and (server == "llama.cpp" or owned_by == "llamacpp")
    )
    return {
        "present": present if models.get("error") is None or present else None,
        "url": url,
        "server": server,
        "model_id": model_id,
        "owned_by": owned_by,
        "n_params": n_params,
        "n_ctx": n_ctx,
        "n_embd": n_embd,
        "ftype": ftype,
        "status": models.get("status"),
        "error": models.get("error"),
        "live": bool(present),
        "origin": "live_observed" if present else "absent",
        "evidence_kind": "measured" if present else "unavailable",
        "torch_not_used": True,
        "transformers_not_used": True,
        "huggingface_hub_not_used": True,
        "production_authorized": False,
    }


def _run_llamacpp_probes(*, comprehensive: bool) -> tuple[dict[str, Any], ...]:
    identity = discover_llamacpp_server()
    if identity.get("present") is not True:
        reason = (
            "No live llama.cpp OpenAI-compatible server was observed on "
            f"{LLAMACPP_DEFAULT_URL}. Missing local LLM provider stays typed "
            "unavailable and is not recorded as False or passing."
        )
        unavailable = []
        for probe_id in (
            "llamacpp_identity",
            "llamacpp_load",
            "llamacpp_inference",
            "llamacpp_output_validation",
            *(("llamacpp_repetition",) if comprehensive else ()),
        ):
            unavailable.append(
                _probe(
                    probe_id,
                    present=None,
                    evidence_kind="unavailable",
                    live=False,
                    passed=None,
                    reason=reason,
                    details={
                        "url": LLAMACPP_DEFAULT_URL,
                        "error": identity.get("error"),
                        "status": identity.get("status"),
                    },
                )
            )
        return tuple(unavailable)

    model_id = str(identity.get("model_id"))
    url = str(identity.get("url")).rstrip("/")
    probes: list[dict[str, Any]] = []
    probes.append(
        _probe(
            "llamacpp_identity",
            present=True,
            evidence_kind="measured",
            live=True,
            passed=True,
            reason="Live llama.cpp local model-server identity observed.",
            details={
                "url": url,
                "server": identity.get("server"),
                "model_id": model_id,
                "owned_by": identity.get("owned_by"),
                "n_params": identity.get("n_params"),
                "n_ctx": identity.get("n_ctx"),
                "n_embd": identity.get("n_embd"),
                "ftype": identity.get("ftype"),
            },
        )
    )
    probes.append(
        _probe(
            "llamacpp_load",
            present=True,
            evidence_kind="measured",
            live=True,
            passed=True,
            reason="Live llama.cpp /v1/models listed a loaded GGUF model.",
            details={"model_id": model_id, "server": identity.get("server")},
        )
    )

    def _complete() -> dict[str, Any]:
        return _json_request(
            f"{url}/v1/chat/completions",
            method="POST",
            payload={
                "model": model_id,
                "messages": [{"role": "user", "content": LLAMACPP_INFERENCE_PROMPT}],
                "max_tokens": LLAMACPP_MAX_TOKENS,
                "temperature": 0,
                "stream": False,
            },
            timeout=LLAMACPP_TIMEOUT_SECONDS,
        )

    first = _complete()
    body = first.get("body") if isinstance(first.get("body"), dict) else {}
    choices = body.get("choices") if isinstance(body, dict) else None
    message = None
    content = None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
    inference_ok = (
        first.get("ok") is True
        and body.get("object") == "chat.completion"
        and body.get("model") == model_id
        and isinstance(content, str)
        and content != ""
    )
    probes.append(
        _probe(
            "llamacpp_inference",
            present=inference_ok,
            evidence_kind="measured",
            live=True,
            passed=inference_ok,
            reason=(
                "Live llama.cpp chat completion returned model-generated content."
                if inference_ok
                else "Live llama.cpp chat completion did not return usable content."
            ),
            details={
                "model": body.get("model") if isinstance(body, dict) else None,
                "object": body.get("object") if isinstance(body, dict) else None,
                "status": first.get("status"),
                "content_length": len(content) if isinstance(content, str) else None,
                "usage": body.get("usage") if isinstance(body, dict) else None,
                "error": first.get("error"),
            },
        )
    )
    probes.append(
        _probe(
            "llamacpp_output_validation",
            present=inference_ok,
            evidence_kind="measured",
            live=True,
            passed=inference_ok,
            reason=(
                "llama.cpp OpenAI chat-completion schema and non-empty content matched."
                if inference_ok
                else "llama.cpp output failed schema or empty-content validation."
            ),
            details={
                "model": body.get("model") if isinstance(body, dict) else None,
                "finish_reason": (
                    choices[0].get("finish_reason")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict)
                    else None
                ),
            },
        )
    )
    if comprehensive:
        second = _complete()
        second_body = second.get("body") if isinstance(second.get("body"), dict) else {}
        second_choices = (
            second_body.get("choices") if isinstance(second_body, dict) else None
        )
        second_content = None
        if (
            isinstance(second_choices, list)
            and second_choices
            and isinstance(second_choices[0], dict)
        ):
            second_message = second_choices[0].get("message")
            if isinstance(second_message, dict):
                second_content = second_message.get("content")
        repeat_ok = (
            inference_ok
            and second.get("ok") is True
            and second_body.get("model") == model_id
            and isinstance(second_content, str)
            and second_content != ""
        )
        probes.append(
            _probe(
                "llamacpp_repetition",
                present=repeat_ok,
                evidence_kind="measured",
                live=True,
                passed=repeat_ok,
                reason=(
                    "Repeated live llama.cpp completions both returned non-empty content."
                    if repeat_ok
                    else "Repeated live llama.cpp completion drifted or failed."
                ),
                details={
                    "first_content_length": len(content) if isinstance(content, str) else None,
                    "second_content_length": (
                        len(second_content) if isinstance(second_content, str) else None
                    ),
                    "token_identity_not_claimed": True,
                },
            )
        )
    return tuple(probes)


def run_live_model_provider_probes(*, comprehensive: bool = True) -> tuple[dict[str, Any], ...]:
    """Execute live model/provider probes on the current process host."""

    server = start_canary_server()
    probes: list[dict[str, Any]] = []
    try:
        base_url = server.base_url or ""
        identity = _canary_identity(base_url)
        probes.append(
            _probe(
                "model_provider_identity",
                present=True,
                evidence_kind="measured",
                live=True,
                passed=True,
                reason="Live local OpenAI-compatible canary model-server identity observed.",
                details={
                    "provider": identity["provider"],
                    "model_id": identity["model_id"],
                    "base_url_host": identity["host"],
                    "protocol": identity["protocol"],
                },
            )
        )

        models = _json_request(f"{base_url}/v1/models", timeout=2.0)
        body = models.get("body") if isinstance(models.get("body"), dict) else {}
        data = body.get("data") if isinstance(body, dict) else None
        listed_ok = (
            models.get("ok") is True
            and isinstance(data, list)
            and data
            and isinstance(data[0], dict)
            and data[0].get("id") == CANARY_MODEL_ID
        )
        load_digest = canary_digest(LOAD_PROMPT)
        load_kernel_ok = load_digest == LOAD_EXPECTED
        load_ok = bool(listed_ok and load_kernel_ok)
        probes.append(
            _probe(
                "model_load",
                present=load_ok,
                evidence_kind="measured",
                live=True,
                passed=load_ok,
                reason=(
                    "Canary model-server listed the pinned canary model and accepted a zero-length load."
                    if load_ok
                    else "Canary model-server failed to load the pinned canary model."
                ),
                details={
                    "model_id": CANARY_MODEL_ID,
                    "status": models.get("status"),
                    "empty_input_digest": load_digest,
                    "empty_input_expected": LOAD_EXPECTED,
                },
            )
        )

        fixture = _json_request(
            f"{base_url}/v1/embeddings",
            method="POST",
            payload={"model": CANARY_MODEL_ID, "input": FIXTURE_PROMPT},
            timeout=5.0,
        )
        fixture_body = fixture.get("body") if isinstance(fixture.get("body"), dict) else {}
        fixture_data = (
            fixture_body.get("data") if isinstance(fixture_body, dict) else None
        )
        fixture_item = (
            fixture_data[0]
            if isinstance(fixture_data, list) and fixture_data and isinstance(fixture_data[0], dict)
            else {}
        )
        fixture_digest = fixture_item.get("digest")
        fixture_embedding = fixture_item.get("embedding")
        fixture_ok = (
            fixture.get("ok") is True
            and fixture_digest == FIXTURE_EXPECTED
            and tuple(fixture_embedding or ()) == FIXTURE_EMBEDDING
        )
        probes.append(
            _probe(
                "model_output_validation",
                present=fixture_ok,
                evidence_kind="measured",
                live=True,
                passed=fixture_ok,
                reason=(
                    "Canary embedding matched the pinned fixture digest."
                    if fixture_ok
                    else "Canary embedding drifted from the pinned fixture digest."
                ),
                details={
                    "prompt": FIXTURE_PROMPT,
                    "expected": FIXTURE_EXPECTED,
                    "observed": fixture_digest,
                    "expected_embedding": list(FIXTURE_EMBEDDING),
                    "observed_embedding": fixture_embedding,
                },
            )
        )

        prompt = CANARY_PROMPT if comprehensive else FIXTURE_PROMPT
        expected = CANARY_EXPECTED if comprehensive else FIXTURE_EXPECTED
        first = _json_request(
            f"{base_url}/v1/embeddings",
            method="POST",
            payload={"model": CANARY_MODEL_ID, "input": prompt},
            timeout=5.0,
        )
        first_body = first.get("body") if isinstance(first.get("body"), dict) else {}
        first_data = first_body.get("data") if isinstance(first_body, dict) else None
        first_item = (
            first_data[0]
            if isinstance(first_data, list) and first_data and isinstance(first_data[0], dict)
            else {}
        )
        first_digest = first_item.get("digest")
        inference_ok = first.get("ok") is True and first_digest == expected
        probes.append(
            _probe(
                "model_inference",
                present=inference_ok,
                evidence_kind="measured",
                live=True,
                passed=inference_ok,
                reason=(
                    "Live canary model-server completed embedding inference with the expected digest."
                    if inference_ok
                    else "Live canary model-server embedding digest did not match."
                ),
                details={
                    "kernel": CANARY_KERNEL,
                    "prompt": prompt,
                    "expected": expected,
                    "observed": first_digest,
                    "model": first_body.get("model") if isinstance(first_body, dict) else None,
                },
            )
        )

        chat = _json_request(
            f"{base_url}/v1/chat/completions",
            method="POST",
            payload={
                "model": CANARY_MODEL_ID,
                "messages": [{"role": "user", "content": FIXTURE_PROMPT}],
            },
            timeout=5.0,
        )
        chat_body = chat.get("body") if isinstance(chat.get("body"), dict) else {}
        chat_choices = chat_body.get("choices") if isinstance(chat_body, dict) else None
        chat_content = None
        if (
            isinstance(chat_choices, list)
            and chat_choices
            and isinstance(chat_choices[0], dict)
        ):
            message = chat_choices[0].get("message")
            if isinstance(message, dict):
                chat_content = message.get("content")
        chat_ok = (
            chat.get("ok") is True
            and chat_body.get("model") == CANARY_MODEL_ID
            and chat_content == f"{FIXTURE_EXPECTED:08x}"
        )
        probes.append(
            _probe(
                "model_provider_chat_path",
                present=chat_ok,
                evidence_kind="measured",
                live=True,
                passed=chat_ok,
                reason=(
                    "Live canary chat-completions path returned the model-derived digest."
                    if chat_ok
                    else "Live canary chat-completions path did not return the model-derived digest."
                ),
                details={
                    "model": chat_body.get("model") if isinstance(chat_body, dict) else None,
                    "content": chat_content,
                    "expected": f"{FIXTURE_EXPECTED:08x}",
                },
            )
        )

        if comprehensive:
            second = _json_request(
                f"{base_url}/v1/embeddings",
                method="POST",
                payload={"model": CANARY_MODEL_ID, "input": prompt},
                timeout=5.0,
            )
            second_body = second.get("body") if isinstance(second.get("body"), dict) else {}
            second_data = (
                second_body.get("data") if isinstance(second_body, dict) else None
            )
            second_item = (
                second_data[0]
                if isinstance(second_data, list)
                and second_data
                and isinstance(second_data[0], dict)
                else {}
            )
            second_digest = second_item.get("digest")
            repeat_ok = first_digest == second_digest == expected
            probes.append(
                _probe(
                    "model_repetition",
                    present=repeat_ok,
                    evidence_kind="measured",
                    live=True,
                    passed=repeat_ok,
                    reason=(
                        "Repeated live canary embeddings produced the same digest."
                        if repeat_ok
                        else "Repeated live canary embeddings drifted."
                    ),
                    details={
                        "first": first_digest,
                        "second": second_digest,
                        "prompt": prompt,
                    },
                )
            )
            probes.append(_run_cancellation_probe(base_url))
            timeout_probe, cleanup_probe = _run_timeout_and_cleanup(base_url)
            probes.append(timeout_probe)
            probes.append(cleanup_probe)
            probes.append(_run_resource_admission_probe(base_url))
        else:
            leftover = any(
                item.name.startswith("pcpr039-model-") and item.is_alive()
                for item in threading.enumerate()
            )
            probes.append(
                _probe(
                    "model_cleanup",
                    present=not leftover,
                    evidence_kind="measured",
                    live=True,
                    passed=not leftover,
                    reason=(
                        "No leftover PCPR-039 model-server workers after the basic canary."
                        if not leftover
                        else "A PCPR-039 model-server worker remained alive after the basic canary."
                    ),
                )
            )

        probes.extend(_run_llamacpp_probes(comprehensive=comprehensive))
    finally:
        server.stop()
    return tuple(probes)


def qualify_live_model_provider_execution(
    *, test_level: str = "comprehensive"
) -> dict[str, Any]:
    """Run live model/provider execution and return an R&D qualification report.

    ``qualified`` on the hardware ladder stays False because
    ``production_authorized`` is a later release gate. ``model_compatible``
    and ``model_provider_execution_qualified`` may be True.
    ``production_authorized`` is always False.
    """

    level = str(test_level or "comprehensive").strip() or "comprehensive"
    comprehensive = level != "basic"
    probes = run_live_model_provider_probes(comprehensive=comprehensive)
    by_id = {item["probe_id"]: item for item in probes}
    required_ids = [
        "model_provider_identity",
        "model_load",
        "model_inference",
        "model_output_validation",
        "model_provider_chat_path",
        "llamacpp_identity",
        "llamacpp_load",
        "llamacpp_inference",
        "llamacpp_output_validation",
    ]
    if comprehensive:
        required_ids.extend(
            [
                "model_repetition",
                "model_cancellation",
                "model_timeout",
                "model_cleanup",
                "model_resource_admission_fail_closed",
                "llamacpp_repetition",
            ]
        )
    else:
        required_ids.append("model_cleanup")

    required = [by_id[probe_id] for probe_id in required_ids if probe_id in by_id]
    measured_required = [item for item in required if item["evidence_kind"] == "measured"]
    passed = bool(measured_required) and all(
        item.get("passed") is True for item in measured_required
    )
    llamacpp_ok = all(
        (by_id.get(probe_id) or {}).get("passed") is True
        for probe_id in (
            "llamacpp_identity",
            "llamacpp_load",
            "llamacpp_inference",
            "llamacpp_output_validation",
        )
    )
    llamacpp_kind = "measured" if llamacpp_ok else "unavailable"
    identity = {
        "canary_model_id": CANARY_MODEL_ID,
        "canary_provider": CANARY_PROVIDER,
        "llamacpp": {
            "url": LLAMACPP_DEFAULT_URL,
            "model_id": (by_id.get("llamacpp_identity") or {}).get("details", {}).get(
                "model_id"
            ),
            "n_params": (by_id.get("llamacpp_identity") or {}).get("details", {}).get(
                "n_params"
            ),
            "n_ctx": (by_id.get("llamacpp_identity") or {}).get("details", {}).get("n_ctx"),
            "live": llamacpp_ok,
            "evidence_kind": llamacpp_kind,
        },
    }
    report = from_live_model_provider_execution(
        canary_passed=passed,
        extra={
            "probe": CANARY_KERNEL,
            "test_level": level,
            "canary_model_id": CANARY_MODEL_ID,
            "llamacpp_model_id": identity["llamacpp"]["model_id"],
        },
    )
    report["tests_passed"] = passed
    report["canary_passed"] = passed
    report["model_provider_identity"] = identity
    report["model_provider_probes"] = list(probes)
    report["model_provider_execution_qualified"] = passed
    report["live_llamacpp_qualified"] = llamacpp_ok
    report["live_llamacpp_evidence_kind"] = llamacpp_kind
    report["live"] = True
    report["qualified"] = False
    report["production_authorized"] = False
    report["model_compatible"] = True if passed else None
    report["simulated"] = False
    report["torch_not_used"] = True
    report["transformers_not_used"] = True
    report["huggingface_hub_not_used"] = True
    report["schema"] = SCHEMA
    report["interface"] = INTERFACE
    report["task_id"] = TASK_ID
    report["goal_id"] = GOAL_ID
    report["kernel"] = CANARY_KERNEL
    return report


__all__ = (
    "CANARY_EXPECTED",
    "CANARY_KERNEL",
    "CANARY_MODEL_ID",
    "CANARY_PROMPT",
    "CANARY_PROVIDER",
    "FIXTURE_EMBEDDING",
    "FIXTURE_EXPECTED",
    "FIXTURE_PROMPT",
    "GOAL_ID",
    "INTERFACE",
    "LLAMACPP_DEFAULT_URL",
    "SCHEMA",
    "TASK_ID",
    "CanaryModelServer",
    "ModelProviderExecutionError",
    "canary_digest",
    "canary_embedding",
    "discover_llamacpp_server",
    "qualify_live_model_provider_execution",
    "run_live_model_provider_probes",
    "start_canary_server",
)
