"""Prompt-only public Python facade for agent supervisor.

Normal callers supply only a prompt for run/preview and a prompt plus optional
run handle for steer. status/follow infer the sole compatible run. Advanced
flags remain explicit overrides rather than requirements.
"""

from __future__ import annotations

from typing import Any, Optional

# ---------------------------------------------------------------------------
# Canonical outcome constants (shared across Python / CLI / MCP / MCP++)
# ---------------------------------------------------------------------------

OUTCOME_ACCEPTED = "accepted"
OUTCOME_COMPLETED = "completed"
OUTCOME_FAILED = "failed"
OUTCOME_AMBIGUOUS = "ambiguous"
OUTCOME_NOT_FOUND = "not_found"
OUTCOME_INVALID = "invalid"

CANONICAL_OUTCOMES = frozenset(
    {
        OUTCOME_ACCEPTED,
        OUTCOME_COMPLETED,
        OUTCOME_FAILED,
        OUTCOME_AMBIGUOUS,
        OUTCOME_NOT_FOUND,
        OUTCOME_INVALID,
    }
)

# Exit codes aligned with CLI and process semantics
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_AMBIGUOUS = 2
EXIT_NOT_FOUND = 3
EXIT_INVALID = 4

_OUTCOME_TO_EXIT = {
    OUTCOME_ACCEPTED: EXIT_SUCCESS,
    OUTCOME_COMPLETED: EXIT_SUCCESS,
    OUTCOME_FAILED: EXIT_FAILURE,
    OUTCOME_AMBIGUOUS: EXIT_AMBIGUOUS,
    OUTCOME_NOT_FOUND: EXIT_NOT_FOUND,
    OUTCOME_INVALID: EXIT_INVALID,
}


def outcome_to_exit_code(outcome: str) -> int:
    """Map a canonical outcome string to a process exit code."""
    return _OUTCOME_TO_EXIT.get(outcome, EXIT_FAILURE)


def _result(
    outcome: str,
    *,
    run_id: Optional[str] = None,
    message: Optional[str] = None,
    data: Optional[dict] = None,
    error: Optional[str] = None,
) -> dict:
    """Build a canonical result envelope shared by all transports."""
    if outcome not in CANONICAL_OUTCOMES:
        outcome = OUTCOME_INVALID
    body: dict[str, Any] = {
        "outcome": outcome,
        "exit_code": outcome_to_exit_code(outcome),
    }
    if run_id is not None:
        body["run_id"] = run_id
    if message is not None:
        body["message"] = message
    if data is not None:
        body["data"] = data
    if error is not None:
        body["error"] = error
    return body


def _require_prompt(prompt: Any) -> Optional[str]:
    """Validate that prompt is a non-empty string; return error message or None."""
    if prompt is None:
        return "prompt is required"
    if not isinstance(prompt, str):
        return "prompt must be a string"
    if not prompt.strip():
        return "prompt must be non-empty"
    return None


class AgentSupervisorFacade:
    """Prompt-only agent supervisor facade.

    Default surface:
      - run(prompt) / preview(prompt)
      - steer(prompt, run_id=None)
      - status() / follow()  — infer sole compatible run

    Advanced kwargs (backend, model, timeout, ...) are optional overrides only.
    """

    def __init__(self, backend: Any = None, **_kwargs: Any) -> None:
        self._backend = backend
        self._runs: dict[str, dict] = {}
        self._counter = 0

    # -- internal helpers ----------------------------------------------------

    def _next_run_id(self) -> str:
        self._counter += 1
        return f"run-{self._counter}"

    def _compatible_runs(self) -> list[str]:
        """Runs that are active/compatible for status/follow inference."""
        return [
            rid
            for rid, meta in self._runs.items()
            if meta.get("status") in ("running", "accepted", "steering")
        ]

    def _infer_sole_run(self) -> tuple[Optional[str], Optional[dict]]:
        """Infer the sole compatible run.

        Returns (run_id, None) on success, or (None, error_result).
        """
        compatible = self._compatible_runs()
        if len(compatible) == 0:
            # Fall back to most recent completed run if exactly one total run
            if len(self._runs) == 1:
                return next(iter(self._runs)), None
            if len(self._runs) == 0:
                return None, _result(
                    OUTCOME_NOT_FOUND,
                    error="no runs available to infer",
                    message="no runs available to infer",
                )
            return None, _result(
                OUTCOME_AMBIGUOUS,
                error="multiple runs; specify run_id",
                message="multiple runs; specify run_id",
            )
        if len(compatible) == 1:
            return compatible[0], None
        return None, _result(
            OUTCOME_AMBIGUOUS,
            error="multiple compatible runs; specify run_id",
            message="multiple compatible runs; specify run_id",
        )

    def _resolve_run(
        self, run_id: Optional[str] = None, *,
        allow_infer: bool = True,
    ) -> tuple[Optional[str], Optional[dict]]:
        if run_id is not None:
            if run_id not in self._runs:
                return None, _result(
                    OUTCOME_NOT_FOUND,
                    error=f"run not found: {run_id}",
                    message=f"run not found: {run_id}",
                )
            return run_id, None
        if not allow_infer:
            return None, _result(
                OUTCOME_INVALID,
                error="run_id is required",
                message="run_id is required",
            )
        return self._infer_sole_run()

    def _dispatch(self, action: str, prompt: str, **overrides: Any) -> dict:
        """Dispatch to backend if present, else use in-memory simulation."""
        if self._backend is not None:
            handler = getattr(self._backend, action, None)
            if callable(handler):
                return handler(prompt=prompt, **overrides)
        # In-memory default path
        if action == "run":
            rid = self._next_run_id()
            self._runs[rid] = {
                "prompt": prompt,
                "status": "completed",
                "action": "run",
                "overrides": {k: v for k, v in overrides.items() if v is not None},
            }
            return _result(
                OUTCOME_COMPLETED,
                run_id=rid,
                message="run completed",
                data={"prompt": prompt, "status": "completed"},
            )
        if action == "preview":
            rid = self._next_run_id()
            self._runs[rid] = {
                "prompt": prompt,
                "status": "completed",
                "action": "preview",
                "overrides": {k: v for k, v in overrides.items() if v is not None},
            }
            return _result(
                OUTCOME_COMPLETED,
                run_id=rid,
                message="preview completed",
                data={"prompt": prompt, "status": "completed", "preview": True},
            )
        if action == "steer":
            run_id = overrides.pop("run_id", None)
            resolved, err = self._resolve_run(run_id, allow_infer=True)
            if err is not None:
                return err
            assert resolved is not None
            meta = self._runs[resolved]
            meta["status"] = "steering"
            meta["steer_prompt"] = prompt
            meta["status"] = "completed"
            return _result(
                OUTCOME_COMPLETED,
                run_id=resolved,
                message="steer completed",
                data={"prompt": prompt, "status": "completed"},
            )
        return _result(OUTCOME_INVALID, error=f"unknown action: {action}")

    # -- public prompt-only API ----------------------------------------------

    def run(self, prompt: str, **overrides: Any) -> dict:
        """Start a run from a prompt. Advanced flags are optional overrides."""
        err = _require_prompt(prompt)
        if err:
            return _result(OUTCOME_INVALID, error=err, message=err)
        return self._dispatch("run", prompt, **overrides)

    def preview(self, prompt: str, **overrides: Any) -> dict:
        """Preview a plan from a prompt without committing side effects."""
        err = _require_prompt(prompt)
        if err:
            return _result(OUTCOME_INVALID, error=err, message=err)
        return self._dispatch("preview", prompt, **overrides)

    def steer(
        self,
        prompt: str,
        run_id: Optional[str] = None,
        **overrides: Any,
    ) -> dict:
        """Steer an existing run with a prompt; run_id optional if sole run."""
        err = _require_prompt(prompt)
        if err:
            return _result(OUTCOME_INVALID, error=err, message=err)
        return self._dispatch("steer", prompt, run_id=run_id, **overrides)

    def status(self, run_id: Optional[str] = None, **overrides: Any) -> dict:
        """Status of a run; infers the sole compatible run when run_id omitted."""
        resolved, err = self._resolve_run(run_id, allow_infer=True)
        if err is not None:
            return err
        assert resolved is not None
        if self._backend is not None:
            handler = getattr(self._backend, "status", None)
            if callable(handler):
                return handler(run_id=resolved, **overrides)
        meta = self._runs[resolved]
        return _result(
            OUTCOME_COMPLETED,
            run_id=resolved,
            message="status ok",
            data={
                "status": meta.get("status", "unknown"),
                "prompt": meta.get("prompt"),
                "action": meta.get("action"),
            },
        )

    def follow(self, run_id: Optional[str] = None, **overrides: Any) -> dict:
        """Follow/tail a run; infers the sole compatible run when run_id omitted."""
        resolved, err = self._resolve_run(run_id, allow_infer=True)
        if err is not None:
            return err
        assert resolved is not None
        if self._backend is not None:
            handler = getattr(self._backend, "follow", None)
            if callable(handler):
                return handler(run_id=resolved, **overrides)
        meta = self._runs[resolved]
        return _result(
            OUTCOME_COMPLETED,
            run_id=resolved,
            message="follow ok",
            data={
                "status": meta.get("status", "unknown"),
                "prompt": meta.get("prompt"),
                "events": meta.get("events", []),
            },
        )


# Module-level singleton helpers for transport-thin wrappers
_default_facade: Optional[AgentSupervisorFacade] = None


def get_facade(backend: Any = None, **kwargs: Any) -> AgentSupervisorFacade:
    """Return a process-wide default facade, creating one if needed."""
    global _default_facade
    if backend is not None or _default_facade is None:
        _default_facade = AgentSupervisorFacade(backend=backend, **kwargs)
    return _default_facade


def run(prompt: str, **overrides: Any) -> dict:
    return get_facade().run(prompt, **overrides)


def preview(prompt: str, **overrides: Any) -> dict:
    return get_facade().preview(prompt, **overrides)


def steer(prompt: str, run_id: Optional[str] = None, **overrides: Any) -> dict:
    return get_facade().steer(prompt, run_id=run_id, **overrides)


def status(run_id: Optional[str] = None, **overrides: Any) -> dict:
    return get_facade().status(run_id=run_id, **overrides)


def follow(run_id: Optional[str] = None, **overrides: Any) -> dict:
    return get_facade().follow(run_id=run_id, **overrides)


__all__ = [
    "AgentSupervisorFacade",
    "get_facade",
    "run",
    "preview",
    "steer",
    "status",
    "follow",
    "outcome_to_exit_code",
    "OUTCOME_ACCEPTED",
    "OUTCOME_COMPLETED",
    "OUTCOME_FAILED",
    "OUTCOME_AMBIGUOUS",
    "OUTCOME_NOT_FOUND",
    "OUTCOME_INVALID",
    "CANONICAL_OUTCOMES",
    "EXIT_SUCCESS",
    "EXIT_FAILURE",
    "EXIT_AMBIGUOUS",
    "EXIT_NOT_FOUND",
    "EXIT_INVALID",
]
