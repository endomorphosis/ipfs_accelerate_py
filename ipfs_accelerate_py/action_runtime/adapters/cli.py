"""Sandboxed CLI action adapter.

Safety rules:

* ``shell=False`` only via the shared ProcessRunner
* executable identity is absolute and operator-reviewed at registration time
* domain packs / proposals never supply executable paths, cwd, or env maps
* environment is a fixed allowlist (default empty inherit set + PATH only)
* argv is built from fixed prefixes plus validated slot values
* output is size-bounded and only digests/redacted public fields leave the adapter
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from ipfs_accelerate_py.cli_runtime.process_runner import (
    ProcessBounds,
    ProcessRunner,
    ProcessSpec,
)

from ..catalog import resolve_reviewed_executable
from ..contracts import (
    ActionDecision,
    ActionProposal,
    ActionReceipt,
    ActionStatus,
    content_digest,
)

_SAFE_ARG_RE = re.compile(r"^[A-Za-z0-9_./:@+-]{1,256}$")
_INJECTION_MARKERS = (
    ";",
    "&&",
    "||",
    "|",
    "`",
    "$(",
    "${",
    "\n",
    "\r",
    "\x00",
    ">",
    "<",
    "*",
    "?",
    "~",
    " ",
    "\t",
)


@dataclass(frozen=True)
class CLISandboxPolicy:
    """Resource and environment bounds for CLI invocation."""

    timeout_seconds: float = 5.0
    max_stdout_bytes: int = 65_536
    max_stderr_bytes: int = 16_384
    max_argv_items: int = 32
    max_argv_item_chars: int = 256
    # Empty means: do not inherit ambient environment; only inject listed keys.
    allowed_env: Mapping[str, str] = field(default_factory=dict)
    allowed_cwd_roots: tuple[str, ...] = ()
    # When True, only the fixed allowed_env is used (no ambient inheritance).
    isolate_environment: bool = True

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.timeout_seconds > 60:
            raise ValueError("timeout_seconds must be in (0, 60]")
        for key, value in self.allowed_env.items():
            if not key or not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("allowed_env must be string mappings")
            if any(marker in key.lower() for marker in ("secret", "token", "password", "key")):
                raise ValueError(f"refusing secret-shaped env key {key!r}")


@dataclass(frozen=True)
class CLIActionRegistration:
    """Reviewed CLI binding for a catalog descriptor."""

    descriptor_id: str
    executable: str | Path
    fixed_argv_prefix: tuple[str, ...] = ()
    # Optional named slots filled from proposal.arguments (values validated).
    argument_slots: tuple[str, ...] = ()
    sandbox: CLISandboxPolicy = field(default_factory=CLISandboxPolicy)
    interface_name: str = "cli"

    def __post_init__(self) -> None:
        if not self.descriptor_id:
            raise ValueError("descriptor_id is required")
        pinned = resolve_reviewed_executable(self.executable)
        object.__setattr__(self, "executable", pinned)
        for item in self.fixed_argv_prefix:
            _validate_argv_token(item, role="fixed_argv_prefix")
        for slot in self.argument_slots:
            if not slot or not slot.replace("_", "").isalnum():
                raise ValueError(f"invalid argument slot name {slot!r}")

    @property
    def interface_identity(self) -> str:
        return f"cli:{self.executable}:{','.join(self.fixed_argv_prefix)}"


def _validate_argv_token(token: str, *, role: str) -> str:
    if not isinstance(token, str) or not token:
        raise ValueError(f"{role} token must be a non-empty string")
    if len(token) > 256:
        raise ValueError(f"{role} token exceeds max length")
    for marker in _INJECTION_MARKERS:
        if marker in token:
            raise ValueError(f"{role} rejects injection marker {marker!r} in {token!r}")
    if not _SAFE_ARG_RE.match(token):
        raise ValueError(f"{role} token has disallowed characters: {token!r}")
    # Refuse path traversal in slots/prefix extras.
    if ".." in token.split("/"):
        raise ValueError(f"{role} rejects path traversal: {token!r}")
    return token


def build_argv(
    registration: CLIActionRegistration,
    arguments: Mapping[str, str],
) -> list[str]:
    """Construct a validated argv vector from a registration and proposal args."""

    argv: list[str] = [str(registration.executable), *registration.fixed_argv_prefix]
    for slot in registration.argument_slots:
        if slot not in arguments:
            raise ValueError(f"missing required argument slot {slot!r}")
        argv.append(_validate_argv_token(arguments[slot], role=f"slot:{slot}"))
    # Reject unexpected arguments so packs cannot smuggle extras.
    unexpected = set(arguments) - set(registration.argument_slots)
    if unexpected:
        raise ValueError(f"unexpected arguments: {sorted(unexpected)}")
    if len(argv) > registration.sandbox.max_argv_items:
        raise ValueError("argv exceeds sandbox max_argv_items")
    for item in argv:
        if len(item) > registration.sandbox.max_argv_item_chars:
            raise ValueError("argv item exceeds sandbox max_argv_item_chars")
    return argv


class CLIActionAdapter:
    """Execute only after a permitting decision binds the exact proposal."""

    def __init__(
        self,
        registrations: Sequence[CLIActionRegistration],
        *,
        runner: ProcessRunner | None = None,
    ) -> None:
        self._by_descriptor: dict[str, CLIActionRegistration] = {}
        for registration in registrations:
            if registration.descriptor_id in self._by_descriptor:
                raise ValueError(
                    f"duplicate CLI registration for {registration.descriptor_id!r}"
                )
            self._by_descriptor[registration.descriptor_id] = registration
        self._runner = runner or ProcessRunner()

    def get_registration(self, descriptor_id: str) -> CLIActionRegistration | None:
        return self._by_descriptor.get(descriptor_id)

    def invoke(
        self,
        *,
        proposal: ActionProposal,
        decision: ActionDecision,
    ) -> ActionReceipt:
        receipt_id = f"rcpt-{uuid.uuid4().hex[:16]}"
        started = time.time()

        if not decision.permits_execution:
            return ActionReceipt(
                receipt_id=receipt_id,
                status=ActionStatus.DENIED,
                proposal_id=proposal.proposal_id,
                decision_id=decision.decision_id,
                descriptor_id=proposal.descriptor_id,
                adapter="cli",
                interface_identity="cli:none",
                started_epoch_s=started,
                completed_epoch_s=time.time(),
                error=f"decision_does_not_permit_execution:{decision.kind.value}",
            )

        if decision.proposal_id != proposal.proposal_id:
            return self._failed(
                receipt_id,
                proposal,
                decision,
                "proposal_decision_mismatch",
                started,
            )
        if decision.descriptor_id != proposal.descriptor_id:
            return self._failed(
                receipt_id,
                proposal,
                decision,
                "descriptor_decision_mismatch",
                started,
            )
        if decision.arguments_digest != proposal.arguments_digest:
            return self._failed(
                receipt_id,
                proposal,
                decision,
                "arguments_digest_mismatch",
                started,
            )
        if decision.expires_at_epoch_s is not None and time.time() > decision.expires_at_epoch_s:
            return self._failed(
                receipt_id,
                proposal,
                decision,
                "decision_expired",
                started,
            )

        registration = self._by_descriptor.get(proposal.descriptor_id)
        if registration is None:
            return self._failed(
                receipt_id,
                proposal,
                decision,
                "no_cli_registration",
                started,
            )

        try:
            argv = build_argv(registration, proposal.arguments)
        except ValueError as exc:
            return self._failed(
                receipt_id,
                proposal,
                decision,
                f"argv_build_failed:{exc}",
                started,
                interface_identity=registration.interface_identity,
            )

        sandbox = registration.sandbox
        # Isolated env: ProcessSpec with env_overlay=False and only allowlisted keys.
        env: dict[str, str | None]
        if sandbox.isolate_environment:
            env = {key: value for key, value in sandbox.allowed_env.items()}
            # Provide a minimal PATH if caller did not pin one, so absolute
            # executables still run; ambient secrets are not inherited.
            env.setdefault("PATH", "/usr/bin:/bin")
            env.setdefault("LANG", "C.UTF-8")
            env_overlay = False
        else:
            # Even in overlay mode, only allowlisted keys may be *added*;
            # ambient inheritance is still reduced by not passing secrets.
            env = {key: value for key, value in sandbox.allowed_env.items()}
            env_overlay = True

        bounds = ProcessBounds(
            max_argv_items=sandbox.max_argv_items,
            max_argv_item_chars=sandbox.max_argv_item_chars,
            max_stdout_bytes=sandbox.max_stdout_bytes,
            max_stderr_bytes=sandbox.max_stderr_bytes,
            max_elapsed_seconds=sandbox.timeout_seconds,
        )
        spec = ProcessSpec(
            argv=argv,
            cwd=None,  # never caller-controlled
            env=env,
            env_overlay=env_overlay,
            timeout_seconds=sandbox.timeout_seconds,
            allowed_cwd_roots=sandbox.allowed_cwd_roots,
            side_effecting=False,
            metadata={
                "descriptor_id": proposal.descriptor_id,
                "proposal_id": proposal.proposal_id,
                "decision_id": decision.decision_id,
            },
        )

        # Bounds and environment isolation are fixed on the runner instance;
        # ProcessSpec cannot carry bounds and never inherits ambient secrets
        # when isolate_environment is enabled (env_overlay=False, empty base).
        runner = ProcessRunner(
            bounds=bounds,
            base_env={} if sandbox.isolate_environment else None,
            popen_factory=getattr(self._runner, "_popen", None),
            clock=getattr(self._runner, "_clock", None),
        )

        try:
            result = runner.run(spec)
        except Exception as exc:
            return self._failed(
                receipt_id,
                proposal,
                decision,
                f"runner_error:{type(exc).__name__}:{exc}",
                started,
                interface_identity=registration.interface_identity,
            )

        completed = time.time()
        status = ActionStatus.SUCCEEDED if result.ok else ActionStatus.FAILED
        if result.timed_out:
            status = ActionStatus.TIMED_OUT
        if result.cancelled:
            status = ActionStatus.CANCELLED

        public: dict[str, str] = {
            "ok": "true" if result.ok else "false",
            "exit_code": "" if result.exit_code is None else str(result.exit_code),
        }
        # Never return raw stdout/stderr to voice layer; digests only.
        return ActionReceipt(
            receipt_id=receipt_id,
            status=status,
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            descriptor_id=proposal.descriptor_id,
            adapter="cli",
            interface_identity=registration.interface_identity,
            started_epoch_s=started,
            completed_epoch_s=completed,
            exit_code=result.exit_code,
            stdout_digest=content_digest(result.stdout),
            stderr_digest=content_digest(result.stderr),
            public_result=public,
            error=result.error_message,
            metadata={
                "argv_preview": " ".join(result.argv_preview[:8]),
                "elapsed_seconds": f"{result.elapsed_seconds:.4f}",
            },
        )

    def _failed(
        self,
        receipt_id: str,
        proposal: ActionProposal,
        decision: ActionDecision,
        error: str,
        started: float,
        *,
        interface_identity: str = "cli:none",
    ) -> ActionReceipt:
        return ActionReceipt(
            receipt_id=receipt_id,
            status=ActionStatus.FAILED,
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            descriptor_id=proposal.descriptor_id,
            adapter="cli",
            interface_identity=interface_identity,
            started_epoch_s=started,
            completed_epoch_s=time.time(),
            error=error,
        )
