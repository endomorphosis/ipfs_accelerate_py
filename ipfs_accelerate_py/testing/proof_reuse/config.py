"""Pure configuration model for the proof-reuse pytest plugin.

This module must remain safe to import in a cold Python process.  In
particular, it must not import pytest or any optional proof, cache, IPFS, or ZK
provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional, Tuple

PROOF_REUSE_MODE_ENV = "IPFS_TEST_PROOF_REUSE_MODE"
PROOF_REUSE_REQUIRED_AUDIT_ENV = "IPFS_TEST_PROOF_REUSE_REQUIRED_AUDIT"


class ProofReuseConfigurationError(ValueError):
    """An invalid proof-reuse setting detected before test execution."""


class ProofReuseMode(str, Enum):
    """Operational modes; required-audit intentionally is not a mode."""

    OFF = "off"
    SHADOW = "shadow"
    READ = "read"
    WRITE = "write"
    READWRITE = "readwrite"

    @classmethod
    def parse(cls, value: Any) -> ProofReuseMode:
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.OFF
        if not isinstance(value, str):
            raise ProofReuseConfigurationError(
                "proof reuse mode must be a string"
            )
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError:
            raise ProofReuseConfigurationError(
                "invalid proof reuse mode; expected one of: "
                f"{', '.join(mode.value for mode in cls)}"
            ) from None


PROOF_REUSE_MODES: Tuple[str, ...] = tuple(mode.value for mode in ProofReuseMode)

_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_VALUES = frozenset(("0", "false", "no", "off", ""))


def _parse_bool(value: Any, *, setting_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    raise ProofReuseConfigurationError(
        f"{setting_name} must be a boolean "
        "(true/false, yes/no, on/off, or 1/0)"
    )


@dataclass(frozen=True)
class ProofReuseConfig:
    """Resolved, immutable plugin configuration.

    Invalid environment or ini values fail open to ``off`` by default while
    retaining a bounded diagnostic in ``configuration_error``.  Callers that
    own a strict configuration boundary may request an explicit exception via
    :meth:`resolve`.
    """

    mode: ProofReuseMode = ProofReuseMode.OFF
    required_audit: bool = False
    source: str = "default"
    configuration_error: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ProofReuseMode.parse(self.mode))
        if not isinstance(self.required_audit, bool):
            object.__setattr__(
                self,
                "required_audit",
                _parse_bool(
                    self.required_audit,
                    setting_name="proof reuse required-audit",
                ),
            )
    @property
    def enabled(self) -> bool:
        return self.mode is not ProofReuseMode.OFF

    @property
    def reads_candidates(self) -> bool:
        return self.mode in (
            ProofReuseMode.SHADOW,
            ProofReuseMode.READ,
            ProofReuseMode.READWRITE,
        )

    @property
    def may_skip(self) -> bool:
        return self.mode in (ProofReuseMode.READ, ProofReuseMode.READWRITE)

    @property
    def writes_receipts(self) -> bool:
        return self.mode in (
            ProofReuseMode.WRITE,
            ProofReuseMode.READWRITE,
        )

    @classmethod
    def resolve(
        cls,
        *,
        command_line_mode: Any = None,
        ini_mode: Any = None,
        environ: Optional[Mapping[str, str]] = None,
        command_line_required_audit: Any = False,
        ini_required_audit: Any = None,
        strict: bool = False,
    ) -> ProofReuseConfig:
        """Resolve CLI, ini, and environment values without provider access.

        Precedence is command line, ini, environment, then the safe ``off``
        default.  Invalid values either raise here (``strict=True``) or produce
        an explicitly disabled configuration.
        """

        environment = environ or {}
        raw_mode: Any
        source: str
        if command_line_mode not in (None, ""):
            raw_mode, source = command_line_mode, "command-line"
        elif ini_mode not in (None, ""):
            raw_mode, source = ini_mode, "ini"
        elif environment.get(PROOF_REUSE_MODE_ENV) not in (None, ""):
            raw_mode, source = environment.get(PROOF_REUSE_MODE_ENV), "environment"
        else:
            raw_mode, source = ProofReuseMode.OFF, "default"

        if command_line_required_audit:
            raw_required_audit, audit_source = (
                command_line_required_audit,
                "command-line",
            )
        elif ini_required_audit not in (None, "", False):
            raw_required_audit, audit_source = ini_required_audit, "ini"
        elif environment.get(PROOF_REUSE_REQUIRED_AUDIT_ENV) not in (None, ""):
            raw_required_audit, audit_source = (
                environment.get(PROOF_REUSE_REQUIRED_AUDIT_ENV),
                "environment",
            )
        else:
            raw_required_audit, audit_source = False, "default"

        try:
            mode = ProofReuseMode.parse(raw_mode)
            required_audit = _parse_bool(
                raw_required_audit,
                setting_name="proof reuse required-audit",
            )
        except ProofReuseConfigurationError as exc:
            if strict:
                raise
            return cls(
                mode=ProofReuseMode.OFF,
                required_audit=False,
                source="invalid",
                configuration_error=str(exc)[:512],
            )

        resolved_source = source
        if audit_source != "default" and audit_source != source:
            resolved_source = f"{source};required-audit={audit_source}"
        return cls(
            mode=mode,
            required_audit=required_audit,
            source=resolved_source,
        )

    def disabled(self, error: Optional[str] = None) -> ProofReuseConfig:
        """Return a fail-open, off-mode copy suitable for provider failures."""

        return replace(
            self,
            mode=ProofReuseMode.OFF,
            configuration_error=(error[:512] if error else self.configuration_error),
        )


__all__ = [
    "PROOF_REUSE_MODE_ENV",
    "PROOF_REUSE_MODES",
    "PROOF_REUSE_REQUIRED_AUDIT_ENV",
    "ProofReuseConfigurationError",
    "ProofReuseConfig",
    "ProofReuseMode",
]
