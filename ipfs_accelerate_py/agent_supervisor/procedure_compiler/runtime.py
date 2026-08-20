"""Principal P0 facade for the proof-carrying procedure compiler.

This facade intentionally exposes parsing, structural validation, and
deterministic invocation only.  Synthesis, certificate issuance, registry
promotion, policy changes, and completion authority remain outside P0 and
outside this class's authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from .contracts import ProcedureCertificate, ProcedureInvocation, ProcedureSpec
from .interpreter import (
    ExecutionMode,
    ProcedureExecution,
    ProcedureInterpreter,
    RuntimeIdentity,
)
from .procedure_ir import (
    ProcedureIRParser,
    parse_procedure_spec,
    validate_procedure_spec,
)

PROGRAM_ID: Final[str] = "agent-supervisor-proof-carrying-procedure-compiler-v1"
ROOT_OBJECTIVE_ID: Final[str] = "PCPC-G000"
TASK_PREFIX: Final[str] = "PCPC-"
PRINCIPAL_SUBSYSTEM: Final[str] = "ProofCarryingProcedureCompiler"


class ProcedureCompilerCapabilityError(RuntimeError):
    """The caller requested authority this bounded facade does not possess."""

    def __init__(self, operation: str) -> None:
        super().__init__(
            "{} does not implement or authorize {}".format(PRINCIPAL_SUBSYSTEM, operation)
        )
        self.operation = operation
        self.reason_code = "capability_not_available_in_p0"


@dataclass(frozen=True)
class ProcedureCompilerCapabilities:
    program_id: str = PROGRAM_ID
    principal_subsystem: str = PRINCIPAL_SUBSYSTEM
    parse_and_validate: bool = True
    deterministic_invoke: bool = True
    synthesize: bool = False
    issue_certificate: bool = False
    promote: bool = False
    modify_policy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "principal_subsystem": self.principal_subsystem,
            "parse_and_validate": self.parse_and_validate,
            "deterministic_invoke": self.deterministic_invoke,
            "synthesize": self.synthesize,
            "issue_certificate": self.issue_certificate,
            "promote": self.promote,
            "modify_policy": self.modify_policy,
        }


class ProofCarryingProcedureCompiler:
    """Parse ProcedureIR and invoke the deterministic fail-closed runtime."""

    def __init__(
        self,
        interpreter: ProcedureInterpreter,
        *,
        parser: ProcedureIRParser | None = None,
    ) -> None:
        if not isinstance(interpreter, ProcedureInterpreter):
            raise TypeError("interpreter must be a ProcedureInterpreter")
        self._interpreter = interpreter
        self._parser = parser or ProcedureIRParser()

    @property
    def capabilities(self) -> ProcedureCompilerCapabilities:
        return ProcedureCompilerCapabilities()

    @property
    def operation_catalog_revision(self) -> str:
        return self._interpreter.operation_catalog_revision

    def parse(self, value: ProcedureSpec | Mapping[str, Any]) -> ProcedureSpec:
        if isinstance(value, ProcedureSpec):
            validate_procedure_spec(value)
            return value
        return self._parser.parse(value)

    parse_procedure = parse

    def validate(self, value: ProcedureSpec | Mapping[str, Any]) -> ProcedureSpec:
        procedure = self.parse(value)
        validate_procedure_spec(procedure)
        return procedure

    validate_procedure = validate

    def invoke(
        self,
        procedure: ProcedureSpec | Mapping[str, Any],
        invocation: ProcedureInvocation | Mapping[str, Any],
        certificate: ProcedureCertificate | Mapping[str, Any],
        runtime: RuntimeIdentity,
        *,
        mode: ExecutionMode = ExecutionMode.LIVE,
    ) -> ProcedureExecution:
        parsed = self.validate(procedure)
        if isinstance(invocation, Mapping):
            invocation = ProcedureInvocation.from_dict(invocation)
        if isinstance(certificate, Mapping):
            certificate = ProcedureCertificate.from_dict(certificate)
        return self._interpreter.execute(parsed, invocation, certificate, runtime, mode=mode)

    execute = invoke
    run = invoke

    def synthesize(self, *_args: Any, **_kwargs: Any) -> None:
        raise ProcedureCompilerCapabilityError("procedure synthesis")

    def issue_certificate(self, *_args: Any, **_kwargs: Any) -> None:
        raise ProcedureCompilerCapabilityError("certificate issuance")

    def promote(self, *_args: Any, **_kwargs: Any) -> None:
        raise ProcedureCompilerCapabilityError("procedure promotion")

    def modify_policy(self, *_args: Any, **_kwargs: Any) -> None:
        raise ProcedureCompilerCapabilityError("authority or safety policy modification")


def compiler_capabilities() -> Mapping[str, Any]:
    """Return an immutable read-only description without constructing runtime."""

    return MappingProxyType(ProcedureCompilerCapabilities().to_dict())


__all__ = [
    "PRINCIPAL_SUBSYSTEM",
    "PROGRAM_ID",
    "ROOT_OBJECTIVE_ID",
    "TASK_PREFIX",
    "ProofCarryingProcedureCompiler",
    "ProcedureCompilerCapabilities",
    "ProcedureCompilerCapabilityError",
    "compiler_capabilities",
    "parse_procedure_spec",
]
