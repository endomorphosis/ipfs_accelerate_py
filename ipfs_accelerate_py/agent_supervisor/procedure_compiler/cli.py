"""Thin ``ipfs-accelerate agent procedures`` adapter for PCPC-028.

This module deliberately owns no procedure policy.  It serializes a bounded
request into :class:`ProcedureControlRequest` and calls the injected service
directly; in particular it does not spawn a shell or an MCP subprocess.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any, TextIO

from ..control.control_plane import (
    ProcedureControlRequest,
    ProcedureControlServiceAdapter,
    ProcedureOperation,
)


class ProcedureCLI:
    """Typed command adapter with the same operation vocabulary as MCP."""

    def __init__(self, service: ProcedureControlServiceAdapter | None = None) -> None:
        self.service = service or ProcedureControlServiceAdapter()

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="ipfs-accelerate agent procedures")
        parser.add_argument("operation", choices=[item.value.removeprefix("procedures.") for item in ProcedureOperation])
        parser.add_argument("--request-json", help="Complete ProcedureControlRequest JSON object.")
        parser.add_argument("--target-json", default="{}", help="Exact target JSON object.")
        parser.add_argument("--parameters-json", default="{}", help="Operation parameter JSON object.")
        parser.add_argument("--authorization-json", help="Authorization decision JSON object.")
        parser.add_argument("--lease-fence-json", default="{}", help="Lease/fence JSON object for mutations.")
        parser.add_argument("--idempotency-key", default="")
        parser.add_argument("--request-id", default="")
        parser.add_argument("--dry-run", action="store_true")
        return parser

    @staticmethod
    def _object(value: str, label: str) -> Mapping[str, Any]:
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} must be valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise ValueError(f"{label} must be a JSON object")
        return decoded

    def request_from_namespace(self, namespace: argparse.Namespace) -> ProcedureControlRequest:
        operation = ProcedureOperation(f"procedures.{namespace.operation}")
        if namespace.request_json:
            payload = dict(self._object(namespace.request_json, "request-json"))
            supplied = ProcedureControlRequest(**payload)
            if supplied.operation is not operation:
                raise ValueError("positional operation and request-json operation differ")
            return supplied
        return ProcedureControlRequest(
            operation=operation,
            target=self._object(namespace.target_json, "target-json"),
            parameters=self._object(namespace.parameters_json, "parameters-json"),
            authorization=(self._object(namespace.authorization_json, "authorization-json") if namespace.authorization_json else None),
            idempotency_key=namespace.idempotency_key,
            lease_fence=self._object(namespace.lease_fence_json, "lease-fence-json"),
            dry_run=namespace.dry_run,
            request_id=namespace.request_id,
        )

    @staticmethod
    def _record(result: Any) -> Mapping[str, Any]:
        audit = result.audit
        return {
            "operation": result.operation.value,
            "status": result.status,
            "data": dict(result.data),
            "error": result.error,
            "replayed": result.replayed,
            "audit": None if audit is None else {
                "schema": audit.schema, "operation": audit.operation.value,
                "request_id": audit.request_id, "target": dict(audit.target),
                "authorization_id": audit.authorization_id, "dry_run": audit.dry_run,
                "replayed": audit.replayed,
            },
        }

    def run(self, argv: Sequence[str] | None = None, *, stdout: TextIO | None = None, stderr: TextIO | None = None) -> int:
        stdout, stderr = stdout or sys.stdout, stderr or sys.stderr
        try:
            parsed = self.parser().parse_args(argv)
            result = self.service.execute(self.request_from_namespace(parsed))
            print(json.dumps(self._record(result), sort_keys=True, separators=(",", ":")), file=stdout)
            return 0 if result.successful else 1
        except (ValueError, TypeError) as exc:
            print(f"invalid procedure request: {exc}", file=stderr)
            return 2


def main(argv: Sequence[str] | None = None) -> int:
    return ProcedureCLI().run(argv)


__all__ = ["ProcedureCLI", "main"]
