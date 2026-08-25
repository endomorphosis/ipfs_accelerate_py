#!/usr/bin/env python3
"""Seal the terminal source, semantic and authority state (EAAEF-175).

A worker principal cannot self-seal.  Independent sealer identity is required.
Valid requests write ``docs/.../receipts/terminal_seal.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final


SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-agent-terminal-seal@1"
)
RECEIPT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "terminal_seal.json"
)
REQUIRED_ROOT_FIELDS: Final[tuple[str, ...]] = ("source_root", "semantic_root")
WORKER_PREFIXES: Final[tuple[str, ...]] = ("worker:", "sha256:")


class SealError(ValueError):
    """Terminal seal request is not admissible."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _text(value: object, name: str, *, required: bool = True) -> str:
    text = "" if value is None else str(value).strip()
    if required and not text:
        raise SealError(f"{name} is required", reason_code="missing_field")
    return text


def _empty_sequence(value: object, name: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SealError(f"{name} must be a sequence", reason_code="malformed")
    return tuple(value)


def _present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _content_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_seal_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a terminal seal request.  Does not write a receipt."""

    if not isinstance(request, Mapping):
        raise SealError("seal request must be an object", reason_code="malformed")

    source_root = _text(request.get("source_root"), "source_root")
    semantic_root = _text(request.get("semantic_root"), "semantic_root")
    if not source_root or not semantic_root:
        raise SealError("source and semantic roots are required", reason_code="missing_roots")

    tests = request.get("tests", request.get("tests_present"))
    proofs = request.get("proofs", request.get("proofs_present"))
    if not _present(tests):
        raise SealError("tests are required", reason_code="missing_tests")
    if not _present(proofs):
        raise SealError("proofs are required", reason_code="missing_proofs")

    claims = _empty_sequence(request.get("claims"), "claims")
    merge_queue = _empty_sequence(request.get("merge_queue"), "merge_queue")
    if claims:
        raise SealError("claims must be empty", reason_code="claims_not_empty")
    if merge_queue:
        raise SealError("merge queue must be empty", reason_code="merge_queue_not_empty")

    ducklake_cursor = _text(request.get("ducklake_cursor"), "ducklake_cursor")
    sealer_id = _text(
        request.get("sealer_id") or request.get("principal_id"),
        "sealer_id",
    )
    lowered = sealer_id.lower()
    if lowered.startswith(WORKER_PREFIXES):
        raise SealError(
            "a worker principal cannot self-seal",
            reason_code="worker_self_seal",
        )
    worker_principal = _text(
        request.get("worker_principal_id"),
        "worker_principal_id",
        required=False,
    )
    if worker_principal and worker_principal == sealer_id:
        raise SealError(
            "a worker principal cannot self-seal",
            reason_code="worker_self_seal",
        )

    body = {
        "schema": SCHEMA,
        "source_root": source_root,
        "semantic_root": semantic_root,
        "tests_present": True,
        "proofs_present": True,
        "claims": [],
        "merge_queue": [],
        "ducklake_cursor": ducklake_cursor,
        "sealer_id": sealer_id,
        "worker_self_seal": False,
        "live_runtime_invoked": False,
        "live_eight_container_qualification": False,
        "evidence_mode": "contract_fail_closed",
    }
    body["content_id"] = _content_id(body)
    body["terminal_report_id"] = body["content_id"]
    return body


def seal_release(
    request: Mapping[str, Any],
    *,
    receipt_path: Path | None = None,
    write_receipt: bool = True,
) -> dict[str, Any]:
    """Validate and optionally persist the terminal seal receipt."""

    sealed = validate_seal_request(request)
    if write_receipt:
        path = receipt_path if receipt_path is not None else RECEIPT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sealed


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seal the terminal EAAEF release state")
    parser.add_argument("--request", type=Path, help="JSON seal request")
    parser.add_argument("--output", type=Path, default=RECEIPT_PATH)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.request is None:
        print("seal request JSON is required", file=sys.stderr)
        return 2
    try:
        payload = json.loads(args.request.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise SealError("seal request JSON must be an object", reason_code="malformed")
        sealed = seal_release(payload, receipt_path=args.output)
    except (OSError, json.JSONDecodeError, SealError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(sealed, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
