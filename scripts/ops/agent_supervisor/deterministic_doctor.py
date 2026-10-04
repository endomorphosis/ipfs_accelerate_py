#!/usr/bin/env python3
"""Thin ops facade for deterministic-doctor operations (LPR-039).

Contains only argument parsing, config bootstrap, and delegation into
:class:`~ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_service.DeterministicDoctorService`.

Cold import and ``--help`` start no process, open no database, access no
network/storage, and import no optional datasets / prover / embedding /
model provider.  The wrapper owns no analysis, proof, rendering,
transaction, or mutation logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_ABSTAIN = 3
EXIT_APPROVAL = 4

# Closed subcommand names only; service owns execution.
_SUBCOMMANDS = (
    "inspect",
    "explain",
    "plan",
    "repair",
    "replay",
    "rollback",
    "status",
    "verify",
    "discovery",
)

# Forbidden argv tokens that smuggle bodies/secrets.
_FORBIDDEN_ARGV_MARKERS = (
    "--body",
    "--source",
    "--source-text",
    "--secret",
    "--password",
    "--token",
    "--api-key",
    "--apikey",
    "--authorization",
    "--private-key",
    "--credential",
    "--cookie",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deterministic_doctor",
        description=(
            "Thin facade for the deterministic-doctor control service. "
            "Parses closed operations and delegates; never invokes a model "
            "provider or mutates sources itself."
        ),
    )
    parser.add_argument(
        "--checkout-root",
        default=None,
        help="Exact repository checkout root for runtime-backed operations",
    )
    parser.add_argument(
        "--policy-json",
        default=None,
        help="Path to a deterministic-doctor policy JSON object (body-free)",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=("report_only", "plan", "sandbox_auto", "narrow_auto"),
        help="Doctor mode (default: policy default / report_only)",
    )
    parser.add_argument(
        "--incident-id",
        default=None,
        help="Incident CID / opaque incident identifier",
    )
    parser.add_argument(
        "--snapshot-json",
        default=None,
        help="Path to a DoctorEvidenceSnapshot@1 JSON record",
    )
    parser.add_argument(
        "--plan-json",
        default=None,
        help="Path to a DeterministicDoctorPlan@1 JSON record",
    )
    parser.add_argument(
        "--receipt-json",
        default=None,
        help="Path to a prior DeterministicDoctorRunReceipt@1 JSON record",
    )
    parser.add_argument(
        "--roots-json",
        default=None,
        help="Path to a DoctorAuthorityRoots@1 JSON record",
    )
    parser.add_argument(
        "--lease-id",
        default=None,
        help="Writer lease identifier (required for repair)",
    )
    parser.add_argument(
        "--checkpoint-ref",
        default=None,
        help="Checkpoint reference (required for repair)",
    )
    parser.add_argument(
        "--rollback-ref",
        default=None,
        help="Rollback evidence reference (required for repair)",
    )
    parser.add_argument(
        "--target-tree-cid",
        default=None,
        help="Exact target tree CID for clean-target checks",
    )
    parser.add_argument(
        "--exact-clean-target",
        action="store_true",
        help="Operator asserts the bound target is exact and clean",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Emit machine-readable JSON (default; always on)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("inspect", "Read-only evidence / roots inspection"),
        ("explain", "Read-only disposition explanation"),
        ("plan", "Read-only plan report (no source writes)"),
        ("repair", "Write-capable repair (policy + lease + plan gated)"),
        ("replay", "Identity-equivalent replay by incident CID"),
        ("rollback", "Compensating rollback against checkpoint evidence"),
        ("status", "Bounded service / incident status projection"),
        ("verify", "Verify a run receipt under deterministic policy"),
        ("discovery", "Static discovery (no providers, no process)"),
    ):
        sub.add_parser(name, help=help_text)
    return parser


def _reject_forbidden_argv(argv: Sequence[str]) -> None:
    lowered = [item.lower() for item in argv]
    for marker in _FORBIDDEN_ARGV_MARKERS:
        if marker in lowered or any(item.startswith(marker + "=") for item in lowered):
            raise SystemExit(
                f"error: forbidden argument {marker!r}; secrets/bodies must not enter argv"
            )


def _load_json_object(path: str | None, label: str) -> Mapping[str, Any] | None:
    if not path:
        return None
    raw = Path(path).expanduser().read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SystemExit(f"error: {label} must be a JSON object")
    # Fail closed on smuggled body/secret keys without echoing values.
    forbidden = {
        "body",
        "source",
        "source_text",
        "secret",
        "password",
        "token",
        "api_key",
        "private_key",
        "credential",
    }
    for key in payload:
        normalized = str(key).lower().replace("-", "_")
        if normalized in forbidden or any(
            normalized.endswith("_" + marker) for marker in forbidden
        ):
            raise SystemExit(
                f"error: {label} may not contain source bodies or secrets"
            )
    return payload


def _bootstrap_service(policy_payload: Mapping[str, Any] | None):
    """Lazy-import the service only after argument validation."""

    from ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_service import (  # noqa: WPS433
        DeterministicDoctorService,
        create_deterministic_doctor_service,
    )

    if policy_payload is None:
        return create_deterministic_doctor_service()
    return DeterministicDoctorService(policy=policy_payload)


def _bootstrap_runtime(
    checkout_root: str,
    policy_payload: Mapping[str, Any] | None,
):
    """Lazy-import the production runtime after parsing and discovery gates."""

    from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_doctor_runtime import (  # noqa: WPS433
        create_deterministic_doctor_runtime,
    )

    return create_deterministic_doctor_runtime(
        checkout_root,
        policy=policy_payload,
    )


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    try:
        _reject_forbidden_argv(raw_argv)
    except SystemExit as exc:
        message = str(exc)
        if message:
            print(message, file=sys.stderr)
        return EXIT_USAGE

    parser = build_parser()
    try:
        args = parser.parse_args(raw_argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return EXIT_SUCCESS
        return int(code) if isinstance(code, int) else EXIT_USAGE

    if args.command not in _SUBCOMMANDS:
        print(f"unknown command: {args.command}", file=sys.stderr)
        return EXIT_USAGE

    try:
        policy_payload = _load_json_object(args.policy_json, "policy")
        if args.command == "discovery":
            service = _bootstrap_service(policy_payload)
            payload = service.discovery(policy=policy_payload)
            sys.stdout.write(
                json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
                + "\n"
            )
            return EXIT_SUCCESS

        request: dict[str, Any] = {"operation": args.command}
        if args.mode:
            request["mode"] = args.mode
        if args.incident_id:
            request["incident_id"] = args.incident_id
        if args.lease_id:
            request["lease_id"] = args.lease_id
        if args.checkpoint_ref:
            request["checkpoint_ref"] = args.checkpoint_ref
        if args.rollback_ref:
            request["rollback_ref"] = args.rollback_ref
        if args.target_tree_cid:
            request["target_tree_cid"] = args.target_tree_cid
        if args.exact_clean_target:
            request["exact_clean_target"] = True

        snapshot = _load_json_object(args.snapshot_json, "snapshot")
        if snapshot is not None:
            request["snapshot"] = dict(snapshot)
        plan = _load_json_object(args.plan_json, "plan")
        if plan is not None:
            request["plan"] = dict(plan)
        receipt = _load_json_object(args.receipt_json, "receipt")
        if receipt is not None:
            request["prior_receipt"] = dict(receipt)
        roots = _load_json_object(args.roots_json, "roots")
        if roots is not None:
            request["roots"] = dict(roots)

        # A checkout-backed operation uses the lazy production composition.
        # Caller-supplied snapshots retain the provider-free control-only path.
        use_runtime = bool(args.checkout_root) and snapshot is None and args.command in {
            "inspect",
            "explain",
            "plan",
            "repair",
        }
        if use_runtime:
            runtime = _bootstrap_runtime(args.checkout_root, policy_payload)
            report = runtime.execute(request)
            result = report.result
            payload = {
                **result.to_dict(),
                "runtime": {
                    "interface": runtime.INTERFACE,
                    "evidence": (
                        report.evidence.to_dict()
                        if report.evidence is not None
                        else None
                    ),
                    "stage_receipts": {
                        key: dict(value)
                        for key, value in report.stage_receipts.items()
                    },
                    "capability_graph": runtime.capability_graph(),
                },
            }
        else:
            service = _bootstrap_service(policy_payload)
            result = service.execute(request)
            payload = result.to_dict()
        # Never log or print request bodies beyond the machine-readable result.
        sys.stdout.write(
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
            + "\n"
        )
        return int(result.exit_code)
    except Exception as exc:  # noqa: BLE001 - facade maps all failures
        # Do not echo potential secret material; only the exception type/message.
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        name = type(exc).__name__
        if "Safety" in name or "LLM" in str(exc) or "llm" in str(exc).lower():
            return EXIT_FAILURE
        if "body" in str(exc).lower() or "secret" in str(exc).lower():
            return EXIT_USAGE
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
