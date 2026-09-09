"""One explicit ``python -m`` entry point for the governed proof-context CLI."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from . import app

ALL_COMMANDS: Final[tuple[str, ...]] = (
    "init", "scan", "status", "plan", "run", "verify", "resume",
    "expand-context", "explain-impact", "assurance", "seal", "report",
)
STATE_COMMANDS: Final[frozenset[str]] = frozenset(("init", "scan", "status", "plan"))
EXECUTION_COMMANDS: Final[frozenset[str]] = frozenset(("run", "verify", "resume"))
EVIDENCE_COMMANDS: Final[frozenset[str]] = frozenset(("expand-context", "explain-impact", "assurance", "seal", "report"))
REQUEST_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/cli-run-request@1"


class _UsageError(ValueError):
    pass


def _read_json(path: str, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _UsageError(f"{label} must name a readable JSON object") from exc
    if not isinstance(value, Mapping):
        raise _UsageError(f"{label} must contain a JSON object")
    return value


def _bytes_option(value: Mapping[str, Any], *, request_dir: Path) -> bytes:
    patch_file = value.get("patch_file")
    patch_base64 = value.get("patch_base64")
    if (patch_file is None) == (patch_base64 is None):
        raise _UsageError("external-patch requires exactly one of patch_file or patch_base64")
    if isinstance(patch_file, str):
        candidate = Path(patch_file)
        if not candidate.is_absolute():
            candidate = request_dir / candidate
        try:
            return candidate.read_bytes()
        except OSError as exc:
            raise _UsageError("external-patch patch_file is not readable") from exc
    if not isinstance(patch_base64, str):
        raise _UsageError("external-patch patch_base64 must be a string")
    try:
        return base64.b64decode(patch_base64.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise _UsageError("external-patch patch_base64 is invalid") from exc


def _adapter_configuration(document: Mapping[str, Any], *, request_dir: Path):
    from ipfs_accelerate_py.proof_context.adapters.registry import AdapterConfiguration
    from ipfs_accelerate_py.proof_context.adapters.replay import ReplayFixture

    adapter = document.get("adapter")
    if not isinstance(adapter, Mapping) or not isinstance(adapter.get("name"), str):
        raise _UsageError("request adapter.name is required")
    options = adapter.get("options", {})
    if not isinstance(options, Mapping):
        raise _UsageError("request adapter.options must be an object")
    materialized = dict(options)
    if adapter["name"] == "external-patch":
        materialized["patch"] = _bytes_option(options, request_dir=request_dir)
        materialized.pop("patch_file", None)
        materialized.pop("patch_base64", None)
    elif adapter["name"] == "replay":
        fixtures = options.get("fixtures")
        if not isinstance(fixtures, list):
            raise _UsageError("replay fixtures must be a JSON array")
        try:
            materialized["fixtures"] = tuple(ReplayFixture.from_mapping(item) for item in fixtures)
        except Exception as exc:  # adapter wire validation supplies the closed semantics
            raise _UsageError("replay fixtures are not admitted fixture records") from exc
    try:
        return AdapterConfiguration(adapter["name"], materialized)
    except Exception as exc:
        raise _UsageError("request adapter configuration is invalid") from exc


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=app.PROG, allow_abbrev=False)
    parser.add_argument("command", choices=ALL_COMMANDS, help="governed command")
    parser.add_argument("--repository", required=True, help="explicit repository path; cwd is never inferred")
    parser.add_argument("--policy", default=app.DEFAULT_POLICY, choices=("production", "supervised", "evaluation", "simulation"))
    parser.add_argument("--task", required=True, help="explicit task identity")
    parser.add_argument("--correlation", required=True, help="explicit trace/correlation identity")
    parser.add_argument("--output-mode", default="json", choices=("json", "human"))
    parser.add_argument("--run-id")
    parser.add_argument("--repository-id")
    parser.add_argument("--state-dir")
    parser.add_argument("--request", help="run request JSON (required by run)")
    parser.add_argument("--run-record", help="exact run record JSON (verify only)")
    parser.add_argument("--patch-id", help="exact patch identity (verify only)")
    parser.add_argument("--checkpoint", help="resume checkpoint JSON (resume only)")
    parser.add_argument("--parent", help="immutable parent record JSON (evidence commands only)")
    parser.add_argument("--human-report", action="store_true", help="render the PCCE-043 patch report after the normal result")
    return parser


def _context(args: argparse.Namespace):
    return app.CliContext(args.command, Path(args.repository), args.policy, args.task, args.correlation, args.output_mode, args.run_id, args.repository_id, Path(args.state_dir) if args.state_dir else None)


def _execute(args: argparse.Namespace):
    if args.command in STATE_COMMANDS:
        if any((args.request, args.run_record, args.patch_id, args.checkpoint, args.parent)):
            raise _UsageError("state commands do not accept execution or evidence input files")
        return app.dispatch(_context(args))
    if args.command in EXECUTION_COMMANDS:
        from ipfs_accelerate_py.proof_context.cli.execution_commands import ExecutionContext, RunRequest, cmd_resume, cmd_run, cmd_verify
        context = ExecutionContext(args.command, Path(args.repository), args.policy, args.task, args.correlation, args.output_mode, args.run_id or "", args.repository_id, Path(args.state_dir) if args.state_dir else None, args.patch_id)
        if args.command == "run":
            if not args.request:
                raise _UsageError("run requires --request")
            request_path = Path(args.request)
            document = _read_json(args.request, label="request")
            if document.get("schema") != REQUEST_SCHEMA:
                raise _UsageError("request schema is unsupported")
            try:
                from ipfs_accelerate_py.proof_context.adapters.models import ContextPack, ModelRouteDecision, TaskSpecification
                return cmd_run(RunRequest(context, _adapter_configuration(document, request_dir=request_path.parent), TaskSpecification.from_mapping(document["task"]), ContextPack.from_mapping(document["context_pack"]), ModelRouteDecision.from_mapping(document["route"])))
            except KeyError as exc:
                raise _UsageError("request requires task, context_pack, route, and adapter") from exc
        if args.command == "verify":
            if bool(args.run_record) == bool(args.patch_id):
                raise _UsageError("verify requires exactly one of --run-record or --patch-id")
            return cmd_verify(context, run=_read_json(args.run_record, label="run-record") if args.run_record else None, patch_id=args.patch_id)
        return cmd_resume(context, checkpoint=_read_json(args.checkpoint, label="checkpoint") if args.checkpoint else None)
    from ipfs_accelerate_py.proof_context.cli.evidence_commands import EvidenceContext, cmd_assurance, cmd_explain_impact, cmd_expand_context, cmd_report, cmd_seal
    if not args.parent:
        raise _UsageError(f"{args.command} requires --parent")
    if not args.run_id or not args.repository_id or not args.patch_id:
        raise _UsageError(f"{args.command} requires --run-id, --repository-id, and --patch-id")
    context = EvidenceContext(args.command, Path(args.repository), args.policy, args.task, args.correlation, args.output_mode, args.run_id, args.repository_id, args.patch_id, Path(args.state_dir) if args.state_dir else None)
    handler = {"expand-context": cmd_expand_context, "explain-impact": cmd_explain_impact, "assurance": cmd_assurance, "seal": cmd_seal, "report": cmd_report}[args.command]
    return handler(context, parent=_read_json(args.parent, label="parent"))


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return only the stable result exit code."""
    values = list(sys.argv[1:] if argv is None else argv)
    parser = _base_parser()
    try:
        args = parser.parse_args(values)
        result = _execute(args)
    except _UsageError as exc:
        output_mode = app.peek_output_mode(values)
        result = app.usage_result(command=app.peek_command(values), message=str(exc), correlation_id=app.peek_correlation(values), output_mode=output_mode)
        sys.stderr.write(f"{app.PROG}: {exc}\n")
    except Exception:  # JSON wire-record admission is a closed CLI boundary.
        output_mode = app.peek_output_mode(values)
        message = "command input is invalid"
        result = app.usage_result(command=app.peek_command(values), message=message, correlation_id=app.peek_correlation(values), output_mode=output_mode)
        sys.stderr.write(f"{app.PROG}: {message}\n")
    except SystemExit as exc:
        return int(exc.code)
    sys.stdout.write(app.render(result))
    if getattr(args, "human_report", False):
        from ipfs_accelerate_py.proof_context.cli.output import command_result
        from ipfs_accelerate_py.proof_context.cli.report import render_human_patch_report
        report_input = command_result(
            status=result.status,
            correlation_id=result.correlation_id,
            identities=result.identities,
            artifact_cids=[result.artifact_cid] if result.artifact_cid else [],
            command=result.command,
            provenance=result.provenance,
            details=result.payload,
            error=result.error,
        )
        sys.stdout.write(render_human_patch_report(report_input))
    return app.exit_code_for(result.status, provenance=result.provenance)


if __name__ == "__main__":
    raise SystemExit(main())
