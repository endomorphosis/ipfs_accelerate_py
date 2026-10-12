"""Frozen repository-tool selection and retained native adoption observations.

Archive compatibility and tool adoption are separate from task completion,
proof validity and provider token savings. Failed provider attempts stay in the
observed population; none can be dropped to qualify an otherwise partial run.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import importlib.util
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import zlib

DEFAULT_REPOSITORY_SOURCE_POLICY = "legacy"
RAW_REPOSITORY_SOURCE_POLICY = "repository-source-raw@1"
COMPACT_REPOSITORY_SOURCE_POLICY = "repository-source-compact@1"
REPOSITORY_SOURCE_POLICIES = (DEFAULT_REPOSITORY_SOURCE_POLICY,
    RAW_REPOSITORY_SOURCE_POLICY, COMPACT_REPOSITORY_SOURCE_POLICY)
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/terminal_repository_source_policy.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
    "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_controls.py",
    "benchmarks/agent_supervisor/container_coding/terminal_deployment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py",
    "benchmarks/agent_supervisor/container_coding/container_worker_deployment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_context_audit.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_source_policy.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_source_mcp.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_source_probe.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_source_session.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_source_index.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_source_view.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/code_source_view.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/codex_usage_receipt.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/native_cli_subreaper.py",
    "ipfs_accelerate_py/cli_runtime/cli_metadata.py",
    "ipfs_accelerate_py/llm_router.py",
)
DATASETS_SOURCE_PATHS = tuple("ipfs_datasets_py/" + name for name in (
    "duckdb_control/codebase_catalog.py",
    "logic/software_contracts/ast_ir.py",
    "logic/software_contracts/cache.py",
    "logic/software_contracts/codebase_ir.py",
    "logic/software_contracts/codebase_resources.py",
    "logic/software_contracts/content.py",
    "logic/software_contracts/duckdb_ast_store.py",
    "logic/software_contracts/duckdb_ingest.py",
    "logic/software_contracts/python_frontend.py",
    "logic/software_contracts/repository.py",
    "logic/software_contracts/semantic_index/identity.py",
    "logic/software_contracts/semantic_index/models.py",
    "logic/software_contracts/semantic_index/python_analysis.py",
    "logic/software_contracts/semantic_index/pytest_analysis.py",
    "logic/software_contracts/semantic_index/scanner.py",
    "logic/software_contracts/semantic_index/snapshot.py",
    "logic/software_contracts/semantic_index/symbol_graph.py",
    "optimizers/logic_theorem_optimizer/resource_scheduler.py",
    "optimizers/logic_theorem_optimizer/proof_resource_safety.py",
    "optimizers/logic_theorem_optimizer/runtime_telemetry.py",
))
_COUNTERS = ("initializations", "tool_calls", "successful_reads", "compact_reads",
    "canonical_edits", "uncertain_edits", "refreshes", "errors", "returned_bytes")
_USAGE_FIELDS = {"status", *_COUNTERS, "store_cleanup_complete", "artifacts_retained_after_invocation"}
_REQUIRED_CHECKS = {"repository_source_helper_source", "model_prompt_sha256", "model_prompt_bytes",
                    "workspace_advisory_sha256", "workspace_advisory_bytes"}
MAX_ARCHIVE_BYTES = 16 * 1024**3
MAX_ARCHIVE_CONTENT_BYTES = 32 * 1024**3


def validate_repository_source_policy(value, *, arm=None, provider=None, source_view_policy="legacy"):
    if type(value) is not str or value not in REPOSITORY_SOURCE_POLICIES:
        raise ValueError("explicit supported repository source policy required")
    if value != DEFAULT_REPOSITORY_SOURCE_POLICY and (
            arm is not None and arm != "full" or provider is not None and provider != "codex_cli"
            or source_view_policy != "legacy"):
        raise ValueError("repository source policy requires the full Codex coding route without the old source helper")
    return value


def repository_source_selection(value):
    """Keep legacy control/configuration bytes unchanged."""
    return {} if validate_repository_source_policy(value) == DEFAULT_REPOSITORY_SOURCE_POLICY else {
        "repository_source_policy": value}


def _source_bytes(path):
    if path.resolve(strict=True) != path:
        raise ValueError("current repository tool source must not traverse symlinks")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 2 * 1024 * 1024:
            raise ValueError("bounded regular repository tool source required")
        raw = stream.read(2 * 1024 * 1024 + 1)
        after = os.fstat(stream.fileno())
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    if len(raw) != info.st_size or identity(info) != identity(after) or identity(after) != identity(path.stat()):
        raise ValueError("repository tool source changed while binding archive")
    return raw


def _current_sources():
    source = Path(__file__).resolve().parents[3]
    spec = importlib.util.find_spec("ipfs_datasets_py")
    roots = list(spec.submodule_search_locations or ()) if spec is not None else []
    if len(roots) != 1:
        raise ValueError("repository tools require one current datasets owner source tree")
    package = Path(roots[0]).absolute()
    if package.name != "ipfs_datasets_py" or package.resolve(strict=True) != package:
        raise ValueError("repository tools require canonical datasets owner source")
    return {**{"source/" + name: _source_bytes(source / name) for name in SOURCE_PATHS},
            **{"datasets/" + name: _source_bytes(package.parent / name) for name in DATASETS_SOURCE_PATHS}}


def _require_owner_api(sources):
    """Check required declared interfaces without importing/executing an owner.

    This is a compatibility prerequisite. Deployment still has to exercise the
    actual native owner; declarations alone cannot establish runtime behavior.
    """
    raw = sources["datasets/ipfs_datasets_py/logic/software_contracts/codebase_ir.py"]
    module = ast.parse(raw)
    classes = {node.name: node for node in module.body if isinstance(node, ast.ClassDef)}
    owner = classes.get("RepositoryCodebaseIndex")
    limits = classes.get("CodebaseScanLimits")
    if owner is None or limits is None:
        raise ValueError("datasets current repository owner API unavailable")
    methods = {node.name: node for node in owner.body if isinstance(node, ast.FunctionDef)}
    required = {
        "__init__": {"ingestor", "artifacts", "catalog"},
        "prepare_current": {"repository", "repository_id", "operation_id", "expected_head", "limits",
                            "scheduler", "admission_timeout_seconds", "timeout_seconds", "memory_mb"},
        "observe_current": {"repository", "expected_head", "scheduler", "admission_timeout_seconds",
                            "timeout_seconds", "memory_mb"},
        "current": {"repository_id"}, "load": {"manifest_cid"}, "lookup": {"manifest", "path"},
    }
    for name, parameters in required.items():
        method = methods.get(name)
        actual = set() if method is None else {arg.arg for arg in
            (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs)}
        if not parameters <= actual:
            raise ValueError("datasets current repository owner API unavailable")
    fields = {node.target.id for node in limits.body if isinstance(node, ast.AnnAssign)
              and isinstance(node.target, ast.Name)}
    if not {"max_entries", "max_file_bytes"} <= fields:
        raise ValueError("datasets bounded repository scan API unavailable")


def repository_source_inventory():
    """Read current compatibility inputs for fixtures and archive diagnostics."""
    sources = _current_sources()
    _require_owner_api(sources)
    return {"files": [{"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                      for name, raw in sources.items()]}


def _archive_name(value):
    if (type(value) is not str or not 0 < len(value.encode("utf-8")) <= 4096
            or any(ord(char) < 32 for char in value) or "\\" in value
            or value.startswith("/") or str(PurePosixPath(value)) != value
            or any(part in {".", ".."} for part in value.split("/"))):
        raise ValueError("canonical relative archive member required")
    return value


class _BoundedSourceTarInfo(tarfile.TarInfo):
    """Bound metadata before tarfile reads extended headers into memory."""

    def _proc_member(self, archive):
        archive._repository_headers = getattr(archive, "_repository_headers", 0) + 1
        if archive._repository_headers > 500_000:
            raise ValueError("archive header population exceeds bound")
        if self.type == tarfile.XHDTYPE:
            if self.size > 16_384 or getattr(archive, "_repository_extended_header", False):
                raise ValueError("archive extended header exceeds bound")
            archive._repository_extended_header = True
            try:
                return super()._proc_member(archive)
            finally:
                archive._repository_extended_header = False
        if self.type != tarfile.REGTYPE or not 0 <= self.size <= MAX_ARCHIVE_CONTENT_BYTES:
            raise ValueError("archive builder regular members required")
        return super()._proc_member(archive)

    def _reject_sparse(self, *_args, **_kwargs):
        # Small PAX headers can request a sparse map from the following body;
        # reject before stdlib reads or constructs that potentially large map.
        raise ValueError("archive sparse members not supported")

    _proc_gnusparse_00 = _reject_sparse
    _proc_gnusparse_01 = _reject_sparse
    _proc_gnusparse_10 = _reject_sparse


def _verify_archive_members(archive, manifest, expected):
    """Join current bodies to the actual archive without extracting anything.

    Streaming over all headers rejects late duplicate/alias replacements. Gzip
    must decompress skipped payloads to reach later headers, but unrelated model
    weights are neither retained nor hashed in memory. Manifest size limits
    constrain skipped members as well as selected source reads.
    """
    inventory = {}
    total = 0
    for row in manifest["files"]:
        if type(row) is not dict:
            raise ValueError("regular archive inventory required")
        name = _archive_name(row.get("path"))
        if (name in inventory or type(row.get("bytes")) is not int
                or not 0 <= row["bytes"] <= MAX_ARCHIVE_CONTENT_BYTES
                or not _valid_digest(row.get("sha256"))
                or "mode" in row and (type(row["mode"]) is not int or row["mode"] not in {0o644, 0o755})):
            raise ValueError("bounded unique archive inventory required")
        inventory[name] = row
        total += row["bytes"]
        if total > MAX_ARCHIVE_CONTENT_BYTES:
            raise ValueError("archive content exceeds bound")
    for name in inventory:
        if any(str(parent) in inventory for parent in PurePosixPath(name).parents if str(parent) != "."):
            raise ValueError("archive member has a file ancestor")
    path = Path(archive).absolute()
    if path.resolve(strict=True) != path:
        raise ValueError("canonical regular runtime archive required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_ARCHIVE_BYTES:
            raise ValueError("runtime archive exceeds bound")
        digest = hashlib.sha256()
        compressed_bytes = 0
        while chunk := stream.read(1024 * 1024):
            compressed_bytes += len(chunk)
            if compressed_bytes > MAX_ARCHIVE_BYTES:
                raise ValueError("runtime archive exceeds bound")
            digest.update(chunk)
        if not _valid_digest(manifest.get("archive_sha256")) or digest.hexdigest() != manifest["archive_sha256"]:
            raise ValueError("runtime archive digest differs")
        stream.seek(0)
        wanted = {row["path"]: row for row in expected}
        seen = set()
        # GzipFile validates CRC/trailer and bounds each requested decode. The
        # tarfile r|gz adapter itself does not validate the gzip trailer.
        with gzip.GzipFile(fileobj=stream, mode="rb") as decoded:
            with tarfile.open(fileobj=decoded, mode="r|", tarinfo=_BoundedSourceTarInfo) as bundle:
                for member in bundle:
                    name = _archive_name(member.name)
                    row = inventory.get(name)
                    if (len(seen) >= len(inventory) or name in seen or row is None or not member.isfile()
                            or member.size != row["bytes"] or member.mode not in {0o644, 0o755}
                            or set(member.pax_headers) - {"path"}
                            or "mode" in row and member.mode != row["mode"]):
                        raise ValueError("runtime archive member population differs")
                    seen.add(name)
                    if name in wanted:
                        # Required bodies already have the separate two-MiB
                        # source bound; no unbounded selected read is allowed.
                        source = bundle.extractfile(member)
                        if source is None:
                            raise ValueError("runtime source member unavailable")
                        with source:
                            raw = source.read(wanted[name]["bytes"] + 1)
                        if (len(raw) != wanted[name]["bytes"]
                                or hashlib.sha256(raw).hexdigest() != wanted[name]["sha256"]):
                            raise ValueError("runtime source member differs from current implementation")
                if seen != set(inventory):
                    raise ValueError("runtime archive inventory is incomplete")
                # Builder output ends in bounded tar-record padding. Draining
                # it verifies the gzip trailer and rejects hidden later members
                # or a concatenated replacement archive after the first end.
                padding = 0
                while tail := bundle.fileobj.read(1024):
                    padding += len(tail)
                    if padding > tarfile.RECORDSIZE or tail.strip(b"\0"):
                        raise ValueError("runtime archive trailing content differs")
        after = os.fstat(stream.fileno())
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(path.stat()):
        raise ValueError("runtime archive changed while checking members")


def require_repository_source_archive(manifest, value, *, archive=None):
    if validate_repository_source_policy(value) == DEFAULT_REPOSITORY_SOURCE_POLICY:
        return
    rows = manifest.get("files") if type(manifest) is dict else None
    if type(rows) is not list or len(rows) > 250_000:
        raise ValueError("repository source policy requires current runtime source inventory")
    try:
        expected = repository_source_inventory()["files"]
        by_path = {}
        for row in rows:
            if type(row) is dict and type(row.get("path")) is str:
                by_path.setdefault(row["path"], []).append(row)
        for item in expected:
            matches = by_path.get(item["path"], [])
            if (len(matches) != 1 or matches[0].get("sha256") != item["sha256"]
                    or type(matches[0].get("bytes")) is not int or matches[0]["bytes"] != item["bytes"]):
                raise ValueError("source inventory differs")
        if archive is not None:
            _verify_archive_members(archive, manifest, expected)
    except (ValueError, OSError, SyntaxError, KeyError, ImportError, tarfile.TarError,
            EOFError, UnicodeError, zlib.error) as error:
        raise ValueError("repository source policy requires a newly qualified runtime archive with current datasets owner APIs") from error


def _valid_id(value):
    return type(value) is str and 0 < len(value) <= 4096


def _valid_digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _usage_observation(value):
    """Reject impossible aggregate populations; unknown counters stay unknown."""
    shape = (type(value) is dict and set(value) in (_USAGE_FIELDS, _USAGE_FIELDS | {"store_cleanup_error"})
             and type(value.get("status")) is str
             and value["status"] in {"observed", "not_observed", "unavailable"}
             and type(value.get("store_cleanup_complete")) is bool
             and (value.get("artifacts_retained_after_invocation") is None
                  or type(value.get("artifacts_retained_after_invocation")) is bool)
             and ("store_cleanup_error" not in value or value["store_cleanup_error"] == "cleanup_unavailable"))
    valid = shape
    if shape and value["status"] == "unavailable":
        valid = all(value[key] is None for key in _COUNTERS)
    elif shape:
        valid = all(type(value[key]) is int and 0 <= value[key] <= 2**31 for key in _COUNTERS)
        if valid:
            init, calls, reads, compact, edits, uncertain, refresh, errors, size = (value[key] for key in _COUNTERS)
            valid = (init <= 1 and calls <= 512 and compact <= reads
                and reads + edits + uncertain <= calls and errors <= calls + (1 - init)
                and uncertain <= errors and reads + refresh <= calls - max(0, errors - (1 - init))
                and edits <= refresh + errors
                and ((calls == 0 and size == 0) or (calls > 0 and calls * 2 <= size <= calls * 2 * 1024 * 1024))
                and (init == 1 or calls == reads == compact == edits == uncertain == refresh == size == 0)
                and (value["status"] != "not_observed" or not any(value[key] for key in _COUNTERS))
                and (value["status"] != "observed" or init + errors + calls > 0))
    cleanup = bool(shape and value["store_cleanup_complete"] is True
                   and value["artifacts_retained_after_invocation"] is False
                   and "store_cleanup_error" not in value)
    if shape and value["store_cleanup_complete"] is True and not cleanup:
        valid = False
    if shape and value["store_cleanup_complete"] is False and value["artifacts_retained_after_invocation"] is False:
        valid = False
    return {"server_audit_valid": bool(valid), "server_audit_status": value.get("status") if shape else "unavailable",
            "store_cleanup_complete": cleanup,
            **{key: value[key] if valid else None for key in _COUNTERS}}


def repository_source_dispatch_observation(report, value):
    """Join all exact-input receipts to native configuration and server counters.

    Counters are retained runtime observations, not an independent audit replay.
    A failing provider may still have used tools; its receipt remains required.
    """
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_policy import validate_binding

    validate_repository_source_policy(value)
    report = report if type(report) is dict else {}
    declared = report.get("repository_source_policy") == value
    problems = [] if declared else ["requested_policy_report_mismatch"]
    if report.get("unreceipted_provider_attempt"):
        problems.append("unreceipted_provider_attempt")
    population = report.get("provider_invocations")
    if type(population) is not list or len(population) > 4096:
        problems.append("provider_invocation_population_unavailable_or_unbounded")
        population = []
    if any(type(row) is not dict or type(row.get("phase")) is not str
           or row["phase"] not in {"planning", "coding"} for row in population):
        problems.append("malformed_provider_invocation")
    coding = [row for row in population if type(row) is dict and row.get("phase") == "coding"]
    planning_absent = all("repository_source_policy" not in row for row in population
                          if type(row) is dict and row.get("phase") == "planning")
    if not planning_absent:
        problems.append("planning_repository_source_policy_present")
    audit = report.get("context_input_audit")
    audit = audit if type(audit) is dict else {}
    audit_rows, matches = audit.get("invocations"), audit.get("matches")
    audit_rows = audit_rows if type(audit_rows) is list and len(audit_rows) <= 16 else []
    matches = matches if type(matches) is list and len(matches) <= 512 else []
    ids = [row.get("invocation_id") for row in coding]
    distinct = all(_valid_id(identity) for identity in ids) and len(set(ids)) == len(ids)
    audit_ids = [row.get("invocation_id") for row in audit_rows if type(row) is dict]
    audit_population = (distinct and len(audit_rows) == len(coding) == len(audit_ids)
        and all(_valid_id(identity) for identity in audit_ids) and len(set(audit_ids)) == len(audit_ids)
        and set(audit_ids) == set(ids) and audit.get("status") == "verified_all_observed_model_inputs"
        and audit.get("all_observed_coding_inputs_verified") is True
        and type(audit.get("coding_receipts_seen")) is int and audit["coding_receipts_seen"] == len(coding)
        and type(audit.get("usable_coding_receipts")) is int and audit["usable_coding_receipts"] == len(coding)
        and audit.get("read_errors") == [] and bool(matches)
        and all(type(item) is dict and _valid_id(item.get("invocation_id"))
                and item["invocation_id"] in ids for item in matches))
    if not coding:
        problems.append("no_observed_coding_receipts")
    elif not distinct:
        problems.append("coding_invocation_identities_missing_or_duplicate")
    if not audit_population:
        problems.append("retained_exact_input_population_unverified")
    bindings = [row.get("repository_source_policy") for row in coding]
    config_ids = [item.get("config_sha256") for item in bindings if type(item) is dict]
    if (len(config_ids) != len(coding) or not all(_valid_digest(item) for item in config_ids)
            or len(set(config_ids)) != len(config_ids)):
        problems.append("repository_configuration_identities_missing_or_duplicate")
    try:
        helper = Path(__file__).resolve().parents[3] / "ipfs_accelerate_py/agent_supervisor/runtime/repository_source_mcp.py"
        helper_digest = hashlib.sha256(_source_bytes(helper)).hexdigest()
    except (ValueError, OSError):
        helper_digest = None
    observations = []
    for row in coding:
        identity, binding = row.get("invocation_id"), row.get("repository_source_policy")
        binding_valid = False
        try:
            workspace = row.get("workspace")
            if not _valid_id(workspace) or not Path(workspace).is_absolute():
                raise ValueError("invalid coding workspace")
            validate_binding(binding, workspace=Path(workspace))
            binding_valid = (binding["requested_policy"] == value and row.get("provider") == "codex_cli"
                and row.get("purpose") == "coding" and "source_view_policy" not in row
                and type(row.get("status")) is str and row["status"] in {"provider_returned", "failed"}
                and helper_digest is not None and binding["helper_sha256"] == helper_digest
                and _valid_digest(row.get("model_prompt_sha256"))
                and type(row.get("model_prompt_bytes")) is int and 0 < row["model_prompt_bytes"] <= 256_000)
        except (ValueError, TypeError, KeyError, OSError):
            pass
        per_audit = [item for item in audit_rows if type(item) is dict and item.get("invocation_id") == identity]
        per_matches = [item for item in matches if type(item) is dict and item.get("invocation_id") == identity]
        def verified_match(item):
            checks = item.get("model_input_checks")
            return (item.get("model_input_verified") is True
                and item.get("model_prompt_sha256") == row.get("model_prompt_sha256")
                and type(item.get("model_prompt_bytes")) is int and item["model_prompt_bytes"] == row.get("model_prompt_bytes")
                and type(checks) is dict and _REQUIRED_CHECKS <= set(checks)
                and all(check is True for check in checks.values()))
        exact = bool(binding_valid and audit_population and len(per_audit) == 1
            and per_audit[0].get("model_input_verified") is True
            and per_audit[0].get("status") == "verified_exact_model_input"
            and per_matches and all(verified_match(item) for item in per_matches))
        native = row.get("usage")
        configured = bool(binding_valid and type(native) is dict
            and native.get("codex_repository_source_configured") is True
            and native.get("codex_repository_source_config_sha256") == binding["config_sha256"])
        usage = _usage_observation(binding.get("usage") if type(binding) is dict else None)
        initialized = usage["server_audit_valid"] and usage["initializations"] == 1
        read = bool(initialized and usage["successful_reads"] and usage["returned_bytes"])
        compact = bool(read and usage["compact_reads"])
        effective = bool(exact and configured and read and usage["store_cleanup_complete"]
            and usage["uncertain_edits"] == 0
            and (value != COMPACT_REPOSITORY_SOURCE_POLICY or compact))
        observations.append({"invocation_id": identity if _valid_id(identity) else None,
            "provider_attempt_status": row.get("status") if row.get("status") in ("provider_returned", "failed") else None,
            "policy_binding_valid": binding_valid, "retained_exact_input_audit_verified": exact,
            "native_tools_configured": configured, **usage,
            "server_initialized": bool(initialized), "actual_reads_observed": read,
            "compact_reads_observed": compact,
            "confirmed_canonical_edit_observed": bool(exact and configured and initialized and usage["canonical_edits"]),
            "effective_repository_tools_observed": effective})
    effective = bool(observations) and not problems and all(
        row["effective_repository_tools_observed"] for row in observations)
    return {"schema": "terminal-repository-source-dispatch-observation@1", "requested_policy": value,
        "requested_policy_matches_report": declared, "planning_policy_absent": planning_absent,
        "coding_receipts_seen": len(coding), "invocations": observations,
        "effective_repository_tools_observed": effective,
        "status": "verified_repository_tool_use" if effective else "unverified", "problems": problems,
        "source_audit_rerun": False, "provider_usage_modified": False,
        "token_savings_established": False, "completion_authority": False, "proof_authority": False}
