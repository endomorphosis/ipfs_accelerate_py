#!/usr/bin/env python3
"""Qualify and materialize the PCPC board into DuckDB plus DuckLake history.

DuckDB/DatabaseTaskSource is the only task-state authority. Quack owns the
live database after this one-process bootstrap exits. DuckLake receives a
bounded, non-authoritative history projection and cannot affect readiness.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (  # noqa: E402
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (  # noqa: E402
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (  # noqa: E402
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    parse_task_text,
)

CONFIG_RELATIVE = "config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json"
BASELINE_RELATIVE = "docs/architecture/procedure_compiler_inventory/baseline.json"
PREREQUISITES_RELATIVE = "docs/architecture/procedure_compiler_inventory/prerequisites.json"
PROGRAM = "agent-supervisor-proof-carrying-procedure-compiler-v1"
BRANCH = "codex/proof-carrying-procedure-compiler-v1"
BASE_COMMIT = "bbf7f68799072c2b81f7d96eac91f2df3c4b3952"
P0_TASKS = tuple(f"PCPC-{index:03d}" for index in range(9))
EXPECTED_READY = ("PCPC-009", "PCPC-011", "PCPC-013")
MAX_CAPTURE_BYTES = 64_000
MAX_PREREQUISITE_PRODUCERS = 64
PREREQUISITE_PRODUCER_TIMEOUT_SECONDS = 900
PYTEST_OUTCOME_FIELDS = ("collected", "passed", "failed", "errors", "returncode")
PYTEST_TERMINAL_OUTCOME_RE = re.compile(
    r"(?P<count>[0-9]+)\s+"
    r"(?P<kind>passed|failed|errors?|skipped|deselected|xfailed|xpassed)\b"
)
PREREQUISITE_STATUSES = frozenset(
    {"available", "available_with_caveats", "incompatible", "stale", "missing"}
)
REQUIRED_PREREQUISITE_AUTHORITIES = frozenset(
    {
        "SemanticCompressionHarness",
        "SemanticCompressionGovernor",
        "AdversarialAssuranceEngine",
        "IncrementalVerificationPlanner",
        "IncrementalProofSealer",
        "AdaptivePlanner",
        "SupervisorControlService",
        "ContextCompiler",
        "ValueOfInformation evidence selection",
        "Delta retry contexts",
        "Provider capacity and route policy",
        "Worktree lease fencing and merge controls",
        "AutonomousMetaController",
        "autonomy package",
        "cognitive scheduler",
        "experience ledger",
        "policy-distillation subsystem",
    }
)
REQUIRED_PREREQUISITE_PRODUCER_IDS = frozenset(
    {
        "TP-SEMANTIC-HARNESS",
        "TP-SEMANTIC-GOVERNOR",
        "TP-ADVERSARIAL-ASSURANCE",
        "TP-INCREMENTAL-VERIFICATION",
        "TP-INCREMENTAL-PROOF-SEALER",
        "TP-ADAPTIVE-PLANNER-IMPORT",
        "TP-CONTROL-SERVICE",
        "TP-CONTEXT-COMPILER",
        "TP-VALUE-OF-INFORMATION",
        "TP-DELTA-RETRY",
        "TP-MODEL-ROUTE",
        "TP-PROVIDER-CAPACITY",
        "TP-DUAL-PROVIDER-CAPACITY",
        "TP-DEFAULT-PROVIDER-ROUTE",
        "TP-WORKTREE-LIFECYCLE",
        "TP-LEASE-COORDINATION",
        "TP-MERGE-TRAIN",
        "TP-FENCE-REGISTRY-QUEUE",
    }
)
PREREQUISITE_BINDING_FIELDS = (
    "source_bindings",
    "symbol_bindings",
    "interface_bindings",
    "schema_bindings",
    "package_bindings",
    "submodule_bindings",
    "test_producer_bindings",
)
QUALIFICATION_COMMANDS = (
    (
        sys.executable,
        "scripts/validate_agent_supervisor_procedure_compiler_board.py",
        "--check-all",
    ),
    (sys.executable, "-m", "pytest", "-q", "test/api/procedure_compiler"),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "test/api/test_agent_supervisor_quack_state_server.py",
        "test/api/test_agent_supervisor_quack_transport_defaults.py",
        "test/api/test_agent_supervisor_quack_owner_mutation.py",
        "test/api/test_agent_supervisor_intent_repository.py",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "test/api/test_agent_supervisor_direct_codex_isolation.py",
    ),
    (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "test/api/test_agent_supervisor_configured_board_scheduler.py",
    ),
    (
        sys.executable,
        "-c",
        "from ipfs_accelerate_py.agent_supervisor.procedure_compiler "
        "import ProofCarryingProcedureCompiler; "
        "print(ProofCarryingProcedureCompiler.__name__)",
    ),
)


class MaterializationError(RuntimeError):
    """Fail-closed bootstrap or qualification error."""


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode:
        raise MaterializationError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MaterializationError(f"{path} must contain one JSON object")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_repository_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise MaterializationError(f"{field} must be a non-empty repository path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise MaterializationError(f"{field} escapes or is not canonical: {value!r}")
    return value


def _object_id(revision: str, path: str) -> str:
    return _git("rev-parse", f"{revision}:{path}")


def _verify_bound_blob(*, path: str, blob_id: str, baseline_commit: str) -> None:
    if (
        not isinstance(blob_id, str)
        or len(blob_id) != 40
        or any(character not in "0123456789abcdef" for character in blob_id)
    ):
        raise MaterializationError(f"invalid blob binding for {path!r}")
    baseline_blob = _object_id(baseline_commit, path)
    head_blob = _object_id("HEAD", path)
    if baseline_blob != blob_id:
        raise MaterializationError(
            f"baseline source drift for {path}: expected {blob_id}, observed {baseline_blob}"
        )
    if head_blob != blob_id:
        raise MaterializationError(
            f"current-tree source drift for {path}: expected {blob_id}, observed {head_blob}"
        )
    working_path = REPO_ROOT / path
    if not working_path.is_file():
        raise MaterializationError(f"bound source is absent from checkout: {path}")
    working_blob = _git("hash-object", path)
    if working_blob != blob_id:
        raise MaterializationError(
            f"working-tree source drift for {path}: expected {blob_id}, observed {working_blob}"
        )


def _python_declarations(
    path: str, cache: dict[str, tuple[set[tuple[str, str]], dict[str, object]]]
) -> tuple[set[tuple[str, str]], dict[str, object]]:
    cached = cache.get(path)
    if cached is not None:
        return cached
    try:
        tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"), filename=path)
    except (OSError, SyntaxError) as exc:
        raise MaterializationError(f"cannot inspect bound Python source {path}: {exc}") from exc
    declarations: set[tuple[str, str]] = set()
    constants: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            declarations.add(("class", node.name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            declarations.add(("function", node.name))
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        if isinstance(target, ast.Name) and value is not None:
            try:
                constants[target.id] = ast.literal_eval(value)
            except (ValueError, TypeError):
                pass
    result = (declarations, constants)
    cache[path] = result
    return result


def _git_path_exists(revision: str, path: str) -> bool:
    result = subprocess.run(
        ("git", "ls-tree", revision, "--", path),
        cwd=REPO_ROOT,
        text=True,
        check=False,
        capture_output=True,
        timeout=60,
    )
    if result.returncode:
        raise MaterializationError(result.stderr.strip() or f"cannot probe path {revision}:{path}")
    return bool(result.stdout.strip())


def _git_class_exists(revision: str, *, scope: str, symbol: str) -> bool:
    if not symbol.isidentifier():
        raise MaterializationError(f"invalid Python class probe symbol {symbol!r}")
    result = subprocess.run(
        (
            "git",
            "grep",
            "-n",
            "-E",
            rf"^class[[:space:]]+{symbol}([[:space:](:]|$)",
            revision,
            "--",
            scope,
        ),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode not in (0, 1):
        raise MaterializationError(result.stderr.strip() or f"cannot probe Python class {symbol!r}")
    return result.returncode == 0


def _require_mapping_list(value: object, *, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise MaterializationError(f"{field} must be an array")
    if any(not isinstance(item, Mapping) for item in value):
        raise MaterializationError(f"{field} entries must be objects")
    return value


def _validate_prerequisite_payloads(
    baseline: Mapping[str, Any], inventory: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if (
        baseline.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/procedure-compiler-baseline@3"
    ):
        raise MaterializationError("unsupported prerequisite baseline schema")
    if (
        inventory.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/procedure-compiler-prerequisite-inventory@2"
    ):
        raise MaterializationError("unsupported prerequisite inventory schema")
    repository = baseline.get("repository")
    if not isinstance(repository, Mapping):
        raise MaterializationError("baseline.repository must be an object")
    baseline_commit = repository.get("commit")
    baseline_tree = repository.get("tree")
    if baseline_commit != BASE_COMMIT or inventory.get("baseline_commit") != BASE_COMMIT:
        raise MaterializationError("prerequisite inventory baseline commit is not sealed")
    if (
        not isinstance(baseline_tree, str)
        or _git("rev-parse", f"{BASE_COMMIT}^{{tree}}") != baseline_tree
        or inventory.get("baseline_tree") != baseline_tree
    ):
        raise MaterializationError("prerequisite inventory baseline tree is not sealed")

    package_rows = _require_mapping_list(
        baseline.get("package_bindings"), field="baseline.package_bindings"
    )
    if not package_rows:
        raise MaterializationError("baseline.package_bindings must not be empty")
    packages: dict[str, Mapping[str, Any]] = {}
    for row in package_rows:
        binding_id = row.get("binding_id")
        if not isinstance(binding_id, str) or not binding_id or binding_id in packages:
            raise MaterializationError("package binding IDs must be non-empty and unique")
        path = _relative_repository_path(row.get("manifest_path"), field="manifest_path")
        _verify_bound_blob(
            path=path,
            blob_id=row.get("manifest_blob_id"),
            baseline_commit=BASE_COMMIT,
        )
        if row.get("name") != "ipfs_accelerate_py" or row.get("version") != "0.0.45":
            raise MaterializationError(f"unsupported package binding {binding_id!r}")
        packages[binding_id] = row

    sibling_rows = _require_mapping_list(
        baseline.get("sibling_release_bindings"),
        field="baseline.sibling_release_bindings",
    )
    siblings: dict[str, Mapping[str, Any]] = {}
    for row in sibling_rows:
        binding_id = row.get("binding_id")
        commit = row.get("gitlink_commit")
        if (
            not isinstance(binding_id, str)
            or not binding_id
            or binding_id in siblings
            or not isinstance(commit, str)
            or len(commit) != 40
        ):
            raise MaterializationError("sibling release bindings are malformed")
        path = _relative_repository_path(row.get("path"), field="sibling path")
        if _object_id(BASE_COMMIT, path) != commit or _object_id("HEAD", path) != commit:
            raise MaterializationError(f"sibling release drift for {path}")
        siblings[binding_id] = row

    producer_rows = _require_mapping_list(
        baseline.get("test_producers"), field="baseline.test_producers"
    )
    if not producer_rows or len(producer_rows) > MAX_PREREQUISITE_PRODUCERS:
        raise MaterializationError("baseline.test_producers has an invalid bound")
    producers: dict[str, Mapping[str, Any]] = {}
    for row in producer_rows:
        producer_id = row.get("producer_id")
        if not isinstance(producer_id, str) or not producer_id or producer_id in producers:
            raise MaterializationError("test producer IDs must be non-empty and unique")
        command = row.get("command")
        expected = row.get("expected")
        sources = _require_mapping_list(
            row.get("source_bindings"), field=f"{producer_id}.source_bindings"
        )
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
            or not isinstance(expected, Mapping)
            or not sources
            or row.get("simulated") is not False
            or not isinstance(row.get("evidence_class"), str)
        ):
            raise MaterializationError(f"test producer {producer_id!r} is incomplete")
        if set(expected) != set(PYTEST_OUTCOME_FIELDS):
            raise MaterializationError(
                f"test producer {producer_id!r} has an open expected-outcome schema"
            )
        for field in PYTEST_OUTCOME_FIELDS:
            if type(expected.get(field)) is not int or expected[field] < 0:
                raise MaterializationError(
                    f"test producer {producer_id!r} has invalid expected {field}"
                )
        if expected["collected"] != expected["passed"] + expected["failed"]:
            raise MaterializationError(
                f"test producer {producer_id!r} has inconsistent expected counts"
            )
        expected_failure = row.get("expected_failure")
        if expected["returncode"] == 0:
            if expected_failure is not None:
                raise MaterializationError(
                    f"passing test producer {producer_id!r} declares an expected failure"
                )
        elif not isinstance(expected_failure, Mapping):
            raise MaterializationError(
                f"test producer {producer_id!r} lacks a typed expected failure"
            )
        else:
            allowed_failure_fields = {
                "reason_code",
                "reason_codes",
                "signature",
                "required_output_fragments",
            }
            if not set(expected_failure) <= allowed_failure_fields:
                raise MaterializationError(
                    f"test producer {producer_id!r} has an open expected-failure schema"
                )
            reason_code = expected_failure.get("reason_code")
            reason_codes = expected_failure.get("reason_codes")
            one_reason = isinstance(reason_code, str) and bool(reason_code)
            many_reasons = (
                isinstance(reason_codes, list)
                and bool(reason_codes)
                and len(reason_codes) <= 8
                and all(isinstance(item, str) and item for item in reason_codes)
            )
            fragments = expected_failure.get("required_output_fragments")
            if (
                one_reason == many_reasons
                or not isinstance(expected_failure.get("signature"), str)
                or not expected_failure["signature"]
                or not isinstance(fragments, list)
                or not fragments
                or len(fragments) > 8
                or any(
                    not isinstance(fragment, str)
                    or not fragment
                    or len(fragment.encode("utf-8")) > 512
                    for fragment in fragments
                )
            ):
                raise MaterializationError(
                    f"test producer {producer_id!r} has an invalid typed expected failure"
                )
        for source in sources:
            path = _relative_repository_path(source.get("path"), field="test source path")
            if path not in command:
                raise MaterializationError(
                    f"test producer {producer_id!r} does not execute bound source {path!r}"
                )
            _verify_bound_blob(
                path=path,
                blob_id=source.get("blob_id"),
                baseline_commit=BASE_COMMIT,
            )
        producers[producer_id] = row
    if set(producers) != REQUIRED_PREREQUISITE_PRODUCER_IDS:
        raise MaterializationError(
            "prerequisite test producer set differs from the closed required vocabulary"
        )

    rows = _require_mapping_list(inventory.get("dispositions"), field="dispositions")
    if not rows:
        raise MaterializationError("prerequisite dispositions must not be empty")
    observations: list[dict[str, Any]] = []
    seen_authorities: set[str] = set()
    referenced_producers: set[str] = set()
    python_cache: dict[str, tuple[set[tuple[str, str]], dict[str, object]]] = {}
    for row in rows:
        authority = row.get("authority")
        status = row.get("status")
        if (
            not isinstance(authority, str)
            or not authority
            or authority in seen_authorities
            or status not in PREREQUISITE_STATUSES
        ):
            raise MaterializationError("prerequisite authority/status is invalid or duplicated")
        seen_authorities.add(authority)
        for field in PREREQUISITE_BINDING_FIELDS:
            if not isinstance(row.get(field), list):
                raise MaterializationError(f"{authority}.{field} binding is missing")
        if not row["package_bindings"]:
            raise MaterializationError(f"{authority}.package_bindings must not be empty")
        source_rows = _require_mapping_list(
            row["source_bindings"], field=f"{authority}.source_bindings"
        )
        symbol_rows = _require_mapping_list(
            row["symbol_bindings"], field=f"{authority}.symbol_bindings"
        )
        interface_rows = _require_mapping_list(
            row["interface_bindings"], field=f"{authority}.interface_bindings"
        )
        schema_rows = _require_mapping_list(
            row["schema_bindings"], field=f"{authority}.schema_bindings"
        )
        if status == "missing":
            if (
                source_rows
                or symbol_rows
                or interface_rows
                or schema_rows
                or row["test_producer_bindings"]
            ):
                raise MaterializationError(
                    f"missing authority {authority!r} cannot carry positive authority bindings"
                )
            if not isinstance(row.get("blocker"), str) or not row.get("negative_probes"):
                raise MaterializationError(
                    f"missing authority {authority!r} lacks blocker/negative probe"
                )
        else:
            if not source_rows or not symbol_rows or not interface_rows or not schema_rows:
                raise MaterializationError(
                    f"non-missing authority {authority!r} lacks source/interface/schema bindings"
                )
            if not row["test_producer_bindings"]:
                raise MaterializationError(
                    f"non-missing authority {authority!r} lacks test producer bindings"
                )
        if status in {"available_with_caveats", "incompatible", "stale"} and not isinstance(
            row.get("caveat"), str
        ):
            raise MaterializationError(f"{authority!r} lacks its typed caveat")
        if status == "incompatible" and not isinstance(row.get("blocker"), str):
            raise MaterializationError(f"{authority!r} lacks its incompatibility blocker")

        bound_paths: set[str] = set()
        source_blob_ids: list[str] = []
        for source in source_rows:
            path = _relative_repository_path(source.get("path"), field="source path")
            blob_id = source.get("blob_id")
            _verify_bound_blob(path=path, blob_id=blob_id, baseline_commit=BASE_COMMIT)
            bound_paths.add(path)
            source_blob_ids.append(blob_id)
        for binding in symbol_rows:
            path = _relative_repository_path(binding.get("path"), field="symbol path")
            kind = binding.get("kind")
            symbol = binding.get("symbol")
            if (
                path not in bound_paths
                or kind not in {"class", "function"}
                or not isinstance(symbol, str)
                or not symbol.isidentifier()
            ):
                raise MaterializationError(f"invalid symbol binding for {authority!r}")
            declarations, _ = _python_declarations(path, python_cache)
            if (kind, symbol) not in declarations:
                raise MaterializationError(f"bound {kind} {symbol!r} is absent for {authority!r}")
        for field, binding_rows in (
            ("interface", interface_rows),
            ("schema", schema_rows),
        ):
            for binding in binding_rows:
                path = _relative_repository_path(binding.get("path"), field=f"{field} path")
                symbol = binding.get("symbol")
                if (
                    path not in bound_paths
                    or not isinstance(symbol, str)
                    or not symbol.isidentifier()
                    or "value" not in binding
                ):
                    raise MaterializationError(f"invalid {field} binding for {authority!r}")
                _, constants = _python_declarations(path, python_cache)
                if constants.get(symbol) != binding["value"]:
                    raise MaterializationError(f"{field} binding drift for {authority!r}: {symbol}")
        for package_id in row["package_bindings"]:
            if package_id not in packages:
                raise MaterializationError(f"unknown package binding for {authority!r}")
        for sibling_id in row["submodule_bindings"]:
            if sibling_id not in siblings:
                raise MaterializationError(f"unknown sibling binding for {authority!r}")
        for producer_id in row["test_producer_bindings"]:
            if producer_id not in producers:
                raise MaterializationError(f"unknown test producer for {authority!r}")
            referenced_producers.add(producer_id)

        related_rows = _require_mapping_list(
            row.get("related_non_equivalent_bindings", []),
            field=f"{authority}.related_non_equivalent_bindings",
        )
        for binding in related_rows:
            path = _relative_repository_path(binding.get("path"), field="related path")
            blob_id = binding.get("blob_id")
            _verify_bound_blob(path=path, blob_id=blob_id, baseline_commit=BASE_COMMIT)
            kind = binding.get("kind")
            symbol = binding.get("symbol")
            declarations, _ = _python_declarations(path, python_cache)
            if (
                kind not in {"class", "function"}
                or not isinstance(symbol, str)
                or not symbol.isidentifier()
                or (kind, symbol) not in declarations
            ):
                raise MaterializationError(
                    f"related non-equivalent binding is stale for {authority!r}"
                )

        negative_rows = _require_mapping_list(
            row.get("negative_probes", []), field=f"{authority}.negative_probes"
        )
        for probe in negative_rows:
            kind = probe.get("kind")
            if kind == "git_path_absent":
                path = _relative_repository_path(probe.get("path"), field="negative path")
                if _git_path_exists(BASE_COMMIT, path) or _git_path_exists("HEAD", path):
                    raise MaterializationError(
                        f"negative path probe no longer holds for {authority!r}: {path}"
                    )
            elif kind == "python_class_absent":
                scope = _relative_repository_path(probe.get("scope"), field="probe scope")
                symbol = probe.get("symbol")
                if not isinstance(symbol, str) or not symbol.isidentifier():
                    raise MaterializationError(f"invalid negative symbol probe for {authority!r}")
                if _git_class_exists(BASE_COMMIT, scope=scope, symbol=symbol) or _git_class_exists(
                    "HEAD", scope=scope, symbol=symbol
                ):
                    raise MaterializationError(
                        f"negative class probe no longer holds for {authority!r}: {symbol}"
                    )
            else:
                raise MaterializationError(f"unknown negative probe kind for {authority!r}")
        observations.append(
            {
                "authority": authority,
                "status": status,
                "source_blob_ids": sorted(source_blob_ids),
                "test_producer_ids": sorted(row["test_producer_bindings"]),
                "negative_probe_count": len(negative_rows),
            }
        )
    if seen_authorities != REQUIRED_PREREQUISITE_AUTHORITIES:
        raise MaterializationError(
            "prerequisite authority set differs from the closed required vocabulary"
        )
    if referenced_producers != set(producers):
        unbound = sorted(set(producers) - referenced_producers)
        raise MaterializationError(f"unbound prerequisite test producers: {unbound}")
    return observations


def _submodule_git(path: str, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(REPO_ROOT / path), *args),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode:
        raise MaterializationError(
            result.stderr.strip() or f"git -C {path} {' '.join(args)} failed"
        )
    return result.stdout.strip()


def _verify_exact_gitlink_checkouts(
    baseline: Mapping[str, Any], *, head: str, tree: str
) -> dict[str, Any]:
    """Prove that every declared sibling is populated at its exact gitlink.

    The materializer deliberately never initializes or updates a submodule.  A
    checkout is an environmental prerequisite, while the superproject gitlink
    alone proves only identity and cannot qualify executable sibling code.
    """

    rows = _require_mapping_list(
        baseline.get("sibling_release_bindings"),
        field="baseline.sibling_release_bindings",
    )
    observations: list[dict[str, Any]] = []
    for row in rows:
        path = _relative_repository_path(row.get("path"), field="sibling path")
        expected_commit = row.get("gitlink_commit")
        if _object_id("HEAD", path) != expected_commit:
            raise MaterializationError(f"current-tree gitlink drift for {path}")
        status = _git("submodule", "status", "--", path)
        fields = status.split()
        if len(fields) < 2 or fields[0] != expected_commit or fields[1] != path:
            raise MaterializationError(
                f"exact sibling checkout required for {path}: {status or 'deinitialized'}"
            )
        top_level = Path(_submodule_git(path, "rev-parse", "--show-toplevel")).resolve()
        if top_level != (REPO_ROOT / path).resolve():
            raise MaterializationError(f"sibling checkout boundary mismatch for {path}")
        observed_commit = _submodule_git(path, "rev-parse", "HEAD")
        observed_tree = _submodule_git(path, "rev-parse", "HEAD^{tree}")
        if observed_commit != expected_commit:
            raise MaterializationError(
                f"sibling checkout drift for {path}: expected {expected_commit}, "
                f"observed {observed_commit}"
            )
        dirty = _submodule_git(path, "status", "--porcelain=v1", "--untracked-files=all")
        if dirty:
            raise MaterializationError(f"sibling checkout is dirty: {path}")
        observations.append(
            {
                "binding_id": row["binding_id"],
                "path": path,
                "gitlink_commit": expected_commit,
                "checkout_commit": observed_commit,
                "checkout_tree": observed_tree,
                "clean": True,
            }
        )
    payload = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/procedure-compiler-exact-gitlink-checkouts@1"
        ),
        "program": PROGRAM,
        "repository_commit": head,
        "repository_tree": tree,
        "bindings": observations,
        "binding_count": len(observations),
        "auto_updated": False,
        "simulated": False,
    }
    payload["gitlink_receipt_cid"] = content_identity(payload)
    return payload


def _captured_command_receipt(argv: Sequence[str], *, timeout: int) -> tuple[dict[str, Any], str]:
    started = time.monotonic_ns()
    environment = dict(os.environ)
    environment.update({"NO_COLOR": "1", "PY_COLORS": "0"})
    result = subprocess.run(
        tuple(argv),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
        env=environment,
    )
    elapsed_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
    stdout = result.stdout.encode("utf-8")
    stderr = result.stderr.encode("utf-8")
    receipt = {
        "argv": list(argv),
        "returncode": int(result.returncode),
        "elapsed_ms": int(elapsed_ms),
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stdout_tail": result.stdout[-MAX_CAPTURE_BYTES:],
        "stderr_tail": result.stderr[-MAX_CAPTURE_BYTES:],
    }
    return receipt, f"{result.stdout}\n{result.stderr}"


def _parse_pytest_outcome(output: str, *, returncode: int) -> dict[str, int]:
    terminal_line = ""
    for line in reversed(output.splitlines()):
        if " in " in line and PYTEST_TERMINAL_OUTCOME_RE.search(line):
            terminal_line = line
            break
    if not terminal_line:
        raise MaterializationError("pytest producer emitted no terminal outcome summary")
    counts = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "deselected": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    for match in PYTEST_TERMINAL_OUTCOME_RE.finditer(terminal_line):
        kind = match.group("kind")
        normalized = "errors" if kind in {"error", "errors"} else kind
        counts[normalized] += int(match.group("count"))
    unsupported = {
        field: counts[field]
        for field in ("skipped", "deselected", "xfailed", "xpassed")
        if counts[field]
    }
    if unsupported:
        raise MaterializationError(
            f"pytest producer emitted undeclared terminal outcomes: {unsupported}"
        )
    return {
        "collected": counts["passed"] + counts["failed"],
        "passed": counts["passed"],
        "failed": counts["failed"],
        "errors": counts["errors"],
        "returncode": int(returncode),
    }


def _execute_prerequisite_test_producers(
    *,
    baseline: Mapping[str, Any],
    inventory: Mapping[str, Any],
    head: str,
    tree: str,
    gitlinks: Mapping[str, Any],
) -> dict[str, Any]:
    producer_rows = _require_mapping_list(
        baseline.get("test_producers"), field="baseline.test_producers"
    )
    receipts: list[dict[str, Any]] = []
    receipt_cids: dict[str, str] = {}
    for row in producer_rows:
        producer_id = str(row["producer_id"])
        declared_argv = list(row["command"])
        executed_argv = [sys.executable, *declared_argv[1:]]
        command, output = _captured_command_receipt(
            executed_argv, timeout=PREREQUISITE_PRODUCER_TIMEOUT_SECONDS
        )
        observed = _parse_pytest_outcome(output, returncode=command["returncode"])
        expected = dict(row["expected"])
        if observed != expected:
            raise MaterializationError(
                f"current-tree prerequisite producer {producer_id} drifted: "
                f"expected {expected}, observed {observed}"
            )
        expected_failure = row.get("expected_failure")
        matched_fragments: list[str] = []
        if expected["returncode"]:
            assert isinstance(expected_failure, Mapping)
            fragments = expected_failure["required_output_fragments"]
            missing = [fragment for fragment in fragments if fragment not in output]
            if missing:
                raise MaterializationError(
                    f"current-tree prerequisite producer {producer_id} did not match "
                    f"typed expected failure fragments: {missing}"
                )
            matched_fragments = list(fragments)
            disposition = "matched_typed_expected_failure"
        else:
            disposition = "matched_pass"
        source_blob_ids = sorted(str(source["blob_id"]) for source in row["source_bindings"])
        receipt = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/procedure-compiler-prerequisite-test-receipt@1"
            ),
            "program": PROGRAM,
            "producer_id": producer_id,
            "repository_commit": head,
            "repository_tree": tree,
            "declared_argv": declared_argv,
            "executed_argv": executed_argv,
            "command": command,
            "source_blob_ids": source_blob_ids,
            "expected": expected,
            "observed": observed,
            "expected_failure": dict(expected_failure) if expected_failure else None,
            "matched_failure_fragments": matched_fragments,
            "disposition": disposition,
            "accepted": True,
            "gitlink_receipt_cid": gitlinks["gitlink_receipt_cid"],
            "simulated": False,
        }
        receipt["producer_receipt_cid"] = content_identity(receipt)
        receipts.append(receipt)
        receipt_cids[producer_id] = str(receipt["producer_receipt_cid"])

    authority_receipts: list[dict[str, Any]] = []
    for row in _require_mapping_list(inventory.get("dispositions"), field="dispositions"):
        producer_ids = sorted(str(item) for item in row["test_producer_bindings"])
        authority_receipt = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/procedure-compiler-authority-test-evidence@1"
            ),
            "authority": row["authority"],
            "status": row["status"],
            "repository_commit": head,
            "repository_tree": tree,
            "producer_ids": producer_ids,
            "producer_receipt_cids": [receipt_cids[item] for item in producer_ids],
            "evidence_disposition": (
                "current_execution" if producer_ids else "not_applicable_missing_authority"
            ),
            "accepted": True,
            "simulated": False,
        }
        authority_receipt["authority_receipt_cid"] = content_identity(authority_receipt)
        authority_receipts.append(authority_receipt)

    payload = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "procedure-compiler-current-prerequisite-execution@1"
        ),
        "program": PROGRAM,
        "repository_commit": head,
        "repository_tree": tree,
        "gitlink_receipt": dict(gitlinks),
        "producer_receipts": receipts,
        "producer_count": len(receipts),
        "authority_receipts": authority_receipts,
        "authority_count": len(authority_receipts),
        "typed_expected_failure_count": sum(
            receipt["disposition"] == "matched_typed_expected_failure" for receipt in receipts
        ),
        "all_declared_outcomes_matched": True,
        "simulated": False,
    }
    payload["execution_cid"] = content_identity(payload)
    return payload


def _probe_prerequisite_inventory() -> dict[str, Any]:
    baseline_path = REPO_ROOT / BASELINE_RELATIVE
    prerequisites_path = REPO_ROOT / PREREQUISITES_RELATIVE
    baseline = _read_json(baseline_path)
    inventory = _read_json(prerequisites_path)
    observations = _validate_prerequisite_payloads(baseline, inventory)
    payload = {
        "schema": ("ipfs_accelerate_py/agent-supervisor/procedure-compiler-prerequisite-probe@1"),
        "program": PROGRAM,
        "baseline_commit": BASE_COMMIT,
        "baseline_tree": baseline["repository"]["tree"],
        "repository_commit": _git("rev-parse", "HEAD"),
        "repository_tree": _git("rev-parse", "HEAD^{tree}"),
        "baseline_sha256": _sha256_file(baseline_path),
        "prerequisites_sha256": _sha256_file(prerequisites_path),
        "authorities": observations,
        "authority_count": len(observations),
        "test_producer_count": len(baseline["test_producers"]),
        "source_drift_permitted": False,
        "simulated": False,
    }
    payload["probe_cid"] = content_identity(payload)
    return payload


def _stored_prerequisite_probe_is_intact(probe: object, *, head: str, tree: str) -> bool:
    if not isinstance(probe, Mapping):
        return False
    baseline_path = REPO_ROOT / BASELINE_RELATIVE
    prerequisites_path = REPO_ROOT / PREREQUISITES_RELATIVE
    return (
        probe.get("schema")
        == "ipfs_accelerate_py/agent-supervisor/procedure-compiler-prerequisite-probe@1"
        and probe.get("program") == PROGRAM
        and probe.get("baseline_commit") == BASE_COMMIT
        and probe.get("repository_commit") == head
        and probe.get("repository_tree") == tree
        and probe.get("baseline_sha256") == _sha256_file(baseline_path)
        and probe.get("prerequisites_sha256") == _sha256_file(prerequisites_path)
        and type(probe.get("authority_count")) is int
        and probe.get("authority_count", 0) > 0
        and type(probe.get("test_producer_count")) is int
        and probe.get("test_producer_count", 0) > 0
        and isinstance(probe.get("authorities"), list)
        and len(probe["authorities"]) == probe["authority_count"]
        and probe.get("source_drift_permitted") is False
        and probe.get("simulated") is False
        and _has_valid_embedded_identity(probe, identity_field="probe_cid")
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _command_receipt(argv: Sequence[str], *, timeout: int) -> dict[str, Any]:
    receipt, _ = _captured_command_receipt(argv, timeout=timeout)
    return receipt


def _has_valid_embedded_identity(payload: Mapping[str, Any], *, identity_field: str) -> bool:
    claimed = payload.get(identity_field)
    if not isinstance(claimed, str) or not claimed:
        return False
    unsigned = dict(payload)
    unsigned.pop(identity_field, None)
    return content_identity(unsigned) == claimed


def _valid_command_receipt(receipt: object, *, argv: Sequence[str], returncode: int) -> bool:
    if not isinstance(receipt, Mapping):
        return False
    if set(receipt) != {
        "argv",
        "returncode",
        "elapsed_ms",
        "stdout_bytes",
        "stderr_bytes",
        "stdout_sha256",
        "stderr_sha256",
        "stdout_tail",
        "stderr_tail",
    }:
        return False
    if receipt.get("argv") != list(argv) or receipt.get("returncode") != returncode:
        return False
    for field in ("elapsed_ms", "stdout_bytes", "stderr_bytes"):
        value = receipt.get(field)
        if type(value) is not int or value < 0:
            return False
    for field in ("stdout_sha256", "stderr_sha256"):
        value = receipt.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            return False
    for field in ("stdout_tail", "stderr_tail"):
        value = receipt.get(field)
        if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_CAPTURE_BYTES:
            return False
    return True


def _stored_prerequisite_execution_is_intact(execution: object, *, head: str, tree: str) -> bool:
    if not isinstance(execution, Mapping):
        return False
    if set(execution) != {
        "schema",
        "program",
        "repository_commit",
        "repository_tree",
        "gitlink_receipt",
        "producer_receipts",
        "producer_count",
        "authority_receipts",
        "authority_count",
        "typed_expected_failure_count",
        "all_declared_outcomes_matched",
        "simulated",
        "execution_cid",
    }:
        return False
    try:
        baseline = _read_json(REPO_ROOT / BASELINE_RELATIVE)
        inventory = _read_json(REPO_ROOT / PREREQUISITES_RELATIVE)
    except (OSError, ValueError, MaterializationError):
        return False
    producers = baseline.get("test_producers")
    dispositions = inventory.get("dispositions")
    if not isinstance(producers, list) or not isinstance(dispositions, list):
        return False
    by_id = {
        str(row.get("producer_id")): row
        for row in producers
        if isinstance(row, Mapping) and isinstance(row.get("producer_id"), str)
    }
    producer_receipts = execution.get("producer_receipts")
    authority_receipts = execution.get("authority_receipts")
    gitlinks = execution.get("gitlink_receipt")
    sibling_rows = baseline.get("sibling_release_bindings")
    if not isinstance(sibling_rows, list):
        return False
    expected_gitlinks = [
        {
            "binding_id": row["binding_id"],
            "path": row["path"],
            "gitlink_commit": row["gitlink_commit"],
        }
        for row in sibling_rows
        if isinstance(row, Mapping)
    ]
    if len(expected_gitlinks) != len(sibling_rows):
        return False
    if (
        execution.get("schema")
        != (
            "ipfs_accelerate_py/agent-supervisor/"
            "procedure-compiler-current-prerequisite-execution@1"
        )
        or execution.get("program") != PROGRAM
        or execution.get("repository_commit") != head
        or execution.get("repository_tree") != tree
        or execution.get("producer_count") != len(producers)
        or execution.get("authority_count") != len(dispositions)
        or execution.get("all_declared_outcomes_matched") is not True
        or execution.get("simulated") is not False
        or not isinstance(producer_receipts, list)
        or len(producer_receipts) != len(producers)
        or not isinstance(authority_receipts, list)
        or len(authority_receipts) != len(dispositions)
        or not isinstance(gitlinks, Mapping)
        or set(gitlinks)
        != {
            "schema",
            "program",
            "repository_commit",
            "repository_tree",
            "bindings",
            "binding_count",
            "auto_updated",
            "simulated",
            "gitlink_receipt_cid",
        }
        or gitlinks.get("schema")
        != ("ipfs_accelerate_py/agent-supervisor/procedure-compiler-exact-gitlink-checkouts@1")
        or gitlinks.get("repository_commit") != head
        or gitlinks.get("repository_tree") != tree
        or gitlinks.get("binding_count") != len(expected_gitlinks)
        or not isinstance(gitlinks.get("bindings"), list)
        or len(gitlinks["bindings"]) != len(expected_gitlinks)
        or any(
            set(observed)
            != {
                "binding_id",
                "path",
                "gitlink_commit",
                "checkout_commit",
                "checkout_tree",
                "clean",
            }
            or observed.get("binding_id") != expected["binding_id"]
            or observed.get("path") != expected["path"]
            or observed.get("gitlink_commit") != expected["gitlink_commit"]
            or observed.get("checkout_commit") != expected["gitlink_commit"]
            or not isinstance(observed.get("checkout_tree"), str)
            or len(observed["checkout_tree"]) != 40
            or observed.get("clean") is not True
            for observed, expected in zip(gitlinks["bindings"], expected_gitlinks, strict=True)
            if isinstance(observed, Mapping)
        )
        or any(not isinstance(observed, Mapping) for observed in gitlinks["bindings"])
        or gitlinks.get("auto_updated") is not False
        or gitlinks.get("simulated") is not False
        or not _has_valid_embedded_identity(gitlinks, identity_field="gitlink_receipt_cid")
        or not _has_valid_embedded_identity(execution, identity_field="execution_cid")
    ):
        return False
    observed_cids: dict[str, str] = {}
    typed_failure_count = 0
    for receipt in producer_receipts:
        if not isinstance(receipt, Mapping):
            return False
        producer_id = receipt.get("producer_id")
        row = by_id.get(str(producer_id))
        if row is None or str(producer_id) in observed_cids:
            return False
        declared_argv = list(row["command"])
        executed_argv = [sys.executable, *declared_argv[1:]]
        expected = dict(row["expected"])
        expected_failure = row.get("expected_failure")
        disposition = "matched_typed_expected_failure" if expected["returncode"] else "matched_pass"
        tails = ""
        command = receipt.get("command")
        if isinstance(command, Mapping):
            tails = f"{command.get('stdout_tail', '')}\n{command.get('stderr_tail', '')}"
        fragments = (
            list(expected_failure["required_output_fragments"])
            if isinstance(expected_failure, Mapping)
            else []
        )
        source_blob_ids = sorted(str(source["blob_id"]) for source in row["source_bindings"])
        if (
            set(receipt)
            != {
                "schema",
                "program",
                "producer_id",
                "repository_commit",
                "repository_tree",
                "declared_argv",
                "executed_argv",
                "command",
                "source_blob_ids",
                "expected",
                "observed",
                "expected_failure",
                "matched_failure_fragments",
                "disposition",
                "accepted",
                "gitlink_receipt_cid",
                "simulated",
                "producer_receipt_cid",
            }
            or receipt.get("schema")
            != (
                "ipfs_accelerate_py/agent-supervisor/procedure-compiler-prerequisite-test-receipt@1"
            )
            or receipt.get("program") != PROGRAM
            or receipt.get("repository_commit") != head
            or receipt.get("repository_tree") != tree
            or receipt.get("declared_argv") != declared_argv
            or receipt.get("executed_argv") != executed_argv
            or receipt.get("expected") != expected
            or receipt.get("observed") != expected
            or receipt.get("source_blob_ids") != source_blob_ids
            or receipt.get("expected_failure")
            != (dict(expected_failure) if expected_failure else None)
            or receipt.get("matched_failure_fragments") != fragments
            or any(fragment not in tails for fragment in fragments)
            or receipt.get("disposition") != disposition
            or receipt.get("accepted") is not True
            or receipt.get("simulated") is not False
            or receipt.get("gitlink_receipt_cid") != gitlinks.get("gitlink_receipt_cid")
            or not _valid_command_receipt(
                command, argv=executed_argv, returncode=expected["returncode"]
            )
            or not _has_valid_embedded_identity(receipt, identity_field="producer_receipt_cid")
        ):
            return False
        cid = receipt.get("producer_receipt_cid")
        assert isinstance(cid, str)
        observed_cids[str(producer_id)] = cid
        typed_failure_count += disposition == "matched_typed_expected_failure"
    if set(observed_cids) != set(by_id):
        return False
    if execution.get("typed_expected_failure_count") != typed_failure_count:
        return False
    authority_by_name = {
        str(row.get("authority")): row
        for row in dispositions
        if isinstance(row, Mapping) and isinstance(row.get("authority"), str)
    }
    seen_authorities: set[str] = set()
    for receipt in authority_receipts:
        if not isinstance(receipt, Mapping):
            return False
        authority = str(receipt.get("authority"))
        row = authority_by_name.get(authority)
        if row is None or authority in seen_authorities:
            return False
        producer_ids = sorted(str(item) for item in row["test_producer_bindings"])
        evidence_disposition = (
            "current_execution" if producer_ids else "not_applicable_missing_authority"
        )
        if (
            set(receipt)
            != {
                "schema",
                "authority",
                "status",
                "repository_commit",
                "repository_tree",
                "producer_ids",
                "producer_receipt_cids",
                "evidence_disposition",
                "accepted",
                "simulated",
                "authority_receipt_cid",
            }
            or receipt.get("schema")
            != ("ipfs_accelerate_py/agent-supervisor/procedure-compiler-authority-test-evidence@1")
            or receipt.get("authority") != authority
            or receipt.get("producer_ids") != producer_ids
            or receipt.get("producer_receipt_cids")
            != [observed_cids[item] for item in producer_ids]
            or receipt.get("repository_commit") != head
            or receipt.get("repository_tree") != tree
            or receipt.get("status") != row["status"]
            or receipt.get("evidence_disposition") != evidence_disposition
            or receipt.get("accepted") is not True
            or receipt.get("simulated") is not False
            or not _has_valid_embedded_identity(receipt, identity_field="authority_receipt_cid")
        ):
            return False
        seen_authorities.add(authority)
    return seen_authorities == set(authority_by_name)


def _stored_qualification_receipt_is_intact(
    qualification: Mapping[str, Any],
    *,
    head: str,
    tree: str,
    require_prerequisite_probe: bool = False,
) -> bool:
    commands = qualification.get("commands")
    if not isinstance(commands, list) or len(commands) != len(QUALIFICATION_COMMANDS):
        return False
    for observed, expected_argv in zip(commands, QUALIFICATION_COMMANDS, strict=True):
        if not _valid_command_receipt(observed, argv=expected_argv, returncode=0):
            return False
    valid = (
        qualification.get("schema")
        == "ipfs_accelerate_py/agent-supervisor/procedure-compiler-p0-qualification@2"
        and qualification.get("program") == PROGRAM
        and qualification.get("repository_commit") == head
        and qualification.get("repository_tree") == tree
        and qualification.get("branch") == BRANCH
        and qualification.get("p0_tasks") == list(P0_TASKS)
        and qualification.get("test_evidence_class") == "current_tree_hermetic"
        and qualification.get("simulated") is False
        and _has_valid_embedded_identity(qualification, identity_field="qualification_cid")
    )
    if not valid:
        return False
    if require_prerequisite_probe:
        return _stored_prerequisite_probe_is_intact(
            qualification.get("prerequisite_probe"), head=head, tree=tree
        ) and _stored_prerequisite_execution_is_intact(
            qualification.get("prerequisite_execution"), head=head, tree=tree
        )
    return True


def _qualify_exact_tree() -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    if branch != BRANCH:
        raise MaterializationError(f"required branch {BRANCH!r}, observed {branch!r}")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        timeout=60,
    )
    if ancestor.returncode:
        raise MaterializationError("starting commit is not an ancestor of HEAD")
    dirty = _git("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise MaterializationError("qualification requires a clean tracked and untracked checkout")

    prerequisite_probe = _probe_prerequisite_inventory()
    baseline = _read_json(REPO_ROOT / BASELINE_RELATIVE)
    inventory = _read_json(REPO_ROOT / PREREQUISITES_RELATIVE)
    gitlink_receipt = _verify_exact_gitlink_checkouts(baseline, head=head, tree=tree)
    prerequisite_execution = _execute_prerequisite_test_producers(
        baseline=baseline,
        inventory=inventory,
        head=head,
        tree=tree,
        gitlinks=gitlink_receipt,
    )
    receipts = [_command_receipt(command, timeout=900) for command in QUALIFICATION_COMMANDS]
    failures = [item for item in receipts if item["returncode"] != 0]
    if failures:
        summary = [
            {
                "argv": item["argv"],
                "returncode": item["returncode"],
                "stderr_tail": item["stderr_tail"][-2000:],
            }
            for item in failures
        ]
        raise MaterializationError(f"exact-tree qualification failed: {summary}")
    if _git("rev-parse", "HEAD") != head or _git("rev-parse", "HEAD^{tree}") != tree:
        raise MaterializationError("repository identity changed during qualification")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise MaterializationError("qualification changed the checkout")
    post_command_probe = _probe_prerequisite_inventory()
    if post_command_probe.get("probe_cid") != prerequisite_probe.get("probe_cid"):
        raise MaterializationError("prerequisite bindings changed during qualification")
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/procedure-compiler-p0-qualification@2",
        "program": PROGRAM,
        "repository_commit": head,
        "repository_tree": tree,
        "branch": branch,
        "commands": receipts,
        "prerequisite_probe": post_command_probe,
        "prerequisite_execution": prerequisite_execution,
        "p0_tasks": list(P0_TASKS),
        "test_evidence_class": "current_tree_hermetic",
        "simulated": False,
    }
    payload["qualification_cid"] = content_identity(payload)
    return payload


def _population(*, head: str, tree: str, qualification_cid: str) -> tuple[dict[str, Any], str]:
    board = load_configured_board(CONFIG_RELATIVE, repo_root=REPO_ROOT)
    todo_path = REPO_ROOT / board.taskboard_path
    objective_path = REPO_ROOT / board.objectives_path
    tasks = parse_task_text(
        todo_path.read_text(encoding="utf-8"),
        path=todo_path,
        task_header_prefix="## PCPC-",
    )
    goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    goal_cids = {
        goal.goal_id: content_identity(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/procedure-compiler-goal@1",
                "program": PROGRAM,
                "goal_id": goal.goal_id,
                "title": goal.title,
                "fields": dict(goal.fields),
            }
        )
        for goal in goals
    }
    tracked_inputs = {}
    for relative in (board.taskboard_path, board.objectives_path, board.plan_path, CONFIG_RELATIVE):
        body = (REPO_ROOT / relative).read_bytes()
        tracked_inputs[relative] = hashlib.sha256(body).hexdigest()
    plan_cid = content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/procedure-compiler-plan-root@1",
            "program": PROGRAM,
            "repository_commit": head,
            "repository_tree": tree,
            "tracked_inputs_sha256": tracked_inputs,
        }
    )
    objectives: list[dict[str, Any]] = []
    for ordinal, goal in enumerate(goals, start=1):
        fields = dict(goal.fields)
        parent = str(fields.get("parent") or "")
        objectives.append(
            {
                "goal_cid": goal_cids[goal.goal_id],
                "goal_id": goal.goal_id,
                "goal_alias": goal.goal_id,
                "objective_id": "PCPC-G000" if goal.goal_id == "PCPC-G000" else "",
                "objective_alias": "PCPC-G000",
                "title": goal.title,
                "parent_goal_cid": goal_cids.get(parent, ""),
                "ordinal": ordinal,
                "status": "open",
                "priority": fields.get("priority", "P2"),
                "fields": fields,
            }
        )
    taskboard: list[dict[str, Any]] = []
    for ordinal, task in enumerate(tasks, start=1):
        goal_id = str(task.metadata.get("goal id") or "PCPC-G000")
        acceptance: list[Any]
        if task.task_id in P0_TASKS:
            acceptance = [
                {
                    "criterion": task.acceptance,
                    "required_digest": qualification_cid,
                    "evidence_kind": "validation",
                }
            ]
        else:
            acceptance = [task.acceptance]
        taskboard.append(
            {
                "task_cid": task.canonical_task_cid,
                "task_id": task.task_id,
                "goal_cid": goal_cids[goal_id],
                "goal_id": goal_id,
                "plan_cid": plan_cid,
                "objective_id": "PCPC-G000",
                "ordinal": ordinal,
                "title": task.title,
                "status": "ready",
                "priority": task.priority,
                "depends_on": list(task.depends_on),
                "outputs": [
                    {"path": path, "effect": task.metadata.get("effect class", "declared")}
                    for path in task.outputs
                ],
                "acceptance_criteria": acceptance,
                "validation_commands": list(task.validation),
                "board_status": task.status,
                "metadata": dict(task.metadata),
                "repository_commit": head,
                "repository_tree": tree,
            }
        )
    population = {
        "schema": "ipfs_accelerate_py/agent-supervisor/procedure-compiler-population@1",
        "repository_tree_id": tree,
        "plan_root_cid": plan_cid,
        "objectives": objectives,
        "goal_edges": [
            {
                "parent_goal_cid": goal_cids["PCPC-G000"],
                "child_goal_cid": goal_cids[goal_id],
                "edge_kind": "refinement",
            }
            for goal_id in GOAL_IDS[1:]
        ],
        "plans": [
            {
                "plan_cid": plan_cid,
                "plan_alias": PROGRAM,
                "goal_cid": goal_cids["PCPC-G000"],
                "status": "active",
                "repository_commit": head,
                "repository_tree": tree,
                "tracked_inputs_sha256": tracked_inputs,
            }
        ],
        "taskboard": taskboard,
    }
    return population, plan_cid


GOAL_IDS = ("PCPC-G000", "PCPC-G010", "PCPC-G020", "PCPC-G030", "PCPC-G040")


def _extension_profile() -> dict[str, Any]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - environment gate
        return {"available": False, "reason": f"ImportError: {exc}"}
    connection = duckdb.connect(":memory:")
    rows: list[dict[str, Any]] = []
    try:
        for name in ("quack", "ducklake", "httpfs"):
            try:
                connection.execute(f"LOAD {name}")
                record = connection.execute(
                    "SELECT extension_name, extension_version, installed, loaded "
                    "FROM duckdb_extensions() WHERE extension_name = ?",
                    [name],
                ).fetchone()
                rows.append(
                    {
                        "name": name,
                        "version": str(record[1] or "") if record else "",
                        "installed": bool(record[2]) if record else False,
                        "loaded": bool(record[3]) if record else True,
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - typed capability result
                rows.append(
                    {
                        "name": name,
                        "version": "",
                        "installed": False,
                        "loaded": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return {
            "available": all(item["loaded"] for item in rows),
            "duckdb_version": str(duckdb.__version__),
            "extensions": rows,
        }
    finally:
        connection.close()


def _project_ducklake(
    *,
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    qualification: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    projection = config.get("ducklake_projection_program")
    if not isinstance(projection, Mapping):
        return {"projected": False, "reason": "configuration_missing"}
    catalog = REPO_ROOT / str(projection["catalog_path"])
    data = REPO_ROOT / str(projection["data_path"])
    for path in (catalog, data):
        try:
            path.resolve(strict=False).relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise MaterializationError("DuckLake path escapes repository runtime root") from exc
        if "'" in str(path) or "\x00" in str(path):
            raise MaterializationError("DuckLake path cannot be represented safely")
    catalog.parent.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    try:
        import duckdb

        connection = duckdb.connect(":memory:")
        try:
            connection.execute("LOAD ducklake")
            connection.execute("LOAD httpfs")
            connection.execute(f"ATTACH 'ducklake:{catalog}' AS pcpc_history (DATA_PATH '{data}')")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pcpc_history.program_runs (
                    run_id VARCHAR,
                    repository_commit VARCHAR,
                    repository_tree VARCHAR,
                    plan_root_cid VARCHAR,
                    qualification_cid VARCHAR,
                    task_count INTEGER,
                    ready_count INTEGER,
                    projected_at_epoch_ms BIGINT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pcpc_history.task_history (
                    run_id VARCHAR,
                    task_cid VARCHAR,
                    task_alias VARCHAR,
                    status VARCHAR,
                    revision BIGINT,
                    goal_cid VARCHAR
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pcpc_history.qualification (
                    run_id VARCHAR,
                    qualification_cid VARCHAR,
                    repository_commit VARCHAR,
                    repository_tree VARCHAR,
                    test_evidence_class VARCHAR,
                    simulated BOOLEAN,
                    projected_at_epoch_ms BIGINT
                )
                """
            )
            run_id = str(run["run_id"])
            connection.execute("DELETE FROM pcpc_history.program_runs WHERE run_id = ?", [run_id])
            connection.execute("DELETE FROM pcpc_history.task_history WHERE run_id = ?", [run_id])
            connection.execute("DELETE FROM pcpc_history.qualification WHERE run_id = ?", [run_id])
            connection.execute(
                "INSERT INTO pcpc_history.program_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    run["repository_commit"],
                    run["repository_tree"],
                    run["plan_root_cid"],
                    run["qualification_cid"],
                    int(run["task_count"]),
                    int(run["ready_count"]),
                    int(time.time() * 1000),
                ],
            )
            for task in tasks[:4096]:
                connection.execute(
                    "INSERT INTO pcpc_history.task_history VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        run_id,
                        task["task_cid"],
                        task["task_alias"],
                        task["status"],
                        int(task["revision"]),
                        task["goal_cid"],
                    ],
                )
            connection.execute(
                "INSERT INTO pcpc_history.qualification VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    qualification["qualification_cid"],
                    qualification["repository_commit"],
                    qualification["repository_tree"],
                    qualification["test_evidence_class"],
                    bool(qualification["simulated"]),
                    int(time.time() * 1000),
                ],
            )
            observed = connection.execute(
                "SELECT COUNT(*) FROM pcpc_history.task_history WHERE run_id = ?", [run_id]
            ).fetchone()
            qualification_rows = connection.execute(
                "SELECT COUNT(*) FROM pcpc_history.qualification WHERE run_id = ?",
                [run_id],
            ).fetchone()
            return {
                "projected": True,
                "authority": False,
                "catalog_path": catalog.relative_to(REPO_ROOT).as_posix(),
                "data_path": data.relative_to(REPO_ROOT).as_posix(),
                "run_id": run_id,
                "task_rows": int(observed[0]) if observed else 0,
                "qualification_rows": (int(qualification_rows[0]) if qualification_rows else 0),
            }
        finally:
            connection.close()
    except Exception as exc:  # noqa: BLE001 - non-authoritative projection receipt
        return {"projected": False, "authority": False, "reason": f"{type(exc).__name__}: {exc}"}


def materialize() -> dict[str, Any]:
    qualification = _qualify_exact_tree()
    head = str(qualification["repository_commit"])
    tree = str(qualification["repository_tree"])
    config = _read_json(REPO_ROOT / CONFIG_RELATIVE)
    program = config.get("database_program")
    if not isinstance(program, Mapping):
        raise MaterializationError("database_program is missing")
    database_path = REPO_ROOT / str(program["store_id"])
    try:
        database_path.resolve(strict=False).relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise MaterializationError("control database escapes repository") from exc
    if database_path.exists():
        raise MaterializationError(
            f"refusing to overwrite existing control database: {database_path}"
        )
    database_path.parent.mkdir(parents=True, exist_ok=True)
    qualification_path = (
        REPO_ROOT / str(config["runtime_paths"]["evidence"]) / "p0-qualification.json"
    )
    _atomic_json(qualification_path, qualification)
    population, plan_cid = _population(
        head=head, tree=tree, qualification_cid=str(qualification["qualification_cid"])
    )

    with DatabaseTaskSource(
        database_path,
        owner_id="pcpc-bootstrap:single-writer",
        repository_tree_id=tree,
        plan_root_cid=plan_cid,
    ) as source:
        materialization = dict(
            source.materialize(population, repository_tree_id=tree, plan_root_cid=plan_cid)
        )
        for task_id in P0_TASKS:
            task = source.get_task(task_id)
            if task is None:
                raise MaterializationError(f"materialized task is absent: {task_id}")
            source.record_validation_result(
                task_cid=task.task_cid,
                outcome="passed",
                evidence_digest=str(qualification["qualification_cid"]),
                argv=[sys.executable, "-m", "pytest", "-q", "test/api/procedure_compiler"],
                body={"repository_commit": head, "repository_tree": tree, "simulated": False},
            )
            current = source.get_task(task.task_cid)
            if current is None:
                raise MaterializationError(f"task vanished before completion: {task_id}")
            source.compare_and_set_status(
                current.task_cid,
                current.revision,
                "completed",
                {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/procedure-compiler-p0-completion@1"
                    ),
                    "task_id": task_id,
                    "qualification_cid": qualification["qualification_cid"],
                    "repository_commit": head,
                    "repository_tree": tree,
                },
                evidence_digests=[str(qualification["qualification_cid"])],
            )
        snapshot = source.snapshot()
        ready = tuple(item.task_alias for item in source.ready_tasks(limit=64).tasks)
        records = [item.to_dict() for item in source.list_tasks(limit=64).tasks]
        projection_matches = source.projection_matches_events()
        final_snapshot = source.snapshot()
    if ready != EXPECTED_READY:
        raise MaterializationError(
            f"ready task mismatch: expected {EXPECTED_READY}, observed {ready}"
        )
    blocked = [item["task_alias"] for item in records if item["status"] == "blocked"]
    if blocked:
        raise MaterializationError(f"materialization produced blocked tasks: {blocked}")
    run = {
        "run_id": content_identity(
            {
                "program": PROGRAM,
                "repository_commit": head,
                "repository_tree": tree,
                "plan_root_cid": plan_cid,
                "qualification_cid": qualification["qualification_cid"],
            }
        ),
        "repository_commit": head,
        "repository_tree": tree,
        "plan_root_cid": plan_cid,
        "qualification_cid": qualification["qualification_cid"],
        "task_count": len(records),
        "ready_count": len(ready),
    }
    ducklake = _project_ducklake(
        config=config,
        run=run,
        qualification=qualification,
        tasks=records,
    )
    receipt = {
        "schema": "ipfs_accelerate_py/agent-supervisor/procedure-compiler-materialization@1",
        "program": PROGRAM,
        **run,
        "database_path": database_path.relative_to(REPO_ROOT).as_posix(),
        "database_authority": True,
        "quack_required_for_live_mutation": True,
        "completed_task_ids": list(P0_TASKS),
        "ready_task_ids": list(ready),
        "blocked_task_ids": blocked,
        "materialization": materialization,
        "projection_matches_events": projection_matches,
        "initial_projection_cid": snapshot.projection_cid,
        "final_projection_cid": final_snapshot.projection_cid,
        "extension_profile": _extension_profile(),
        "ducklake_projection": ducklake,
        "simulated": False,
    }
    receipt["receipt_cid"] = content_identity(receipt)
    receipt_path = REPO_ROOT / str(config["runtime_paths"]["evidence"]) / "materialization.json"
    _atomic_json(receipt_path, receipt)
    return receipt


def verify_existing() -> dict[str, Any]:
    # Stored command-shaped JSON and a public CID are provenance, not producer
    # authority.  Re-run the exact-tree qualification here so launch admission
    # depends on current observed test results rather than a replayable file.
    fresh_qualification = _qualify_exact_tree()
    config = _read_json(REPO_ROOT / CONFIG_RELATIVE)
    database_path = REPO_ROOT / str(config["database_program"]["store_id"])
    if not database_path.is_file():
        raise MaterializationError("control database is absent")
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    evidence_root = REPO_ROOT / str(config["runtime_paths"]["evidence"])
    materialization_receipt = _read_json(evidence_root / "materialization.json")
    qualification = _read_json(evidence_root / "p0-qualification.json")
    # A freshly opened DatabaseTaskSource does not infer a plan root from task
    # goals: PCPC tasks are owned by refinement goals while the single plan is
    # rooted at PCPC-G000.  Recompute the expected root from the exact current
    # tree and tracked board inputs, instead of trusting the stored receipt or
    # asking ``snapshot()`` to guess it from a child goal.
    _, expected_plan_cid = _population(
        head=head,
        tree=tree,
        qualification_cid=str(fresh_qualification.get("qualification_cid") or ""),
    )
    with DatabaseTaskSource(
        database_path,
        install_schema=False,
        repository_tree_id=tree,
        plan_root_cid=expected_plan_cid,
    ) as source:
        records = [item.to_dict() for item in source.list_tasks(limit=64).tasks]
        ready = [item.task_alias for item in source.ready_tasks(limit=64).tasks]
        snapshot = source.snapshot()
        matches = source.projection_matches_events()
        plan = source.get_plan(snapshot.plan_root_cid)
    completed = [item["task_alias"] for item in records if item["status"] == "completed"]
    blocked = [item["task_alias"] for item in records if item["status"] == "blocked"]
    plan_body = plan.get("body") if isinstance(plan, Mapping) else None
    plan_current = (
        isinstance(plan_body, Mapping)
        and plan_body.get("repository_commit") == head
        and plan_body.get("repository_tree") == tree
    )
    tasks_current = all(
        isinstance(item.get("body"), Mapping)
        and item["body"].get("repository_commit") == head
        and item["body"].get("repository_tree") == tree
        for item in records
    )
    receipt_current = (
        materialization_receipt.get("schema")
        == "ipfs_accelerate_py/agent-supervisor/procedure-compiler-materialization@1"
        and materialization_receipt.get("program") == PROGRAM
        and materialization_receipt.get("repository_commit") == head
        and materialization_receipt.get("repository_tree") == tree
        and snapshot.plan_root_cid == expected_plan_cid
        and materialization_receipt.get("plan_root_cid") == snapshot.plan_root_cid
        and materialization_receipt.get("simulated") is False
        and materialization_receipt.get("qualification_cid")
        == qualification.get("qualification_cid")
        and _has_valid_embedded_identity(materialization_receipt, identity_field="receipt_cid")
        and _stored_qualification_receipt_is_intact(
            qualification,
            head=head,
            tree=tree,
            require_prerequisite_probe=True,
        )
    )
    freshly_qualified = (
        fresh_qualification.get("repository_commit") == head
        and fresh_qualification.get("repository_tree") == tree
        and fresh_qualification.get("simulated") is False
        and _has_valid_embedded_identity(fresh_qualification, identity_field="qualification_cid")
    )
    valid = (
        len(records) == 32
        and completed == list(P0_TASKS)
        and ready == list(EXPECTED_READY)
        and not blocked
        and matches
        and plan_current
        and tasks_current
        and receipt_current
        and freshly_qualified
    )
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/procedure-compiler-materialization-verification@1"
        ),
        "valid": valid,
        "repository_commit": head,
        "repository_tree": tree,
        "database_path": database_path.relative_to(REPO_ROOT).as_posix(),
        "projection_cid": snapshot.projection_cid,
        "task_count": len(records),
        "completed_task_ids": completed,
        "ready_task_ids": ready,
        "blocked_task_ids": blocked,
        "projection_matches_events": matches,
        "plan_current": plan_current,
        "tasks_current": tasks_current,
        "qualification_current": receipt_current,
        "freshly_qualified": freshly_qualified,
        "fresh_qualification_cid": fresh_qualification.get("qualification_cid", ""),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = materialize() if args.materialize else verify_existing()
        valid = result.get("valid", True) is True
    except Exception as exc:  # noqa: BLE001 - CLI must return typed failure
        result = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/procedure-compiler-materialization-error@1"
            ),
            "valid": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        valid = False
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
