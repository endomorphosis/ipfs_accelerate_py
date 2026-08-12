#!/usr/bin/env python3
"""Fail-closed validator for the IPFS Datasets logic-parser Wave-2 board."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (  # noqa: E402
    ConfiguredBoardError,
    configured_board_launch_plan,
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    parse_task_text,
)

PLAN_PATH = REPO_ROOT / "docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_V2_PLAN.md"
OBJECTIVE_PATH = REPO_ROOT / "docs/architecture/ipfs_datasets_logic_family_parser_v2.objectives.md"
TODO_PATH = REPO_ROOT / "docs/architecture/ipfs_datasets_logic_family_parser_v2.todo.md"
SCHEDULER_PATH = REPO_ROOT / "config/agent_supervisor_ipfs_datasets_logic_family_parser_v2_scheduler.json"

BOARD_NAMESPACE = "ipfs-datasets-logic-family-parser-v2"
MERGE_TARGET_BRANCH = "agent/logic-family-parser-v2-supervisor"
RUNTIME_ROOT = "data/agent_supervisor/ipfs_datasets_logic_family_parser_v2"
TASK_IDS = tuple(f"LFP2-{index:03d}" for index in range(51))
GOAL_IDS = ("LFP2-G000",) + tuple(
    f"LFP2-G{index:03d}" for index in range(10, 101, 10)
)
INITIAL_COMPLETED = ("LFP2-000",)
INITIAL_READY = ("LFP2-001", "LFP2-002", "LFP2-003", "LFP2-004")
TERMINAL_TASK = "LFP2-050"

FIXED_POINT_PATH = (
    REPO_ROOT / RUNTIME_ROOT / "refill/fixed_point_receipt.json"
)
GAP_LEDGER_PATH = REPO_ROOT / RUNTIME_ROOT / "refill/gap_ledger.jsonl"
RELEASE_MARKDOWN_RELATIVE_PATH = Path(
    "docs/architecture/logic/LOGIC_FAMILY_PARSER_V2_RELEASE.md"
)
RELEASE_JSON_RELATIVE_PATH = Path(
    "data/logic/conformance/logic_family_parser_v2_release.json"
)
RELEASE_MARKDOWN_PATH = (
    REPO_ROOT / "ipfs_datasets_py" / RELEASE_MARKDOWN_RELATIVE_PATH
)
RELEASE_JSON_PATH = REPO_ROOT / "ipfs_datasets_py" / RELEASE_JSON_RELATIVE_PATH

PREDECESSOR_ACCELERATOR_COMMIT = "e162c19d087d4e6511f8eb97fd34ecb449777897"
PREDECESSOR_DATASETS_COMMIT = "fc49cbb3e0e96bf07b367859da32123187d706c1"
PREDECESSOR_SEED_DEFINITION = (
    "sha256:f5d01bcc13c0b62d35b713cccb2e04abe49da454e9fa6f35cd28a5ad4b72eb44"
)
PREDECESSOR_RELEASE_SHA256 = (
    "sha256:86412a60bfde9b8a13156ab097b44443a4a8f70a7b286f1c7a707366c93757ce"
)
PREDECESSOR_FILE_DIGESTS = {
    "docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_PLAN.md": (
        "sha256:9d07ef064e80081a67d13f754fff10b84b6176facf687b88ed1164d71a90e9c0"
    ),
    "docs/architecture/ipfs_datasets_logic_family_parser.objectives.md": (
        "sha256:1bc111b24e44508d56f4932da4ce0a76357eaaf01bf5ea22842cf06621b24217"
    ),
    "docs/architecture/ipfs_datasets_logic_family_parser.todo.md": (
        "sha256:8e851a11e3fbd1a0b174e2077abaa398c15fecdf9b9bb8baf9592b3311f5aaa8"
    ),
    "ipfs_datasets_py/data/logic/conformance/logic_family_parser_release.json": (
        PREDECESSOR_RELEASE_SHA256
    ),
}
PREDECESSOR_RUNTIME_ARTIFACT_DIGESTS = {
    "data/agent_supervisor/ipfs_datasets_logic_family_parser/refill/fixed_point_receipt.json": (
        "sha256:df389198f2f1a5982ede95ce775c468ad7a85abf8447f4d0cc51f8b6f5eddc2c"
    ),
    "data/agent_supervisor/ipfs_datasets_logic_family_parser/refill/gap_ledger.jsonl": (
        "sha256:6258dc0a9070fd531b77f96d1044f840454d02517022aa1c9e0f3e7b8debbcac"
    ),
}

# Filled after the 51 seed cards are materialized. Only Status values are
# normalized, so implementation progress cannot mutate semantic task identity.
SEALED_SEED_DEFINITION_SHA256 = (
    "sha256:ac4a347a84f049d8d64d43a004544be62e37490a190c78ac82c44cdcbc347e8c"
)

EXPECTED_TASK_GROUPS: Mapping[str, tuple[str, ...]] = {
    "LFP2-G010": tuple(f"LFP2-{index:03d}" for index in range(1, 5)),
    "LFP2-G020": tuple(f"LFP2-{index:03d}" for index in range(5, 10)),
    "LFP2-G030": tuple(f"LFP2-{index:03d}" for index in range(10, 16)),
    "LFP2-G040": tuple(f"LFP2-{index:03d}" for index in range(16, 22)),
    "LFP2-G050": tuple(f"LFP2-{index:03d}" for index in range(22, 28)),
    "LFP2-G060": tuple(f"LFP2-{index:03d}" for index in range(28, 37)),
    "LFP2-G070": tuple(f"LFP2-{index:03d}" for index in range(37, 44)),
    "LFP2-G080": tuple(f"LFP2-{index:03d}" for index in range(44, 48)),
    "LFP2-G090": ("LFP2-048", "LFP2-049"),
    "LFP2-G100": ("LFP2-050",),
}
EXPECTED_TASK_TO_GOAL = {
    "LFP2-000": "LFP2-G000",
    **{
        task_id: goal_id
        for goal_id, task_ids in EXPECTED_TASK_GROUPS.items()
        for task_id in task_ids
    },
}
REQUIRED_INTERFACE_OWNERS: Mapping[str, str] = {
    "ParseArtifact@2": "LFP2-006",
    "ElaborationArtifact@2": "LFP2-006",
    "FormalizationArtifact@3": "LFP2-007",
    "DomainLogicSlice@2": "LFP2-007",
    "ProviderExecutionReceipt@2": "LFP2-008",
    "EvidenceReplayReceipt@1": "LFP2-008",
    "ProtocolTargetTranslationEdges@1": "LFP2-021",
    "LogicFamilyRegistry@3": "LFP2-044",
    "LogicProfileCatalog@3": "LFP2-044",
    "FamilyRoutePublication@1": "LFP2-044",
    "ExecutableVerticalSliceReceipt@1": "LFP2-046",
}
DETERMINISTIC_MATERIALIZERS: Mapping[str, str] = {
    "LFP2-049": "ipfs_datasets_py.logic.conformance.fixed_point_v2",
    "LFP2-050": "ipfs_datasets_py.logic.conformance.release_v2",
}

REQUIRED_TASK_FIELDS = (
    "status",
    "completion",
    "is schedulable",
    "review only",
    "priority",
    "track",
    "depends on",
    "goal id",
    "outputs",
    "validation",
    "board namespace",
    "bundle",
    "parallel lane",
    "resource class",
    "resource stage",
    "estimated tokens",
    "implementation timeout seconds",
    "predicted files",
    "interfaces",
    "allow concurrent with",
    "conflict policy",
    "preconditions",
    "effects",
    "evidence subset",
    "symbolic first",
    "llm context budget bytes",
    "acceptance",
    "embedding query",
)
REQUIRED_GOAL_FIELDS = (
    "status",
    "review_only",
    "parent",
    "depends_on",
    "fib_priority",
    "track",
    "priority",
    "bundle",
    "parallel_lane",
    "resource_class",
    "goal",
    "seed_tasks",
    "evidence",
    "evidence_criteria",
    "evidence_source_policy",
    "outputs",
    "predicted_files",
    "interfaces",
    "validation",
    "acceptance",
    "gap_task",
    "refinement",
    "embedding_query",
    "ast_query",
    "conflict_policy",
)
REQUIRED_PLAN_TERMS = (
    "syntax_core",
    "security_ir",
    "crypto_ir",
    "intent_ir",
    "legal_ir",
    "ui_ux_ir",
    "z3",
    "cvc5",
    "tla_tlc",
    "apalache",
    "datalog_secpal",
    "proverif",
    "tamarin",
    "hyperltl_autohyper_mchyper",
    "vampire",
    "eprover",
    "hammer",
    "lean",
    "rocq",
    "isabelle",
    "ergoai",
    "symbolicai",
    "runtime_mtl",
    "description-logic",
    "argumentation",
    "mu-calculus",
    "finite-field",
    "session",
    "refill",
)
EXPECTED_PROVIDER = {
    "primary_provider_id": "grok_cli",
    "primary_model_id": "grok-4.5",
    "fallback_provider_id": "codex",
    "fallback_model_id": "gpt-5.6-terra",
    "fallback_trigger": "primary_quota_exhausted",
    "fallback_reasoning_effort": "high",
    "max_concurrency": 4,
    "secrets_from_environment_only": True,
    "secrets_in_argv_prompts_logs_or_receipts": False,
}
EXPECTED_ENVIRONMENT = {
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "grok_cli",
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER": "codex",
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER": (
        "primary_quota_exhausted"
    ),
    "IPFS_ACCELERATE_AGENT_GROK_MODEL": "grok-4.5",
    "IPFS_ACCELERATE_AGENT_CODEX_MODEL": "gpt-5.6-terra",
    "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT": "high",
}
CONTROL_PATHS = frozenset(
    {
        ".gitignore",
        "docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_V2_PLAN.md",
        "docs/architecture/ipfs_datasets_logic_family_parser_v2.objectives.md",
        "docs/architecture/ipfs_datasets_logic_family_parser_v2.todo.md",
        "config/agent_supervisor_ipfs_datasets_logic_family_parser_v2_scheduler.json",
        "scripts/validate_ipfs_datasets_logic_family_parser_v2_board.py",
        *PREDECESSOR_FILE_DIGESTS,
    }
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _safe_relative(value: str) -> bool:
    if not value or "\x00" in value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def _git(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def _merge_target_worktree_from_porcelain(payload: str) -> Path:
    target_ref = f"refs/heads/{MERGE_TARGET_BRANCH}"
    matches: list[Path] = []
    for record in payload.split("\0\0"):
        fields = [field for field in record.split("\0") if field]
        values = {
            key: value
            for field in fields
            for key, separator, value in (field.partition(" "),)
            if separator
        }
        if values.get("branch") == target_ref and values.get("worktree"):
            matches.append(Path(values["worktree"]))
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one merge-target worktree for "
            f"{target_ref}; got {len(matches)}"
        )
    return matches[0]


def _canonical_main_worktree(repo_root: Path) -> Path:
    """Resolve the sole merge-target worktree through shared Git metadata.

    Ignored Wave-1 release anchors do not appear in linked implementation
    candidates.  They are read from the supervisor merge-target worktree, not
    from Git's configured primary worktree.  Every identity hop is verified so
    a missing, duplicate, or malformed anchor fails closed.
    """

    candidate = repo_root.resolve(strict=True)
    common_result = _git("rev-parse", "--git-common-dir", cwd=candidate)
    raw_common = common_result.stdout.strip()
    if common_result.returncode != 0 or not raw_common or "\n" in raw_common:
        raise RuntimeError("git rev-parse --git-common-dir failed")
    common_path = Path(raw_common)
    if not common_path.is_absolute():
        common_path = candidate / common_path
    try:
        common_dir = common_path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("Git common directory is missing") from exc
    if not common_dir.is_dir():
        raise RuntimeError("Git common directory is not a directory")

    worktrees = _git("worktree", "list", "--porcelain", "-z", cwd=candidate)
    if worktrees.returncode != 0:
        raise RuntimeError("git worktree list --porcelain failed")
    primary_path = _merge_target_worktree_from_porcelain(worktrees.stdout)
    if not primary_path.is_absolute():
        raise RuntimeError("merge-target worktree path is not absolute")

    try:
        primary = primary_path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("canonical main worktree is missing") from exc
    if not primary.is_dir():
        raise RuntimeError("canonical main worktree is not a directory")

    identity = _git(
        "rev-parse",
        "--show-toplevel",
        "--git-common-dir",
        "--git-path",
        "config",
        cwd=primary,
    )
    lines = [line.strip() for line in identity.stdout.splitlines() if line.strip()]
    if identity.returncode != 0 or len(lines) != 3:
        raise RuntimeError("canonical main worktree Git identity is unavailable")
    main_root = Path(lines[0])
    if not main_root.is_absolute():
        main_root = primary / main_root
    reported_common = Path(lines[1])
    if not reported_common.is_absolute():
        reported_common = primary / reported_common
    reported_config = Path(lines[2])
    if not reported_config.is_absolute():
        reported_config = primary / reported_config
    try:
        same_root = main_root.resolve(strict=True) == primary
        same_common = reported_common.resolve(strict=True) == common_dir
        same_config = reported_config.resolve(strict=True) == (
            common_dir / "config"
        ).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("canonical main worktree Git identity is missing") from exc
    if not same_root or not same_common or not same_config:
        raise RuntimeError("canonical main worktree does not share candidate Git identity")
    return primary


def _seed_text(text: str) -> str:
    start = text.find("## LFP2-000 ")
    if start < 0:
        return ""
    appended = re.search(r"(?m)^## LFP2-(?:05[1-9]|0[6-9][0-9]|[1-9][0-9]{2,}) ", text[start:])
    end = start + appended.start() if appended else len(text)
    seed = text[start:end].rstrip() + "\n"
    return re.sub(r"(?m)^- Status: .+$", "- Status: <normalized>", seed)


def _seed_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(_seed_text(text).encode("utf-8")).hexdigest()


def _task_blocks(text: str) -> Mapping[str, str]:
    matches = list(re.finditer(r"(?m)^## (LFP2-[0-9]{3,}) .+$", text))
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result[match.group(1)] = text[match.start():end]
    return result


def _validate_goals(text: str, scheduler: Mapping[str, object], errors: list[str]) -> None:
    goals = parse_goal_heap(text)
    by_id = {goal.goal_id: goal for goal in goals}
    if tuple(by_id) != GOAL_IDS:
        errors.append(f"goal IDs/order differ: {tuple(by_id)!r}")
    if len(by_id) != len(goals):
        errors.append("duplicate goal ID")
    for goal_id, goal in by_id.items():
        missing = [field for field in REQUIRED_GOAL_FIELDS if field not in goal.fields]
        if missing:
            errors.append(f"{goal_id} missing goal fields: {missing}")
            continue
        if re.search(r"\bLFP2-[0-9]{3}\b", goal.fields["evidence"]):
            errors.append(f"{goal_id} Evidence conflates task IDs with evidence")
        if goal_id == "LFP2-G000":
            expected_seed = ("LFP2-000",)
            expected_parent = ""
        else:
            expected_seed = EXPECTED_TASK_GROUPS.get(goal_id, ())
            expected_parent = "LFP2-G000"
        if _split_csv(goal.fields["seed_tasks"]) != expected_seed:
            errors.append(f"{goal_id} Seed tasks differ from sealed task group")
        if goal.fields["parent"].strip() != expected_parent:
            errors.append(f"{goal_id} parent differs from sealed hierarchy")
    if scheduler.get("task_groups") != {
        goal: list(tasks) for goal, tasks in EXPECTED_TASK_GROUPS.items()
    }:
        errors.append("scheduler task_groups differ from objective Seed tasks")


def _validate_tasks(text: str, errors: list[str]) -> dict[str, object]:
    tasks = parse_task_text(text, path=TODO_PATH, task_header_prefix="## LFP2-")
    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        errors.append("duplicate task ID")
    actual_ids = tuple(by_id)
    if actual_ids[: len(TASK_IDS)] != TASK_IDS:
        errors.append("seed task IDs/order are not exactly LFP2-000..LFP2-050")
    for offset, task_id in enumerate(actual_ids[len(TASK_IDS):], start=len(TASK_IDS)):
        if task_id != f"LFP2-{offset:03d}":
            errors.append(f"appended task ID discontinuity at {task_id}")
    blocks = _task_blocks(text)
    completed: set[str] = set()
    open_ids: set[str] = set()
    dependencies: dict[str, tuple[str, ...]] = {}
    output_sets: dict[str, set[str]] = {}
    interface_owners: dict[str, set[str]] = {}
    for position, task in enumerate(tasks):
        metadata = task.metadata
        missing = [field for field in REQUIRED_TASK_FIELDS if field not in metadata]
        if missing:
            errors.append(f"{task.task_id} missing task fields: {missing}")
            continue
        block = blocks.get(task.task_id, "")
        normalized_keys = re.findall(r"(?m)^- ([^:\n]+):", block)
        duplicates = sorted({key.lower() for key in normalized_keys if sum(1 for item in normalized_keys if item.lower() == key.lower()) > 1})
        if duplicates:
            errors.append(f"{task.task_id} duplicate metadata keys: {duplicates}")
        seed = position < len(TASK_IDS)
        allowed_states = {"todo", "completed"} if seed else {"todo", "completed", "blocked"}
        if task.status not in allowed_states:
            errors.append(f"{task.task_id} invalid status {task.status!r}")
        if task.status == "completed":
            completed.add(task.task_id)
        else:
            open_ids.add(task.task_id)
        if metadata["board namespace"] != BOARD_NAMESPACE:
            errors.append(f"{task.task_id} board namespace mismatch")
        if seed and EXPECTED_TASK_TO_GOAL.get(task.task_id) != metadata["goal id"]:
            errors.append(f"{task.task_id} goal mapping mismatch")
        if not seed:
            generated_by = metadata.get("generated by", "")
            if not generated_by.startswith("ipfs_accelerate_py.agent_supervisor."):
                errors.append(f"{task.task_id} appended card lacks trusted provenance")
        if metadata["completion"] != "manual":
            errors.append(f"{task.task_id} completion must be manual")
        materializer_module = DETERMINISTIC_MATERIALIZERS.get(task.task_id)
        if materializer_module is not None:
            if metadata.get("provider role") != "deterministic-only":
                errors.append(
                    f"{task.task_id} Provider role must be deterministic-only"
                )
            validation = metadata["validation"]
            materializer_command = f"python -m {materializer_module} materialize"
            validator_command = (
                "scripts/validate_ipfs_datasets_logic_family_parser_v2_board.py"
            )
            materializer_offset = validation.find(materializer_command)
            validator_offset = validation.find(validator_command)
            if (
                materializer_offset < 0
                or validator_offset < 0
                or materializer_offset >= validator_offset
            ):
                errors.append(
                    f"{task.task_id} Validation must run its deterministic "
                    "materializer before the board validator"
                )
        schedulable = metadata["is schedulable"].lower()
        if task.task_id == "LFP2-000":
            if schedulable != "false" or metadata["review only"].lower() != "true":
                errors.append("LFP2-000 must be non-schedulable review-only")
        elif seed and schedulable != "true":
            errors.append(f"{task.task_id} seed implementation task must be schedulable")
        if metadata["symbolic first"].lower() != "true":
            errors.append(f"{task.task_id} Symbolic first must be true")
        for interface in _split_csv(metadata["interfaces"]):
            interface_owners.setdefault(interface, set()).add(task.task_id)
        for field in ("estimated tokens", "implementation timeout seconds", "llm context budget bytes"):
            try:
                if int(metadata[field]) <= 0:
                    raise ValueError
            except ValueError:
                errors.append(f"{task.task_id} {field} must be a positive integer")
        deps = tuple(task.depends_on)
        dependencies[task.task_id] = deps
        for dependency in deps:
            if dependency not in by_id:
                errors.append(f"{task.task_id} has unknown dependency {dependency}")
            elif actual_ids.index(dependency) >= position:
                errors.append(f"{task.task_id} dependency {dependency} is not earlier")
        outputs = set(task.outputs)
        predicted = set(_split_csv(metadata["predicted files"]))
        output_sets[task.task_id] = outputs
        if outputs != predicted:
            errors.append(f"{task.task_id} Outputs/Predicted files differ")
        for output in outputs:
            if not _safe_relative(output):
                errors.append(f"{task.task_id} has unsafe output {output!r}")
            if task.task_id != "LFP2-000" and output in CONTROL_PATHS:
                errors.append(f"{task.task_id} owns protected control output {output}")
            allowed_output = output.startswith("ipfs_datasets_py/") or (
                task.task_id == "LFP2-049" and output.startswith(f"{RUNTIME_ROOT}/refill/")
            ) or task.task_id == "LFP2-000"
            if not allowed_output:
                errors.append(f"{task.task_id} output is outside admitted owner roots: {output}")
    for interface, expected_owner in REQUIRED_INTERFACE_OWNERS.items():
        actual_owners = sorted(interface_owners.get(interface, set()))
        if actual_owners != [expected_owner]:
            errors.append(
                f"{interface} must be owned exactly by {expected_owner}; "
                f"got {actual_owners}"
            )
    for task_id in completed:
        missing_completed = set(dependencies.get(task_id, ())) - completed
        if missing_completed:
            errors.append(f"{task_id} completed before dependencies {sorted(missing_completed)}")
    ancestors: set[str] = set()
    stack = list(dependencies.get(TERMINAL_TASK, ()))
    while stack:
        item = stack.pop()
        if item in ancestors:
            continue
        ancestors.add(item)
        stack.extend(dependencies.get(item, ()))
    if set(TASK_IDS[:-1]) - ancestors:
        errors.append(f"terminal task does not cover: {sorted(set(TASK_IDS[:-1]) - ancestors)}")
    ready = tuple(
        task_id
        for task_id in actual_ids
        if task_id in open_ids and set(dependencies.get(task_id, ())).issubset(completed)
    )
    if completed == set(INITIAL_COMPLETED) and ready != INITIAL_READY:
        errors.append(f"initial ready set differs: expected {INITIAL_READY}, got {ready}")
    for left_index, left in enumerate(INITIAL_READY):
        for right in INITIAL_READY[left_index + 1:]:
            overlap = output_sets.get(left, set()) & output_sets.get(right, set())
            if overlap:
                errors.append(f"initial tasks {left}/{right} overlap outputs: {sorted(overlap)}")
    return {
        "task_count": len(tasks),
        "completed_task_ids": sorted(completed),
        "ready_task_ids": list(ready),
        "open_task_ids": sorted(open_ids),
        "refill_task_count": max(0, len(tasks) - len(TASK_IDS)),
    }


def _open_task_ids(tasks: Sequence[object]) -> set[str]:
    return {
        str(getattr(task, "task_id", ""))
        for task in tasks
        if str(getattr(task, "status", "")).lower() != "completed"
    }


def _invoke_artifact_validator(
    *,
    label: str,
    module_name: str,
    function_name: str,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    errors: list[str],
) -> bool:
    try:
        module = importlib.import_module(module_name)
        validator = getattr(module, function_name)
        if not callable(validator):
            raise TypeError(f"{function_name} is not callable")
    except Exception as exc:
        errors.append(
            f"{label} validator is unavailable: {type(exc).__name__}: {exc}"
        )
        return False
    try:
        result = validator(*args, **dict(kwargs))
    except Exception as exc:
        errors.append(f"{label} validation failed: {type(exc).__name__}: {exc}")
        return False
    if not isinstance(result, Mapping) or result.get("valid") is False:
        errors.append(f"{label} validator did not return a validated receipt")
        return False
    return True


def _validate_fixed_point_artifacts(
    tasks: Sequence[object], errors: list[str]
) -> bool:
    status_by_id = {
        str(getattr(task, "task_id", "")): str(getattr(task, "status", "")).lower()
        for task in tasks
    }
    fixed_exists = FIXED_POINT_PATH.is_file()
    ledger_exists = GAP_LEDGER_PATH.is_file()
    task_completed = status_by_id.get("LFP2-049") == "completed"
    if not fixed_exists and not ledger_exists and not task_completed:
        return False

    error_count = len(errors)
    if fixed_exists != ledger_exists:
        errors.append(
            "LFP2-049 fixed-point receipt and gap ledger must both exist or neither exist"
        )
    if task_completed and not (fixed_exists and ledger_exists):
        errors.append("LFP2-049 is completed without both fixed-point artifacts")

    allowed_open = {"LFP2-050"} if task_completed else {"LFP2-049", "LFP2-050"}
    unexpected_open = sorted(_open_task_ids(tasks) - allowed_open)
    if unexpected_open:
        errors.append(
            "LFP2-049 fixed-point validation requires every predecessor and "
            f"derived task to be terminal; open: {unexpected_open}"
        )

    api_valid = _invoke_artifact_validator(
        label="LFP2-049 fixed-point artifacts",
        module_name="ipfs_datasets_py.logic.conformance.fixed_point_v2",
        function_name="validate_fixed_point_artifacts",
        args=(FIXED_POINT_PATH, GAP_LEDGER_PATH),
        kwargs={"repo_root": REPO_ROOT, "tasks": tasks},
        errors=errors,
    )
    return api_valid and len(errors) == error_count


def _validate_release_artifacts(
    tasks: Sequence[object], *, fixed_point_valid: bool, errors: list[str]
) -> None:
    status_by_id = {
        str(getattr(task, "task_id", "")): str(getattr(task, "status", "")).lower()
        for task in tasks
    }
    markdown_exists = RELEASE_MARKDOWN_PATH.is_file()
    json_exists = RELEASE_JSON_PATH.is_file()
    task_completed = status_by_id.get(TERMINAL_TASK) == "completed"
    if not markdown_exists and not json_exists and not task_completed:
        return

    if markdown_exists != json_exists:
        errors.append(
            "LFP2-050 release Markdown and JSON artifacts must both exist or neither exist"
        )
    if task_completed and not (markdown_exists and json_exists):
        errors.append("LFP2-050 is completed without both release artifacts")

    # LFP2-050 is the candidate being validated, so it is deliberately excluded
    # from the terminal prerequisite while its artifacts exist but its status is
    # still todo.  No other seed or appended task may remain open.
    unexpected_open = sorted(_open_task_ids(tasks) - {TERMINAL_TASK})
    if unexpected_open:
        errors.append(
            "LFP2-050 release validation requires every other seed and derived "
            f"task to be terminal; open: {unexpected_open}"
        )
    if status_by_id.get("LFP2-049") != "completed":
        errors.append("LFP2-050 release validation requires completed LFP2-049")
    if not fixed_point_valid:
        errors.append(
            "LFP2-050 release validation requires a current LFP2-049 fixed-point receipt"
        )

    _invoke_artifact_validator(
        label="LFP2-050 release artifacts",
        module_name="ipfs_datasets_py.logic.conformance.release_v2",
        function_name="validate_release_artifacts",
        args=(RELEASE_MARKDOWN_RELATIVE_PATH, RELEASE_JSON_RELATIVE_PATH),
        kwargs={"repo_root": REPO_ROOT},
        errors=errors,
    )


def _validate_completion_artifacts(text: str, errors: list[str]) -> None:
    tasks = parse_task_text(text, path=TODO_PATH, task_header_prefix="## LFP2-")
    fixed_point_valid = _validate_fixed_point_artifacts(tasks, errors)
    _validate_release_artifacts(
        tasks,
        fixed_point_valid=fixed_point_valid,
        errors=errors,
    )


def _validate_plan(text: str, errors: list[str]) -> None:
    lowered = text.lower().replace("-", "_")
    for term in REQUIRED_PLAN_TERMS:
        normalized = term.lower().replace("-", "_")
        if normalized not in lowered:
            errors.append(f"plan missing required term: {term}")


def _validate_predecessor_artifacts(
    scheduler: Mapping[str, object], errors: list[str]
) -> None:
    for relative, expected in PREDECESSOR_FILE_DIGESTS.items():
        path = REPO_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            errors.append(f"Wave-1 predecessor artifact changed: {relative}")
    if (
        scheduler.get("predecessor_runtime_artifact_digests")
        != PREDECESSOR_RUNTIME_ARTIFACT_DIGESTS
    ):
        errors.append(
            "scheduler predecessor_runtime_artifact_digests differs from release seal"
        )
    try:
        runtime_root = _canonical_main_worktree(REPO_ROOT)
    except (OSError, RuntimeError) as exc:
        errors.append(
            "Wave-1 predecessor canonical runtime root is unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
        return
    for relative, expected in PREDECESSOR_RUNTIME_ARTIFACT_DIGESTS.items():
        path = runtime_root / relative
        if not path.is_file() or _sha256(path) != expected:
            errors.append(
                f"Wave-1 predecessor runtime artifact changed: {relative}"
            )


def _validate_predecessor(scheduler: Mapping[str, object], errors: list[str]) -> None:
    _validate_predecessor_artifacts(scheduler, errors)
    expected_binding = {
        "predecessor_board_namespace": "ipfs-datasets-logic-family-parser-v1",
        "predecessor_terminal_task_id": "LFP-047",
        "predecessor_accelerator_commit": PREDECESSOR_ACCELERATOR_COMMIT,
        "predecessor_datasets_commit": PREDECESSOR_DATASETS_COMMIT,
        "predecessor_seed_definition_sha256": PREDECESSOR_SEED_DEFINITION,
        "predecessor_release_receipt_path": "ipfs_datasets_py/data/logic/conformance/logic_family_parser_release.json",
        "predecessor_release_receipt_sha256": PREDECESSOR_RELEASE_SHA256,
    }
    if scheduler.get("predecessor_binding") != expected_binding:
        errors.append("scheduler predecessor_binding differs from release seal")
    if _git("merge-base", "--is-ancestor", PREDECESSOR_ACCELERATOR_COMMIT, "HEAD").returncode != 0:
        errors.append("Wave-1 accelerator release is not an ancestor of HEAD")
    nested = REPO_ROOT / "ipfs_datasets_py"
    if _git("merge-base", "--is-ancestor", PREDECESSOR_DATASETS_COMMIT, "HEAD", cwd=nested).returncode != 0:
        errors.append("Wave-1 datasets release is not an ancestor of nested HEAD")


def _common_args(plan: Mapping[str, object]) -> list[str]:
    prefix = "--common-arg="
    return [
        item[len(prefix):]
        for item in plan.get("argv", [])
        if isinstance(item, str) and item.startswith(prefix)
    ]


def _validate_scheduler(scheduler: Mapping[str, object], errors: list[str]) -> None:
    expected_projection = {
        "task_count": 51,
        "completed_task_ids": list(INITIAL_COMPLETED),
        "ready_task_ids": list(INITIAL_READY),
        "blocked_task_ids": [],
        "terminal_task_id": TERMINAL_TASK,
        "goal_count": 11,
        "root_goal_id": "LFP2-G000",
    }
    if scheduler.get("initial_projection") != expected_projection:
        errors.append("scheduler initial_projection differs from launch seal")
    if scheduler.get("provider") != EXPECTED_PROVIDER:
        errors.append("scheduler provider route differs from Grok/quota-only Terra-high seal")
    if scheduler.get("merge_target_branch") != MERGE_TARGET_BRANCH:
        errors.append("scheduler merge target differs")
    if scheduler.get("board_namespace") != BOARD_NAMESPACE:
        errors.append("scheduler board namespace differs")
    if scheduler.get("task_prefix") != "LFP2-" or scheduler.get("goal_prefix") != "LFP2-G":
        errors.append("scheduler task/goal prefix differs")
    protected_paths = scheduler.get("protected_paths")
    if not isinstance(protected_paths, list):
        errors.append("scheduler protected_paths must be a list")
    else:
        missing_protected = sorted(CONTROL_PATHS - set(protected_paths))
        if missing_protected:
            errors.append(
                f"scheduler protected_paths missing controls: {missing_protected}"
            )
        if len(protected_paths) != len(set(protected_paths)):
            errors.append("scheduler protected_paths contain duplicates")
    if scheduler.get("max_lanes") != 4 or scheduler.get("strict_task_sharding") is not False:
        errors.append("scheduler must use four dynamic work-stealing lanes")
    if scheduler.get("objective_refill_enabled") is not True or scheduler.get("codebase_refill_enabled") is not False:
        errors.append("scheduler refill mode differs")
    if scheduler.get("objective_goal_refinement_enabled") is not False:
        errors.append("static objective heap must disable goal refinement")
    runtime_paths = scheduler.get("runtime_paths")
    if not isinstance(runtime_paths, Mapping) or runtime_paths.get("root") != RUNTIME_ROOT:
        errors.append("scheduler v2 runtime root differs")
    if isinstance(runtime_paths, Mapping) and any(
        str(value).startswith("data/agent_supervisor/ipfs_datasets_logic_family_parser/")
        for value in runtime_paths.values()
    ):
        errors.append("v2 runtime overlaps Wave-1 runtime")
    source = scheduler.get("source_binding")
    if not isinstance(source, Mapping) or source.get("accelerator_required_ancestor") != PREDECESSOR_ACCELERATOR_COMMIT or source.get("accelerator_required_branch") != MERGE_TARGET_BRANCH or source.get("ipfs_datasets_planning_revision") != PREDECESSOR_DATASETS_COMMIT:
        errors.append("scheduler source binding differs from v2 predecessor seal")
    refill = scheduler.get("refill_policy")
    derived = refill.get("derived_refill") if isinstance(refill, Mapping) else None
    expected_refill = {
        "max_goals_per_epoch": 8,
        "max_tasks_per_epoch": 24,
        "min_open_tasks": 8,
        "max_open_tasks": 48,
        "max_refinement_depth": 3,
        "max_unchanged_failure_retries": 2,
        "cooldown_seconds": 3600,
        "mutate_seed_board": False,
        "mutate_seed_objectives": False,
    }
    if derived != expected_refill:
        errors.append("scheduler derived refill policy differs")
    try:
        board = load_configured_board(SCHEDULER_PATH, repo_root=REPO_ROOT)
        plan = configured_board_launch_plan(
            board,
            implement=True,
            detach=True,
            duration_seconds=300,
            stamp="20260809T000000Z",
        )
    except (ConfiguredBoardError, OSError, ValueError) as exc:
        errors.append(f"scheduler loader/renderer rejected config: {type(exc).__name__}: {exc}")
        return
    if plan.get("environment") != EXPECTED_ENVIRONMENT:
        errors.append("rendered provider environment differs from quota-only route")
    common = _common_args(plan)
    for flag in (
        "--objective-refill-scan",
        "--no-objective-goal-refinement",
        "--no-objective-goal-completion-reconcile",
        "--no-objective-goal-migration",
        "--no-objective-task-janitor",
    ):
        if common.count(flag) != 1:
            errors.append(f"rendered launch must contain exactly one {flag}")
    for forbidden in ("--strict-task-sharding", "--codebase-refill-scan"):
        if forbidden in common:
            errors.append(f"rendered launch unexpectedly contains {forbidden}")


def validate_all() -> dict[str, object]:
    errors: list[str] = []
    for path in (PLAN_PATH, OBJECTIVE_PATH, TODO_PATH, SCHEDULER_PATH):
        if not path.is_file():
            errors.append(f"missing control file: {path.relative_to(REPO_ROOT)}")
    if errors:
        return {"valid": False, "errors": errors}
    try:
        scheduler = json.loads(SCHEDULER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"scheduler unreadable: {exc}"]}
    plan_text = PLAN_PATH.read_text(encoding="utf-8")
    objective_text = OBJECTIVE_PATH.read_text(encoding="utf-8")
    todo_text = TODO_PATH.read_text(encoding="utf-8")
    seed_digest = _seed_digest(todo_text)
    if SEALED_SEED_DEFINITION_SHA256 != "TO_BE_FILLED" and seed_digest != SEALED_SEED_DEFINITION_SHA256:
        errors.append("Wave-2 seed task definition differs from sealed digest")
    _validate_plan(plan_text, errors)
    _validate_predecessor(scheduler, errors)
    _validate_goals(objective_text, scheduler, errors)
    task_report = _validate_tasks(todo_text, errors)
    _validate_completion_artifacts(todo_text, errors)
    _validate_scheduler(scheduler, errors)
    return {
        "schema": "ipfs_accelerate_py/ipfs-datasets-logic-family-parser-v2-preflight@1",
        "valid": not errors,
        "errors": errors,
        "board_namespace": BOARD_NAMESPACE,
        "plan_path": str(PLAN_PATH),
        "objective_path": str(OBJECTIVE_PATH),
        "todo_path": str(TODO_PATH),
        "scheduler_path": str(SCHEDULER_PATH),
        "plan_sha256": _sha256(PLAN_PATH),
        "objective_sha256": _sha256(OBJECTIVE_PATH),
        "todo_sha256": _sha256(TODO_PATH),
        "seed_definition_sha256": seed_digest,
        "seed_task_count": len(TASK_IDS),
        "goal_count": len(GOAL_IDS),
        "terminal_task_id": TERMINAL_TASK,
        "root_goal_ids": ["LFP2-G000"],
        "predecessor_accelerator_commit": PREDECESSOR_ACCELERATOR_COMMIT,
        "predecessor_datasets_commit": PREDECESSOR_DATASETS_COMMIT,
        **task_report,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-all", action="store_true")
    parser.parse_args(argv)
    report = validate_all()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
