"""Compose source and live-intent evidence for one existing supervisor task.

Preparation produces task metadata nominations, not canonical task mutations,
plan admission, scheduling, or completion authority. Source and owner checks
detect drift during preparation; dispatch still revalidates both artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
from ..semantic_state.intent_world_snapshot import (
    capture_intent_world_snapshot,
    load_intent_world_context,
    persist_intent_world_snapshot,
)
from ..task_sources.intent_repository import IntentRepository
from .semantic_context_runtime import load_semantic_worker_context, prepare_semantic_context


def prepare_supervised_task_context(
    *,
    repository: Path,
    intent: IntentRepository,
    task_cid: str,
    paths: Sequence[str],
    required_raw_paths: Sequence[str],
    output: Path,
    coordinator=None,
    code_vector_snapshot=None,
    code_vector_result=None,
    code_query_text: str = "",
    semantic_max_symbols: int = 256,
    semantic_context_input_tokens: int = 8192,
    semantic_worker_query: str = "",
    semantic_worker_max_bytes: int = 32768,
    security_source_program_config=None,
    intent_code_effect_instruction=None,
    intent_code_effect_intent_advice=None,
    intent_code_effect_config=None,
) -> dict:
    """Join native task, capsules, Doctor and world evidence from one scope.

    This operation reads a file-backed native intent owner. A caller-owned
    database transaction must use the lower-level capture API with explicit
    transaction ownership instead of this preparation convenience boundary.
    """
    if not isinstance(intent, IntentRepository):
        raise TypeError("intent must be an IntentRepository")
    if intent.uses_bound_connection:
        raise ValueError("task context preparation requires a file-backed intent owner")
    root = Path(repository).resolve(strict=True)
    retrieval_requested = any(value is not None for value in (code_vector_snapshot, code_vector_result)) or bool(code_query_text)
    if retrieval_requested:
        from ..analysis.code_symbol_vector_index import CodeVectorIndexSnapshot, CodeVectorSearchResult

        if (not isinstance(code_vector_snapshot, CodeVectorIndexSnapshot)
                or not isinstance(code_vector_result, CodeVectorSearchResult)
                or not code_query_text
                or not set(code_vector_snapshot.included_paths).issubset(paths)):
            raise ValueError("retrieval requires native index/result and the same permitted source scope")
    output = Path(output).absolute()
    if (
        not output.is_relative_to(root)
        or output.resolve() != output
        or output.exists()
        or any((root / name).is_relative_to(output) for name in paths)
    ):
        raise ValueError(
            "task context output must be a new separate repository-contained directory"
        )
    watermark = intent.event_watermark()
    plan = intent.plan_projection(task_cids=[task_cid])
    tasks = [row for row in plan["tasks"] if row["task_cid"] == task_cid]
    if len(tasks) != 1 or watermark != intent.event_watermark():
        raise ValueError("task intent changed before preparation")
    task = tasks[0]
    alias = task["task_alias"]
    body = task["body"]
    title = body.get("title") or body.get("objective") or body.get("description")
    if (
        not isinstance(alias, str)
        or not alias.strip()
        or not isinstance(title, str)
        or not title.strip()
    ):
        raise ValueError("task context requires a native task alias and declared title/objective")
    semantic = prepare_semantic_context(
        repository=root,
        paths=paths,
        required_raw_paths=required_raw_paths,
        objective=title,
        task_id=alias,
        output=output / "semantic",
        max_symbols=semantic_max_symbols,
        context_input_tokens=semantic_context_input_tokens,
        worker_query=semantic_worker_query,
        worker_max_bytes=semantic_worker_max_bytes,
    )
    blocks = output / "semantic/blocks"

    def get_block(cid):
        # Producer CIDs are single path components; never accept a path from a
        # corrupt block graph even though the native reader also verifies CIDs.
        if not isinstance(cid, str) or Path(cid).name != cid or cid in {"", ".", ".."}:
            raise ValueError("invalid semantic block identity")
        return (blocks / cid).read_bytes()

    view = IpfsDatasetsSemanticStateProvider().open_verified_view(
        semantic["semantic_root_cid"], get_block
    )
    repository_id = view.root.repository_id
    capture = capture_intent_world_snapshot(
        intent,
        repository_id=repository_id,
        task_cids=[task_cid],
        semantic_root_cid=semantic["semantic_root_cid"],
        get_semantic_block=get_block,
        coordinator=coordinator,
    )
    if (
        capture["plan_projection"]["projection_cid"] != plan["projection_cid"]
        or capture["planning_context"]["event_watermark"] != watermark
    ):
        raise ValueError("task intent changed during context preparation")
    world = persist_intent_world_snapshot(capture, output=output / "world", task_id=alias)
    semantic_artifact = (output / "semantic/worker-context.json").relative_to(root).as_posix()
    world_artifact = (output / "world/intent-world.json").relative_to(root).as_posix()
    # Only publish nominations after both independently owned observations
    # still agree with their current source bytes and task revision.
    load_semantic_worker_context(
        repository=root,
        artifact=semantic_artifact,
        expected_sha256=semantic["worker_payload_sha256"],
        task_id=alias,
    )
    load_intent_world_context(
        artifact=root / world_artifact,
        expected_sha256=world["artifact_sha256"],
        task_id=alias,
        repository_id=repository_id,
        intent=intent,
    )
    if intent.event_watermark() != watermark:
        raise ValueError("task intent changed after context preparation")
    retrieval = None
    if retrieval_requested:
        from .code_retrieval_context import prepare_code_retrieval_context

        retrieval = prepare_code_retrieval_context(
            repository=root, task_id=alias, query_text=code_query_text,
            snapshot=code_vector_snapshot, result=code_vector_result,
            output=output / "code-retrieval.json",
        )
        load_semantic_worker_context(
            repository=root, artifact=semantic_artifact,
            expected_sha256=semantic["worker_payload_sha256"], task_id=alias,
        )
        if intent.event_watermark() != watermark:
            raise ValueError("task intent changed during retrieval preparation")
    result = {
        "schema": "supervisor-task-context-preparation@1",
        "task_cid": task_cid,
        "task_id": alias,
        "task_title": title,
        "task_revision": task["revision"],
        "plan_projection_cid": plan["projection_cid"],
        "event_watermark": watermark,
        "repository_id": repository_id,
        "semantic_root_cid": semantic["semantic_root_cid"],
        "world_snapshot_cid": capture["snapshot"]["snapshot_cid"],
        "metadata": {
            "Semantic context artifact": semantic_artifact,
            "Semantic context sha256": semantic["worker_payload_sha256"],
            "Semantic context refresh": "true",
            "World context artifact": world_artifact,
            "World context sha256": world["artifact_sha256"],
            "World context repository": repository_id,
            **(retrieval["metadata"] if retrieval is not None else {}),
        },
        "semantic": semantic,
        "world": world,
        "retrieval": retrieval,
        "execution_authority": False,
        "completion_authority": False,
        "canonical_task_mutated": False,
    }
    if security_source_program_config is not None:
        from .security_source_program_advisor_384 import prepare_repository_source_program_advice
        result["security_source_program_advice"] = prepare_repository_source_program_advice(
            repository=root, paths=list(paths), config=security_source_program_config,
            output=output / "security-source-program-advice.json",
        )
        # Optional inference may take time; keep native task/source evidence
        # bound to the same owner revision before returning any nominations.
        load_semantic_worker_context(repository=root, artifact=semantic_artifact,
            expected_sha256=semantic["worker_payload_sha256"], task_id=alias)
        if intent.event_watermark() != watermark:
            raise ValueError("task intent changed during optional Security inference")
    if intent_code_effect_config is not None:
        try:
            from .intent_code_effect_advisor import prepare_repository_intent_code_effect_advice
            effect_advice = prepare_repository_intent_code_effect_advice(
                repository=root, paths=list(paths), instruction=intent_code_effect_instruction,
                intent_advice=intent_code_effect_intent_advice,
                security_advice=result.get("security_source_program_advice"), config=intent_code_effect_config)
        except Exception as error:
            effect_advice = dict(status="fail_open_unavailable", continue_planning=True,
                error_type=type(error).__name__, proof_authority=False,
                execution_authority=False, completion_authority=False)
        result["intent_code_effect_advice"] = effect_advice
        # An optional contract never selects a prompt from the task title or
        # attaches its declarations to the native IntentRepository. Recheck the
        # independent current owners after this potentially expensive work.
        load_semantic_worker_context(repository=root, artifact=semantic_artifact,
            expected_sha256=semantic["worker_payload_sha256"], task_id=alias)
        if intent.event_watermark() != watermark:
            raise ValueError("task intent changed during optional Intent/code contract check")
    (output / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result
