"""Explicit source-session inputs for the task proposal consumer.

The caller owns the session and reviewed Intent contract. No automatic intent
translation, global metadata catalog, provider, or learned IR is created here.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..runtime.repository_context_compiler import RepositoryContextError
from ..runtime.repository_source_session import RepositorySourceSession


@dataclass(frozen=True)
class RepositoryPlanningContext:
    session: RepositorySourceSession = field(repr=False, compare=False)
    generation: int
    intent_contract: Any = field(repr=False, compare=False)
    query: str
    unresolved_question: str
    security_scope: Any = field(default=None, repr=False, compare=False)
    security_analyses: Any = field(default=None, repr=False, compare=False)
    query_limit: int = 8
    expansion_reference_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if (not isinstance(self.session, RepositorySourceSession)
                or type(self.generation) is not int or self.generation < 1
                or type(self.query_limit) is not int or not 1 <= self.query_limit <= 32
                or any(type(v) is not str or not v.strip() or len(v.encode()) > 1024
                       for v in (self.query, self.unresolved_question))
                or type(self.expansion_reference_ids) is not tuple
                or len(self.expansion_reference_ids) > 32
                or any(type(v) is not str or not v or len(v.encode()) > 1024
                       for v in self.expansion_reference_ids)
                or len(set(self.expansion_reference_ids)) != len(self.expansion_reference_ids)):
            raise RepositoryContextError("invalid_repository_planning_context")

    def validate_paths(self, config):
        root = self.session.workspace.resolve()
        if Path(config.repo_root).resolve() != root:
            raise RepositoryContextError("planning_workspace_mismatch")
        artifact = Path(config.artifact_dir).resolve()
        for owned in (root, self.session.state_dir.resolve()):
            if artifact.is_relative_to(owned) or owned.is_relative_to(artifact):
                raise RepositoryContextError("planning_artifacts_overlap_source_or_state")
        for path in (config.task_board_path, config.plan_path):
            if not Path(path).resolve().is_relative_to(root):
                raise RepositoryContextError("planning_input_outside_source")

    def read_text(self, path):
        try:
            relative = Path(path).resolve().relative_to(self.session.workspace.resolve()).as_posix()
        except ValueError as exc:
            raise RepositoryContextError("planning_input_outside_source") from exc
        result = self.session.read(generation=self.generation, path=relative, representation="original")
        if result.get("encoding") != "utf-8" or not isinstance(result.get("content"), str):
            raise RepositoryContextError("planning_input_not_utf8")
        return result["content"]

    def identity(self):
        status = self.session.status_summary()
        if status["source_current"] is not True or status["generation"] != self.generation:
            raise RepositoryContextError("planning_source_changed")
        return status["datasets_head"]["repository_id"], "sha256:" + status["source_tree_sha256"]

    def required_evidence(self):
        from .repository_planning_evidence import build_repository_planning_evidence
        return build_repository_planning_evidence(
            session=self.session, generation=self.generation, intent_contract=self.intent_contract,
            security_scope=self.security_scope, security_analyses=self.security_analyses)
