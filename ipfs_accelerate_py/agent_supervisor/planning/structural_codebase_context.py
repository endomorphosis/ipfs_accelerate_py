"""Fresh structural codebase context at the existing planning-material seam.

This adapter observes a datasets-owned head and current source before and after
local planning. It creates no observed behavioral facts and does not activate
PlanCreateService, reconcile its authority-root schemas, admit a task or grant
execution/completion authority. Callers must finish the context successfully
before publishing a derived result. Every later use needs a new observation.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar

from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured, validate_cid

if TYPE_CHECKING:
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex

SCHEMA = "supervisor-structural-codebase-context@1"
_HEAD_FIELDS = ("repository_id", "generation", "manifest_cid", "snapshot_cid",
                "ast_revision_id", "receipt_cid")
_COVERAGE_FIELDS = frozenset({
    "inventory_entries", "captured_entries", "ast_ok", "ast_partial", "ast_failed",
    "opaque_entries", "unindexed_entries", "semantic_symbols", "formalized_properties",
    "checked_properties",
})
_Result = TypeVar("_Result")


class StructuralCodebaseContextError(ValueError):
    """An owner/source observation cannot supply the requested context."""


@dataclass(frozen=True, slots=True)
class StructuralCodebaseContext:
    """Immutable, body-free structural metadata, with no discharge authority."""

    head: CodebaseHead
    semantic_state_cid: str
    coverage: Mapping[str, int]

    def __post_init__(self) -> None:
        from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead

        if type(self.head) is not CodebaseHead:
            raise StructuralCodebaseContextError("canonical datasets CodebaseHead required")
        validate_cid(self.semantic_state_cid, codecs={"dag-json"})
        if (not isinstance(self.coverage, Mapping) or set(self.coverage) != _COVERAGE_FIELDS
                or any(type(key) is not str or type(value) is not int or value < 0
                       for key, value in self.coverage.items())):
            raise StructuralCodebaseContextError("coverage requires the exact structural integer counts")
        if self.coverage["formalized_properties"] or self.coverage["checked_properties"]:
            raise StructuralCodebaseContextError("structural context cannot claim formalized or checked properties")
        object.__setattr__(self, "coverage", MappingProxyType(dict(sorted(self.coverage.items()))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "head": {name: getattr(self.head, name) for name in _HEAD_FIELDS},
            "semantic_state_cid": self.semantic_state_cid,
            "coverage": dict(self.coverage),
            "authority": "structural_only",
            "source_semantics_verified": False,
            "proof_authority": False,
            "execution_authority": False,
            "completion_authority": False,
        }

    @property
    def cid(self) -> str:
        return cid_for_structured(self.to_dict())

    def to_plan_create_materials(self):
        """Bind metadata using existing material identity; no static root claim.

        This is an integration seam, not an admission-ready plan. The caller
        still needs independently bound intent/operations and live authority
        roots. ``current_roots=None`` deliberately leaves a real service root
        observer usable instead of shadowing it with a frozen observation.
        """
        from ..prompt.plan_create_service import PlanCreateMaterials

        record = self.to_dict()
        return PlanCreateMaterials(
            scan={"scan_cid": self.cid, "structural_codebase": record},
            extra={"structural_codebase_context_cid": self.cid,
                   "structural_codebase": self.to_dict()},
            current_facts=(),
            current_roots=None,
        )


def _from_observation(observation, expected_head) -> StructuralCodebaseContext:
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import (
        CodebaseIRManifest, CodebaseObservation,
    )

    if type(observation) is not CodebaseObservation or type(observation.manifest) is not CodebaseIRManifest:
        raise StructuralCodebaseContextError("canonical datasets codebase observation required")
    manifest, head = observation.manifest, observation.head
    if (head != expected_head or head.repository_id != manifest.snapshot.repository_id
            or head.manifest_cid != manifest.cid
            or head.snapshot_cid != manifest.snapshot.snapshot_cid
            or head.ast_revision_id != manifest.ast_revision_id):
        raise StructuralCodebaseContextError("observation source/head bindings differ")
    return StructuralCodebaseContext(head, manifest.semantic_state.state_cid, manifest.coverage)


@contextmanager
def structural_codebase_context(
    index: RepositoryCodebaseIndex,
    repository: str | Path,
    *,
    repository_id: str,
    expected_head: CodebaseHead | None = None,
    scheduler=None,
    parent_lease=None,
    cancel_event=None,
    admission_timeout_seconds: float = 30.0,
    timeout_seconds: float = 120.0,
    memory_mb: int = 512,
) -> Iterator[StructuralCodebaseContext]:
    """Observe live source on entry and successful exit under one deadline.

    Repository locators and resource controls stay ephemeral. Preparation,
    training and inference are never called here; observation only checks the
    existing head and source. A failed exit invalidates the enclosing call's
    result, so no derived result may be published inside the context. The
    deadline is cooperative; local planning needs its own execution admission.
    """
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        LeaseCancelledError, LeaseTimeoutError,
    )

    if (type(repository_id) is not str or not repository_id.strip()
            or repository_id != repository_id.strip() or Path(repository_id).is_absolute()):
        raise StructuralCodebaseContextError("explicit repository view identity required, not a locator")
    for name, value, positive in (("timeout_seconds", timeout_seconds, True),
                                  ("admission_timeout_seconds", admission_timeout_seconds, False)):
        if (type(value) not in {int, float} or not math.isfinite(value)
                or value < 0 or (positive and value == 0)):
            raise StructuralCodebaseContextError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    if expected_head is not None and type(expected_head) is not CodebaseHead:
        raise StructuralCodebaseContextError("expected_head must be a datasets CodebaseHead")
    if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
        raise StructuralCodebaseContextError("cancel_event must provide is_set()")
    deadline = time.monotonic() + timeout_seconds

    def remaining() -> float:
        if cancel_event is not None and cancel_event.is_set():
            raise LeaseCancelledError("structural planning observation cancelled")
        duration = deadline - time.monotonic()
        if duration <= 0:
            raise LeaseTimeoutError("structural planning observation deadline exceeded")
        return duration

    remaining()
    head = expected_head if expected_head is not None else index.current(repository_id)
    if type(head) is not CodebaseHead or head.repository_id != repository_id:
        raise StructuralCodebaseContextError("no current structural head for this repository view")

    def observe() -> StructuralCodebaseContext:
        duration = remaining()
        observation = index.observe_current(
            repository, expected_head=head, scheduler=scheduler, parent_lease=parent_lease,
            cancel_event=cancel_event, admission_timeout_seconds=min(admission_timeout_seconds, duration),
            timeout_seconds=duration, memory_mb=memory_mb,
        )
        remaining()
        return _from_observation(observation, head)

    context = observe()
    yield context
    if observe() != context:
        raise StructuralCodebaseContextError("structural planning context changed during use")


def run_with_structural_codebase_context(
    index: RepositoryCodebaseIndex,
    repository: str | Path,
    build: Callable[[StructuralCodebaseContext], _Result],
    **controls,
) -> _Result:
    """Return a local build result only after the completion observation passes.

    ``build`` must not publish/admit/dispatch its result. Exceptions and source
    or head drift propagate without returning the candidate result.
    """
    if not callable(build):
        raise TypeError("build must be callable")
    with structural_codebase_context(index, repository, **controls) as context:
        result = build(context)
    return result


__all__ = ["SCHEMA", "StructuralCodebaseContext", "StructuralCodebaseContextError",
           "structural_codebase_context", "run_with_structural_codebase_context"]
