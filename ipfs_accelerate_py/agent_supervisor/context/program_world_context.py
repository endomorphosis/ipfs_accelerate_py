"""SAWM-022 program-world context planner on the one ContextCompiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessError,
    _bool,
    _text,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_receipts import (
    ContextItemDisposition,
    FreshnessState,
    ProgramWorldContextItem,
    ProgramWorldContextReceipt,
    ProgramWorldReceiptError,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload


ASSEMBLY_ORDER: Final[tuple[str, ...]] = (
    "goal",
    "state",
    "procedure",
    "questions",
    "slice",
    "contracts",
    "counterexamples",
    "tests",
    "proofs",
    "transitions",
    "analogues",
    "raw_fallback",
)
REQUIRED_KINDS: Final[frozenset[str]] = frozenset(
    {"tests", "proofs", "policy", "authority", "raw_source"}
)


class ProgramWorldContextError(HarnessError):
    """Closed program-world context contract violation."""


@dataclass(frozen=True, slots=True)
class ProgramWorldPrefixReuse:
    prefix_cid: str
    exact_bytes: bool
    reused: bool


def _mirror_program_world_context_compilation(result: Any) -> Any:
    """Record the context receipt id. Materials, questions, and the budget are not stored."""

    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        record_ref = str(getattr(result, "receipt_cid", "") or "program-world-context")
        mirror_work_record(
            catalog_kind="capsule",
            record_kind="program_world_context_compilation",
            record_ref=record_ref,
            subject_kind="record_cid",
            subject_ref=record_ref,
        )
    except Exception:
        pass
    return result


@dataclass
class ProgramWorldContextPlanner:
    token_budget: int
    _prefix_cache: dict[str, bytes] = field(default_factory=dict)

    def compile_program_world_context(
        self,
        request: Mapping[str, Any],
    ) -> ProgramWorldContextReceipt:
        if not isinstance(request, Mapping):
            raise ProgramWorldContextError("context request must be an object")
        budget = int(request.get("token_budget") or self.token_budget)
        if budget < 1:
            raise ProgramWorldContextError("token budget must be positive")
        unresolved = tuple(
            _text(item, "unresolved_question")
            for item in (request.get("unresolved_questions") or ())
        )
        materials = list(request.get("materials") or ())
        ordered = sorted(
            materials,
            key=lambda item: (
                ASSEMBLY_ORDER.index(str(item.get("kind")))
                if str(item.get("kind")) in ASSEMBLY_ORDER
                else len(ASSEMBLY_ORDER),
                str(item.get("identity_cid")),
            ),
        )
        included: list[ProgramWorldContextItem] = []
        omitted: list[ProgramWorldContextItem] = []
        raw_fallbacks: list[ProgramWorldContextItem] = []
        spent = 0
        for raw in ordered:
            item = self._item(raw)
            cost = int(raw.get("tokens") or 1)
            kind = item.kind
            required = item.required or kind in REQUIRED_KINDS
            if required and item.disposition == ContextItemDisposition.OMITTED.value:
                raise ProgramWorldContextError(
                    "required tests/proofs/policy/authority/raw source cannot be omitted"
                )
            if required and any(
                marker in f"{item.reason} {kind}".lower()
                for marker in ("embedding", "ann", "similar", "knn", "cosine")
            ):
                raise ProgramWorldContextError(
                    "embeddings cannot suppress required context material"
                )
            if item.disposition == ContextItemDisposition.RAW_FALLBACK.value:
                raw_fallbacks.append(item)
                spent += cost
                continue
            if required or spent + cost <= budget:
                included.append(
                    ProgramWorldContextItem(
                        identity_cid=item.identity_cid,
                        kind=item.kind,
                        reason=item.reason,
                        authority=item.authority,
                        freshness=item.freshness,
                        disposition=ContextItemDisposition.INCLUDED,
                        required=required,
                        source_cid=item.source_cid,
                    )
                )
                spent += cost
            else:
                omitted.append(
                    ProgramWorldContextItem(
                        identity_cid=item.identity_cid,
                        kind=item.kind,
                        reason="token_budget",
                        authority=item.authority,
                        freshness=item.freshness,
                        disposition=ContextItemDisposition.OMITTED,
                        required=False,
                        source_cid=item.source_cid,
                    )
                )
        if unresolved and not any(item.kind == "questions" for item in included):
            raise ProgramWorldContextError("unresolved questions require expansion")
        return _mirror_program_world_context_compilation(ProgramWorldContextReceipt(
            included=included,
            omitted=omitted,
            raw_fallbacks=raw_fallbacks,
            unresolved_questions=unresolved,
            token_budget=budget,
            fallback=bool(raw_fallbacks),
        ))

    def explain_program_world_context(
        self, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        receipt = self.compile_program_world_context(request)
        return {
            "included": [item.kind for item in receipt.included],
            "omitted": [item.kind for item in receipt.omitted],
            "raw_fallbacks": [item.kind for item in receipt.raw_fallbacks],
            "unresolved_questions": list(receipt.unresolved_questions),
            "token_budget": receipt.token_budget,
            "fallback": receipt.fallback,
        }

    def reuse_prefix(self, prefix: Mapping[str, Any]) -> ProgramWorldPrefixReuse:
        payload = dict(prefix)
        encoded = cid_for_payload(payload).encode("utf-8")
        key = cid_for_payload({"schema": "prefix", "body": payload})
        previous = self._prefix_cache.get(key)
        reused = previous == encoded
        self._prefix_cache[key] = encoded
        result = ProgramWorldPrefixReuse(
            prefix_cid=key, exact_bytes=True, reused=reused
        )
        try:
            from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
                mirror_work_record,
            )

            record_ref = str(result.prefix_cid or "program-world-prefix")
            mirror_work_record(
                catalog_kind="capsule",
                record_kind="program_world_prefix_reuse",
                record_ref=record_ref,
                subject_kind="record_cid",
                subject_ref=record_ref,
            )
        except Exception:
            pass
        return result

    def _item(self, raw: Mapping[str, Any]) -> ProgramWorldContextItem:
        try:
            return self._build_item(raw)
        except ProgramWorldReceiptError as exc:
            raise ProgramWorldContextError(str(exc)) from exc

    def _build_item(self, raw: Mapping[str, Any]) -> ProgramWorldContextItem:
        return ProgramWorldContextItem(
            identity_cid=str(raw["identity_cid"]),
            kind=str(raw["kind"]),
            reason=str(raw.get("reason") or "declared"),
            authority=str(raw.get("authority") or "program-world"),
            freshness=raw.get("freshness") or FreshnessState.FRESH,
            disposition=raw.get("disposition") or ContextItemDisposition.INCLUDED,
            required=_bool(raw.get("required", False), "required"),
            source_cid=raw.get("source_cid"),
        )


def compile_program_world_context(
    request: Mapping[str, Any],
    *,
    token_budget: int = 256,
) -> ProgramWorldContextReceipt:
    return ProgramWorldContextPlanner(token_budget=token_budget).compile_program_world_context(
        request
    )


def explain_program_world_context(
    request: Mapping[str, Any],
    *,
    token_budget: int = 256,
) -> dict[str, Any]:
    return ProgramWorldContextPlanner(token_budget=token_budget).explain_program_world_context(
        request
    )
