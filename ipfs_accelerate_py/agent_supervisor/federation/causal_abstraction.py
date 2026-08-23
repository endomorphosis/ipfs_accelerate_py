"""Causal abstraction maps and representative intervention consistency checks.

Only ``EXACT`` maps and separately policy-admitted ``CONSERVATIVE`` maps may
control scheduling or invalidation.  Heuristic, empirical, refuted, unknown, and
stale maps cannot suppress work.  Retrieval remains nomination-only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

from ..task_sources.control_plane_contracts import content_identity
from ..task_sources.quack_state_client import QuackStateClient, StatementKind
from .causal_graph import CausalGraphCommit, CausalGraphError, CausalGraphStore
from .contracts import (
    AbstractionFaithfulness,
    CausalAbstractionMap,
    FederationAuthorityError,
    FederationContractError,
    InterventionTest,
    _identifier,
    utc_now,
)
from .registry import _template

_AUTHORITATIVE_STATUSES = frozenset(
    {
        AbstractionFaithfulness.EXACT,
        AbstractionFaithfulness.CONSERVATIVE,
    }
)
_NOMINATION_STATUSES = frozenset(
    {
        AbstractionFaithfulness.EMPIRICALLY_SUPPORTED,
        AbstractionFaithfulness.HEURISTIC,
        AbstractionFaithfulness.REFUTED,
        AbstractionFaithfulness.UNKNOWN,
    }
)


class CausalAbstractionError(CausalGraphError):
    """Base typed abstraction-map failure."""


class CausalAbstractionAuthorityError(FederationAuthorityError, CausalAbstractionError):
    """An attempt to schedule or suppress work from a non-admitted map."""


@dataclass(frozen=True)
class StoredAbstractionMap:
    """Operational view of one persisted abstraction-map revision."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/stored-abstraction-map@1"
    )

    map_id: str
    revision: int
    faithfulness_status: AbstractionFaithfulness
    policy_admitted: bool
    resulting_status: AbstractionFaithfulness
    content_ref: str
    low_level_model_ref: str
    high_level_model_ref: str
    admitted_domain_ref: str
    excluded_domain_ref: str

    def __post_init__(self) -> None:
        _identifier(self.map_id, "map_id")
        if self.revision < 1:
            raise FederationContractError("map revision must be >= 1")
        if not isinstance(self.faithfulness_status, AbstractionFaithfulness):
            raise FederationContractError("faithfulness status is not closed")
        if not isinstance(self.resulting_status, AbstractionFaithfulness):
            raise FederationContractError("resulting status is not closed")
        if type(self.policy_admitted) is not bool:
            raise FederationContractError("policy_admitted must be boolean")
        _identifier(self.content_ref, "content_ref")
        _identifier(self.low_level_model_ref, "low_level_model_ref")
        _identifier(self.high_level_model_ref, "high_level_model_ref")
        _identifier(self.admitted_domain_ref, "admitted_domain_ref")
        _identifier(self.excluded_domain_ref, "excluded_domain_ref", required=False)

    @property
    def may_control_scheduling(self) -> bool:
        return map_may_control_scheduling(
            self.faithfulness_status,
            policy_admitted=self.policy_admitted,
            resulting_status=self.resulting_status,
        )


def _population_ref(values: Sequence[str], *, prefix: str) -> str:
    if not values:
        return ""
    return prefix + content_identity(list(values))


def evaluate_intervention(
    abstraction: CausalAbstractionMap,
    test: InterventionTest,
) -> InterventionTest:
    """Compare one representative intervention against an abstraction map."""

    if not isinstance(abstraction, CausalAbstractionMap):
        raise FederationContractError("abstraction must be a CausalAbstractionMap")
    if not isinstance(test, InterventionTest):
        raise FederationContractError("test must be an InterventionTest")
    if test.abstraction_map_id != abstraction.record_id:
        raise FederationContractError("intervention test is bound to a different map")
    domain = test.low_level_intervention_ref
    if domain in abstraction.excluded_domain_refs:
        computed = "excluded"
    elif domain not in abstraction.admitted_domain_refs:
        computed = "excluded"
    elif test.abstracted_outcome_ref == test.high_level_outcome_ref:
        computed = "matched"
    else:
        computed = "mismatched"
    if test.outcome != computed:
        raise CausalAbstractionAuthorityError(
            "intervention outcome does not match the observed comparison"
        )
    if computed == "mismatched" and not test.mismatch_ref:
        raise FederationContractError(
            "mismatched intervention requires a durable mismatch_ref"
        )
    if computed in {"matched", "excluded"}:
        return replace(test, mismatch_ref="")
    return test


def resulting_faithfulness(
    abstraction: CausalAbstractionMap,
    tests: Sequence[InterventionTest],
) -> AbstractionFaithfulness:
    """Fold durable intervention results into the live faithfulness status."""

    if any(item.outcome == "mismatched" for item in tests):
        return AbstractionFaithfulness.REFUTED
    return abstraction.faithfulness_status


def map_may_control_scheduling(
    faithfulness: AbstractionFaithfulness,
    *,
    policy_admitted: bool,
    resulting_status: AbstractionFaithfulness | None = None,
    expected_revision: int | None = None,
    live_revision: int | None = None,
) -> bool:
    """Return whether a map may drive scheduling or invalidation."""

    if expected_revision is not None and live_revision is not None:
        if expected_revision != live_revision:
            return False
    status = resulting_status if resulting_status is not None else faithfulness
    if status in _NOMINATION_STATUSES:
        return False
    if status is AbstractionFaithfulness.EXACT:
        return True
    return (
        status is AbstractionFaithfulness.CONSERVATIVE
        and faithfulness is AbstractionFaithfulness.CONSERVATIVE
        and policy_admitted
    )


def refuse_work_suppression(
    abstraction: CausalAbstractionMap | StoredAbstractionMap,
    *,
    expected_revision: int | None = None,
    live_revision: int | None = None,
    resulting_status: AbstractionFaithfulness | None = None,
) -> None:
    """Fail closed when a map must not suppress work."""

    if isinstance(abstraction, StoredAbstractionMap):
        faithfulness = abstraction.faithfulness_status
        policy_admitted = abstraction.policy_admitted
        resulting = (
            resulting_status
            if resulting_status is not None
            else abstraction.resulting_status
        )
        live = live_revision if live_revision is not None else abstraction.revision
        expected = (
            expected_revision if expected_revision is not None else abstraction.revision
        )
    else:
        faithfulness = abstraction.faithfulness_status
        policy_admitted = abstraction.policy_admitted
        resulting = (
            resulting_status if resulting_status is not None else faithfulness
        )
        live = live_revision
        expected = expected_revision
    if expected is not None and live is not None and expected != live:
        raise CausalAbstractionAuthorityError(
            "stale abstraction maps cannot suppress work"
        )
    if resulting is AbstractionFaithfulness.REFUTED:
        raise CausalAbstractionAuthorityError(
            "refuted abstraction maps cannot suppress work"
        )
    if resulting in {
        AbstractionFaithfulness.UNKNOWN,
        AbstractionFaithfulness.HEURISTIC,
        AbstractionFaithfulness.EMPIRICALLY_SUPPORTED,
    }:
        raise CausalAbstractionAuthorityError(
            "nomination-only abstraction maps cannot suppress work"
        )
    if not map_may_control_scheduling(
        faithfulness,
        policy_admitted=policy_admitted,
        resulting_status=resulting,
        expected_revision=expected,
        live_revision=live,
    ):
        raise CausalAbstractionAuthorityError(
            "abstraction map is not admitted to control scheduling"
        )


def _abstraction_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_select_causal_abstraction_map",
            """
            SELECT abstraction_map_id, revision, tenant_id, federation_id,
                   low_level_model_ref, high_level_model_ref,
                   abstraction_function_ref, intervention_mapping_ref,
                   admitted_domain_ref, excluded_domain_ref,
                   validation_evidence_ref, faithfulness_status,
                   policy_admission_ref, content_ref, recorded_at
            FROM causal_abstraction_maps
            WHERE abstraction_map_id = ? AND tenant_id = ? AND federation_id = ?
            ORDER BY revision DESC
            LIMIT 1
            """,
            ("abstraction_map_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_select_causal_abstraction_map_revision",
            """
            SELECT abstraction_map_id, revision, tenant_id, federation_id,
                   low_level_model_ref, high_level_model_ref,
                   abstraction_function_ref, intervention_mapping_ref,
                   admitted_domain_ref, excluded_domain_ref,
                   validation_evidence_ref, faithfulness_status,
                   policy_admission_ref, content_ref, recorded_at
            FROM causal_abstraction_maps
            WHERE abstraction_map_id = ? AND revision = ?
              AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            (
                "abstraction_map_id",
                "revision",
                "tenant_id",
                "federation_id",
            ),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_causal_abstraction_map",
            """
            INSERT INTO causal_abstraction_maps (
                abstraction_map_id, revision, tenant_id, federation_id,
                low_level_model_ref, high_level_model_ref,
                abstraction_function_ref, intervention_mapping_ref,
                admitted_domain_ref, excluded_domain_ref,
                validation_evidence_ref, faithfulness_status,
                policy_admission_ref, content_ref, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "abstraction_map_id",
                "revision",
                "tenant_id",
                "federation_id",
                "low_level_model_ref",
                "high_level_model_ref",
                "abstraction_function_ref",
                "intervention_mapping_ref",
                "admitted_domain_ref",
                "excluded_domain_ref",
                "validation_evidence_ref",
                "faithfulness_status",
                "policy_admission_ref",
                "content_ref",
                "recorded_at",
            ),
        ),
        _template(
            "casf_insert_causal_intervention_test",
            """
            INSERT INTO causal_intervention_tests (
                intervention_test_id, tenant_id, federation_id,
                abstraction_map_id, abstraction_map_revision,
                low_level_variable_ref, low_level_outcome_ref,
                abstracted_outcome_ref, high_level_intervention_ref,
                high_level_outcome_ref, disposition, evidence_ref, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "intervention_test_id",
                "tenant_id",
                "federation_id",
                "abstraction_map_id",
                "abstraction_map_revision",
                "low_level_variable_ref",
                "low_level_outcome_ref",
                "abstracted_outcome_ref",
                "high_level_intervention_ref",
                "high_level_outcome_ref",
                "disposition",
                "evidence_ref",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_causal_intervention_tests",
            """
            SELECT intervention_test_id, disposition, evidence_ref,
                   abstraction_map_revision
            FROM causal_intervention_tests
            WHERE abstraction_map_id = ? AND tenant_id = ? AND federation_id = ?
            ORDER BY recorded_at, intervention_test_id
            """,
            ("abstraction_map_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
        _template(
            "casf_insert_causal_abstraction_validation",
            """
            INSERT INTO causal_abstraction_validations (
                abstraction_validation_id, tenant_id, federation_id,
                abstraction_map_id, abstraction_map_revision,
                intervention_population_ref, mismatch_count,
                excluded_domain_ref, resulting_status, evidence_ref, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "abstraction_validation_id",
                "tenant_id",
                "federation_id",
                "abstraction_map_id",
                "abstraction_map_revision",
                "intervention_population_ref",
                "mismatch_count",
                "excluded_domain_ref",
                "resulting_status",
                "evidence_ref",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_causal_abstraction_validation",
            """
            SELECT resulting_status, mismatch_count, abstraction_map_revision,
                   evidence_ref
            FROM causal_abstraction_validations
            WHERE abstraction_map_id = ? AND tenant_id = ? AND federation_id = ?
            ORDER BY recorded_at DESC, abstraction_validation_id DESC
            LIMIT 1
            """,
            ("abstraction_map_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
    )


class CausalAbstractionStore(CausalGraphStore):
    """Sealed abstraction-map and intervention store over one state client."""

    INTERFACE = "CausalAbstractionStore@1"

    def __init__(
        self,
        client: QuackStateClient,
        *,
        event_notifier: Callable[[int], None] | None = None,
        outbox_notifier: Callable[[int], None] | None = None,
        test_failure_hook: Callable[[str], None] | None = None,
        require_quack_authority: bool = False,
    ) -> None:
        if isinstance(client, (str, bytes, Path)):
            raise CausalAbstractionError(
                "abstraction store never accepts a database path"
            )
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise CausalAbstractionError(
                "abstraction store requires an already-attached typed state client"
            )
        registered = set(client.list_templates())
        missing = [
            template.name
            for template in _abstraction_templates()
            if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise CausalAbstractionError(
                    "abstraction templates are absent from the sealed catalog"
                )
        else:
            for template in _abstraction_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    def record_map(
        self,
        abstraction: CausalAbstractionMap,
        *,
        federation_id: str,
        expected_graph_revision: int,
        idempotency_key: str,
    ) -> CausalGraphCommit:
        if not isinstance(abstraction, CausalAbstractionMap):
            raise FederationContractError("abstraction must be a CausalAbstractionMap")
        return self._commit_fact(
            operation="federation.causal.map.record",
            fact_id=abstraction.record_id,
            federation_id=federation_id,
            binding=abstraction.binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=(
                abstraction.record_id,
                abstraction.low_level_model_ref,
                abstraction.high_level_model_ref,
            ),
            payload_ref=abstraction.cid,
            prepare_fact=lambda: self._prepare_map(
                abstraction, federation_id=federation_id
            ),
            apply_fact=lambda revision, recorded_at: self._insert_map(
                abstraction,
                federation_id=federation_id,
                graph_revision=revision,
                recorded_at=recorded_at,
            ),
        )

    def record_intervention(
        self,
        abstraction: CausalAbstractionMap,
        test: InterventionTest,
        *,
        federation_id: str,
        expected_graph_revision: int,
        expected_map_revision: int,
        idempotency_key: str,
    ) -> CausalGraphCommit:
        evaluated = evaluate_intervention(abstraction, test)
        return self._commit_fact(
            operation="federation.causal.intervention.record",
            fact_id=evaluated.record_id,
            federation_id=federation_id,
            binding=evaluated.binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=(
                evaluated.record_id,
                abstraction.record_id,
                evaluated.low_level_intervention_ref,
            ),
            payload_ref=evaluated.cid,
            prepare_fact=lambda: self._prepare_intervention(
                abstraction,
                evaluated,
                federation_id=federation_id,
                expected_map_revision=expected_map_revision,
            ),
            apply_fact=lambda revision, recorded_at: self._insert_intervention(
                abstraction,
                evaluated,
                federation_id=federation_id,
                expected_map_revision=expected_map_revision,
                recorded_at=recorded_at,
            ),
        )

    def load_map(
        self,
        *,
        map_id: str,
        tenant_id: str,
        federation_id: str,
        revision: int | None = None,
    ) -> StoredAbstractionMap:
        if revision is None:
            rows = self._client.execute(
                "casf_select_causal_abstraction_map",
                {
                    "abstraction_map_id": _identifier(map_id, "map_id"),
                    "tenant_id": _identifier(tenant_id, "tenant_id"),
                    "federation_id": _identifier(federation_id, "federation_id"),
                },
            )
        else:
            rows = self._client.execute(
                "casf_select_causal_abstraction_map_revision",
                {
                    "abstraction_map_id": _identifier(map_id, "map_id"),
                    "revision": revision,
                    "tenant_id": _identifier(tenant_id, "tenant_id"),
                    "federation_id": _identifier(federation_id, "federation_id"),
                },
            )
        if len(rows) != 1:
            raise CausalAbstractionError("abstraction map is absent")
        return self._stored_from_row(rows[0], tenant_id=tenant_id, federation_id=federation_id)

    def scheduling_authority(
        self,
        *,
        map_id: str,
        tenant_id: str,
        federation_id: str,
        expected_revision: int,
    ) -> StoredAbstractionMap:
        stored = self.load_map(
            map_id=map_id,
            tenant_id=tenant_id,
            federation_id=federation_id,
        )
        refuse_work_suppression(stored, expected_revision=expected_revision)
        return stored

    def _prepare_map(self, abstraction: CausalAbstractionMap, *, federation_id: str) -> None:
        existing = self._client.execute(
            "casf_select_causal_abstraction_map",
            {
                "abstraction_map_id": abstraction.record_id,
                "tenant_id": abstraction.binding.tenant_id,
                "federation_id": federation_id,
            },
        )
        if existing:
            raise CausalAbstractionError("abstraction map identity is already bound")

    def _insert_map(
        self,
        abstraction: CausalAbstractionMap,
        *,
        federation_id: str,
        graph_revision: int,
        recorded_at: str,
    ) -> None:
        del graph_revision
        self._client.execute(
            "casf_insert_causal_abstraction_map",
            {
                "abstraction_map_id": abstraction.record_id,
                "revision": abstraction.revision,
                "tenant_id": abstraction.binding.tenant_id,
                "federation_id": federation_id,
                "low_level_model_ref": abstraction.low_level_model_ref,
                "high_level_model_ref": abstraction.high_level_model_ref,
                "abstraction_function_ref": abstraction.abstraction_function_ref,
                "intervention_mapping_ref": abstraction.intervention_mapping_ref,
                "admitted_domain_ref": _population_ref(
                    abstraction.admitted_domain_refs, prefix="admitted-domain:"
                ),
                "excluded_domain_ref": _population_ref(
                    abstraction.excluded_domain_refs, prefix="excluded-domain:"
                ),
                "validation_evidence_ref": _population_ref(
                    abstraction.validation_evidence_refs, prefix="validation-evidence:"
                ),
                "faithfulness_status": abstraction.faithfulness_status.value,
                "policy_admission_ref": (
                    abstraction.binding.policy_ref if abstraction.policy_admitted else ""
                ),
                "content_ref": abstraction.cid,
                "recorded_at": recorded_at,
            },
        )

    def _prepare_intervention(
        self,
        abstraction: CausalAbstractionMap,
        test: InterventionTest,
        *,
        federation_id: str,
        expected_map_revision: int,
    ) -> None:
        stored = self.load_map(
            map_id=abstraction.record_id,
            tenant_id=abstraction.binding.tenant_id,
            federation_id=federation_id,
        )
        if stored.revision != expected_map_revision:
            raise CausalAbstractionAuthorityError(
                "stale abstraction maps cannot suppress work"
            )
        if stored.content_ref != abstraction.cid:
            raise CausalAbstractionAuthorityError(
                "intervention map does not match the stored revision"
            )
        existing = self._client.execute(
            "casf_select_causal_intervention_tests",
            {
                "abstraction_map_id": abstraction.record_id,
                "tenant_id": abstraction.binding.tenant_id,
                "federation_id": federation_id,
            },
        )
        if any(str(item["intervention_test_id"]) == test.record_id for item in existing):
            raise CausalAbstractionError("intervention test identity is already bound")

    def _insert_intervention(
        self,
        abstraction: CausalAbstractionMap,
        test: InterventionTest,
        *,
        federation_id: str,
        expected_map_revision: int,
        recorded_at: str,
    ) -> None:
        prior = self._client.execute(
            "casf_select_causal_intervention_tests",
            {
                "abstraction_map_id": abstraction.record_id,
                "tenant_id": abstraction.binding.tenant_id,
                "federation_id": federation_id,
            },
        )
        mismatch_count = sum(1 for item in prior if str(item["disposition"]) == "mismatched")
        if test.outcome == "mismatched":
            mismatch_count += 1
        resulting = resulting_faithfulness(abstraction, (test,))
        if mismatch_count:
            resulting = AbstractionFaithfulness.REFUTED
        self._client.execute(
            "casf_insert_causal_intervention_test",
            {
                "intervention_test_id": test.record_id,
                "tenant_id": test.binding.tenant_id,
                "federation_id": federation_id,
                "abstraction_map_id": abstraction.record_id,
                "abstraction_map_revision": expected_map_revision,
                "low_level_variable_ref": test.low_level_intervention_ref,
                "low_level_outcome_ref": test.low_level_outcome_ref,
                "abstracted_outcome_ref": test.abstracted_outcome_ref,
                "high_level_intervention_ref": test.high_level_intervention_ref,
                "high_level_outcome_ref": test.high_level_outcome_ref,
                "disposition": test.outcome,
                "evidence_ref": test.mismatch_ref or test.cid,
                "recorded_at": recorded_at or utc_now(),
            },
        )
        self._client.execute(
            "casf_insert_causal_abstraction_validation",
            {
                "abstraction_validation_id": "validation:" + test.cid,
                "tenant_id": test.binding.tenant_id,
                "federation_id": federation_id,
                "abstraction_map_id": abstraction.record_id,
                "abstraction_map_revision": expected_map_revision,
                "intervention_population_ref": "interventions:" + test.cid,
                "mismatch_count": mismatch_count,
                "excluded_domain_ref": _population_ref(
                    abstraction.excluded_domain_refs, prefix="excluded-domain:"
                ),
                "resulting_status": resulting.value,
                "evidence_ref": test.cid,
                "recorded_at": recorded_at,
            },
        )

    def _stored_from_row(
        self,
        row: Mapping[str, Any],
        *,
        tenant_id: str,
        federation_id: str,
    ) -> StoredAbstractionMap:
        validations = self._client.execute(
            "casf_select_causal_abstraction_validation",
            {
                "abstraction_map_id": str(row["abstraction_map_id"]),
                "tenant_id": tenant_id,
                "federation_id": federation_id,
            },
        )
        faithfulness = AbstractionFaithfulness(str(row["faithfulness_status"]))
        resulting = faithfulness
        if validations:
            resulting = AbstractionFaithfulness(str(validations[0]["resulting_status"]))
        return StoredAbstractionMap(
            map_id=str(row["abstraction_map_id"]),
            revision=int(row["revision"]),
            faithfulness_status=faithfulness,
            policy_admitted=bool(str(row["policy_admission_ref"] or "")),
            resulting_status=resulting,
            content_ref=str(row["content_ref"]),
            low_level_model_ref=str(row["low_level_model_ref"]),
            high_level_model_ref=str(row["high_level_model_ref"]),
            admitted_domain_ref=str(row["admitted_domain_ref"]),
            excluded_domain_ref=str(row["excluded_domain_ref"] or ""),
        )


__all__ = (
    "CausalAbstractionAuthorityError",
    "CausalAbstractionError",
    "CausalAbstractionStore",
    "StoredAbstractionMap",
    "evaluate_intervention",
    "map_may_control_scheduling",
    "refuse_work_suppression",
    "resulting_faithfulness",
)
