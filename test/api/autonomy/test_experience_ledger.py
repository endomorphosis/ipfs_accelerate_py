from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AuthorityClass,
    ExperienceEpisode,
    MemoryClass,
    MetaAction,
    TerminalStatus,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.experience_ledger import (
    ALLOWED_COMPACT_EPISODE_FIELDS,
    ALLOWED_EPISODE_ENVELOPE_FIELDS,
    EXPERIENCE_EPISODE_INTERFACE,
    EXPERIENCE_LEDGER_INTERFACE,
    ExperienceInvalidationReceipt,
    ExperienceLedger,
    ExperienceLedgerError,
    ExperienceLedgerSnapshot,
    InMemoryExperienceStore,
    compact_episode_payload,
    field_is_forbidden,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _episode(**overrides: Any) -> ExperienceEpisode:
    values: dict[str, Any] = {
        "frozen_input_ids": ("tree-1", "objective-rev-1"),
        "question_feature_ids": ("feature-cache",),
        "selected_action": MetaAction.RUN_LOCAL_STATIC_ANALYSIS,
        "selection_policy_id": "policy-1",
        "selection_policy_version": "policy-rev-1",
        "terminal_status": TerminalStatus.SUCCEEDED,
        "context_metrics": {"input_tokens": 12, "prefix_reused_tokens": 4},
        "evidence_ids": ("evidence-static-1",),
        "accepted_criterion_ids": ("AC-1",),
        "validation_receipt_ids": ("validation-1",),
        "latency_ms": 40,
        "cost_micros": 0,
    }
    values.update(overrides)
    return ExperienceEpisode(**values)


def _payload(episode: ExperienceEpisode | None = None, **extra: Any) -> dict[str, Any]:
    body = dict((episode or _episode()).to_dict())
    body.update(extra)
    return body


def test_interfaces_are_versioned_and_fields_are_closed() -> None:
    assert EXPERIENCE_LEDGER_INTERFACE == "ExperienceLedger@1"
    assert EXPERIENCE_EPISODE_INTERFACE == "ExperienceEpisode@1"
    assert ExperienceLedger.INTERFACE == EXPERIENCE_LEDGER_INTERFACE
    expected = {
        "frozen_input_ids",
        "question_feature_ids",
        "selected_action",
        "selection_policy_id",
        "selection_policy_version",
        "terminal_status",
        "provider_id",
        "model_id",
        "context_metrics",
        "token_measurement_ids",
        "evidence_ids",
        "accepted_criterion_ids",
        "validation_receipt_ids",
        "proof_receipt_ids",
        "merge_receipt_ids",
        "human_intervention_ids",
        "failure_signature",
        "repair_signature",
        "counterexample_ids",
        "cost_micros",
        "latency_ms",
    }
    assert ALLOWED_COMPACT_EPISODE_FIELDS == expected
    assert expected.issubset(ALLOWED_EPISODE_ENVELOPE_FIELDS)


def test_ledger_stores_only_allowed_compact_episode_fields() -> None:
    ledger = ExperienceLedger()
    episode = _episode(
        provider_id="provider-local",
        model_id="model-static",
        proof_receipt_ids=("proof-1",),
        merge_receipt_ids=("merge-1",),
        token_measurement_ids=("tokens-1",),
    )
    record = ledger.record(
        episode,
        evidence_authority={
            "evidence-static-1": AuthorityClass.VERIFIED,
            "validation-1": AuthorityClass.VERIFIED,
            "proof-1": AuthorityClass.VERIFIED,
            "merge-1": AuthorityClass.VERIFIED,
        },
        claimed_authority=AuthorityClass.VERIFIED,
    )
    payload = compact_episode_payload(record.episode)
    assert set(payload) <= ALLOWED_EPISODE_ENVELOPE_FIELDS
    assert set(payload) - {"schema", "contract_version"} <= ALLOWED_COMPACT_EPISODE_FIELDS | {
        "schema",
        "contract_version",
    }
    projection = ledger.projection(episode.episode_id)
    assert projection is not None
    assert set(projection["episode"]) <= ALLOWED_EPISODE_ENVELOPE_FIELDS
    encoded = json.dumps(projection)
    for marker in ("raw_prompt", "source_body", "transcript", "chain_of_thought", "api_key"):
        assert marker not in encoded
    assert ledger.episodes() == (episode,)
    assert record.authority_class is AuthorityClass.VERIFIED


@pytest.mark.parametrize(
    "field",
    (
        "raw_prompt",
        "prompt",
        "source_body",
        "transcript",
        "chain_of_thought",
        "hidden_reasoning",
        "model_transcript",
        "api_key",
        "private_key",
        "secret",
        "decoded_source",
        "executable_code",
    ),
)
def test_secrets_prompts_source_bodies_and_transcripts_are_rejected(field: str) -> None:
    assert field_is_forbidden(field)
    ledger = ExperienceLedger()
    with pytest.raises(ExperienceLedgerError, match="forbidden"):
        ledger.record(_payload(**{field: "do-not-store"}))
    with pytest.raises(ExperienceLedgerError, match="forbidden"):
        ledger.record(_payload(context_metrics={field: 1, "input_tokens": 1}))


def test_nested_and_textual_secret_material_is_rejected() -> None:
    ledger = ExperienceLedger()
    with pytest.raises(ExperienceLedgerError, match="forbidden"):
        ledger.record(
            _payload(
                context_metrics={
                    "input_tokens": 1,
                    "notes": {"user_prompt": "smuggled"},
                }
            )
        )
    pem = (
        "-----"
        + "BEGIN "
        + "PRIVATE "
        + "KEY-----"
        + "secret-material"
        + "-----END "
        + "PRIVATE "
        + "KEY-----"
    )
    with pytest.raises(ExperienceLedgerError, match="secret"):
        ledger.record(_payload(context_metrics={"input_tokens": 1, "annotation": pem}))
    with pytest.raises(ExperienceLedgerError, match="unsupported compact fields"):
        ledger.record(_payload(unexpected_model_claim="authoritative"))
    with pytest.raises(ExperienceLedgerError, match="integer units"):
        ledger.record(
            _episode(
                context_metrics={"input_tokens": "the full prompt text"}  # type: ignore[arg-type]
            )
        )


def test_size_bounds_and_capacity_fail_closed() -> None:
    with pytest.raises(ExperienceLedgerError, match="bounded"):
        ExperienceLedger().record(_payload(frozen_input_ids=("tree-" + ("x" * 600),)))
    ledger = ExperienceLedger(max_episodes=1)
    ledger.record(_episode())
    second = _episode(frozen_input_ids=("tree-2", "objective-rev-1"))
    with pytest.raises(ExperienceLedgerError, match="bounded size"):
        ledger.record(second)


def test_canonical_replay_is_idempotent_and_order_stable() -> None:
    first = _episode()
    second = _episode(
        frozen_input_ids=("tree-2",),
        evidence_ids=("evidence-other",),
        validation_receipt_ids=(),
    )
    ledger = ExperienceLedger()
    first_record = ledger.record(first, recorded_at_ms=10)
    second_record = ledger.record(second, recorded_at_ms=11)
    replayed = ledger.record(first, recorded_at_ms=99)
    assert replayed is first_record
    snapshot = ledger.snapshot()
    rebuilt = ExperienceLedger.from_snapshot(snapshot)
    assert rebuilt.snapshot().snapshot_id == snapshot.snapshot_id
    assert rebuilt.snapshot().to_dict() == snapshot.to_dict()
    assert content_identity(snapshot.to_dict()) == snapshot.snapshot_id
    assert {item.episode_id for item in rebuilt.current()} == {
        first_record.episode_id,
        second_record.episode_id,
    }
    again = ExperienceLedgerSnapshot.from_dict(snapshot.to_dict())
    assert again.snapshot_id == snapshot.snapshot_id
    with pytest.raises(FrozenInstanceError):
        first_record.withdrawn = True  # type: ignore[misc]


def test_dependency_invalidation_withdraws_only_direct_dependants() -> None:
    shared = _episode(frozen_input_ids=("tree-shared",), evidence_ids=("evidence-shared",))
    dependant = _episode(
        frozen_input_ids=("tree-shared",),
        evidence_ids=("evidence-other",),
        validation_receipt_ids=("validation-2",),
        question_feature_ids=("feature-other",),
    )
    independent = _episode(
        frozen_input_ids=("tree-other",),
        evidence_ids=("evidence-independent",),
        validation_receipt_ids=("validation-3",),
        question_feature_ids=("feature-independent",),
    )
    ledger = ExperienceLedger()
    ledger.record(shared)
    ledger.record(dependant)
    ledger.record(independent)

    receipt = ledger.invalidate("tree-shared")
    assert isinstance(receipt, ExperienceInvalidationReceipt)
    assert set(receipt.invalidated_episode_ids) == {shared.episode_id, dependant.episode_id}
    assert receipt.retained_episode_ids == (independent.episode_id,)
    assert ledger.get(shared.episode_id) is None
    assert ledger.get(shared.episode_id, include_withdrawn=True) is not None
    withdrawn = ledger.get(shared.episode_id, include_withdrawn=True)
    assert withdrawn is not None
    assert withdrawn.memory_class is MemoryClass.WITHDRAWN
    assert ledger.get(independent.episode_id) is not None
    assert ledger.episodes() == (independent,)
    assert ledger.by_dependency("evidence-independent")[0].episode_id == independent.episode_id

    with pytest.raises(ExperienceLedgerError, match="invalidated evidence"):
        ledger.record(
            _episode(
                frozen_input_ids=("tree-shared",),
                question_feature_ids=("feature-new",),
                evidence_ids=("evidence-new",),
                validation_receipt_ids=(),
            )
        )
    with pytest.raises(ExperienceLedgerError, match="revive"):
        ledger.record(shared)


def test_evidence_authority_cannot_be_upgraded() -> None:
    episode = _episode()
    ledger = ExperienceLedger()
    record = ledger.record(
        episode,
        evidence_authority={
            "evidence-static-1": AuthorityClass.ADVISORY,
            "validation-1": AuthorityClass.ADVISORY,
        },
        claimed_authority=AuthorityClass.ADVISORY,
    )
    assert record.authority_class is AuthorityClass.ADVISORY

    with pytest.raises(ExperienceLedgerError, match="cannot upgrade evidence authority"):
        ledger.record(
            episode,
            evidence_authority={
                "evidence-static-1": AuthorityClass.ADVISORY,
                "validation-1": AuthorityClass.ADVISORY,
            },
            claimed_authority=AuthorityClass.AUTHORITATIVE,
        )
    with pytest.raises(ExperienceLedgerError, match="cannot upgrade evidence authority"):
        ExperienceLedger().record(
            episode,
            evidence_authority={
                "evidence-static-1": AuthorityClass.ADVISORY,
                "validation-1": AuthorityClass.VERIFIED,
            },
            claimed_authority=AuthorityClass.VERIFIED,
        )
    with pytest.raises(ExperienceLedgerError, match="cannot upgrade evidence authority"):
        ExperienceLedger().record(episode, claimed_authority=AuthorityClass.DERIVED)
    with pytest.raises(ExperienceLedgerError, match="authoritative_current"):
        ExperienceLedger().record(
            episode,
            memory_class=MemoryClass.AUTHORITATIVE_CURRENT,
            evidence_authority={
                "evidence-static-1": AuthorityClass.ADVISORY,
                "validation-1": AuthorityClass.ADVISORY,
            },
            claimed_authority=AuthorityClass.ADVISORY,
        )

    for _ in range(8):
        sibling = _episode(
            frozen_input_ids=("tree-freq-" + str(_),),
            evidence_ids=("evidence-freq-" + str(_),),
            validation_receipt_ids=(),
        )
        ledger.record(
            sibling,
            evidence_authority={"evidence-freq-" + str(_): AuthorityClass.ADVISORY},
            claimed_authority=AuthorityClass.ADVISORY,
        )
    stored = ledger.get(episode.episode_id)
    assert stored is not None
    assert stored.authority_class is AuthorityClass.ADVISORY

    snapshot = ExperienceLedger.from_snapshot(ledger.snapshot())
    restored = snapshot.get(episode.episode_id)
    assert restored is not None
    assert restored.authority_class is AuthorityClass.ADVISORY
    with pytest.raises(ExperienceLedgerError, match="cannot upgrade evidence authority"):
        snapshot.record(
            episode,
            evidence_authority={
                "evidence-static-1": AuthorityClass.AUTHORITATIVE,
                "validation-1": AuthorityClass.AUTHORITATIVE,
            },
            claimed_authority=AuthorityClass.AUTHORITATIVE,
        )


def test_store_adapter_receives_only_compact_public_projections() -> None:
    store = InMemoryExperienceStore()
    ledger = ExperienceLedger(store=store)
    episode = _episode()
    record = ledger.record(
        episode,
        evidence_authority={
            "evidence-static-1": AuthorityClass.DERIVED,
            "validation-1": AuthorityClass.DERIVED,
        },
        claimed_authority=AuthorityClass.DERIVED,
        scope_ids=("objective:APMC-G040",),
    )
    assert store.keys()
    for raw in store.items().values():
        payload = json.loads(raw.decode("utf-8"))
        dumped = json.dumps(payload)
        for marker in ("raw_prompt", "source_body", "transcript", "password", "chain_of_thought"):
            assert marker not in dumped
        if "episode" in payload:
            assert set(payload["episode"]) <= ALLOWED_EPISODE_ENVELOPE_FIELDS
    raw_projection = store.get("experience-projection/" + episode.episode_id)
    assert raw_projection is not None
    loaded = json.loads(raw_projection.decode("utf-8"))
    assert loaded["episode_id"] == episode.episode_id
    assert loaded["authority_class"] == AuthorityClass.DERIVED.value
    assert loaded["record_id"] == record.record_id
    assert "raw_prompt" not in loaded


def test_compaction_drops_expired_ephemeral_rows_and_keeps_counterexamples() -> None:
    clock = {"now": 0}

    def now() -> int:
        return clock["now"]

    ledger = ExperienceLedger(clock=now)
    ephemeral = _episode(
        frozen_input_ids=("tree-eph",),
        evidence_ids=("evidence-eph",),
        validation_receipt_ids=(),
        terminal_status=TerminalStatus.FAILED,
    )
    negative = _episode(
        frozen_input_ids=("tree-neg",),
        evidence_ids=("evidence-neg",),
        validation_receipt_ids=(),
        counterexample_ids=("counterexample-1",),
        terminal_status=TerminalStatus.FAILED,
        failure_signature="missing-context",
    )
    durable = _episode()
    ledger.record(ephemeral, memory_class=MemoryClass.EPHEMERAL_ATTEMPT)
    ledger.record(negative, memory_class=MemoryClass.SHORT_LIVED_NEGATIVE)
    ledger.record(durable, memory_class=MemoryClass.TASK_EPISODE)
    clock["now"] = 5 * 60 * 1000
    dropped = ledger.compact()
    assert ephemeral.episode_id in dropped
    assert negative.episode_id not in dropped
    assert durable.episode_id not in dropped
    assert ledger.get(ephemeral.episode_id) is None
    assert ledger.get(negative.episode_id, include_withdrawn=True) is not None
    assert ledger.get(durable.episode_id) is not None


def test_authoritative_current_requires_authoritative_result_identities() -> None:
    episode = _episode()
    ledger = ExperienceLedger()
    record = ledger.record(
        episode,
        memory_class=MemoryClass.AUTHORITATIVE_CURRENT,
        evidence_authority={
            "evidence-static-1": AuthorityClass.AUTHORITATIVE,
            "validation-1": AuthorityClass.AUTHORITATIVE,
        },
        claimed_authority=AuthorityClass.AUTHORITATIVE,
    )
    assert record.memory_class is MemoryClass.AUTHORITATIVE_CURRENT
    assert record.authority_class is AuthorityClass.AUTHORITATIVE
    with pytest.raises(ExperienceLedgerError, match="does not cite"):
        ExperienceLedger().record(
            episode,
            evidence_authority={"unrelated-evidence": AuthorityClass.AUTHORITATIVE},
        )


def test_floats_and_non_episode_payloads_fail_closed() -> None:
    ledger = ExperienceLedger()
    with pytest.raises(ExperienceLedgerError, match="floats"):
        ledger.record(_payload(context_metrics={"input_tokens": 1.5}))
    with pytest.raises(ExperienceLedgerError, match="ExperienceEpisode"):
        ledger.record("not-an-episode")  # type: ignore[arg-type]
    with pytest.raises(ExperienceLedgerError, match="withdrawn"):
        ledger.record(_episode(), memory_class=MemoryClass.WITHDRAWN)
