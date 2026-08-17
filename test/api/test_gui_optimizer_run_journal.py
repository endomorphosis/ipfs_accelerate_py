"""VGO-054: interruption-safe improvement journal tests.

Acceptance coverage:

* interrupted runs resume without duplicate effect or canonical mutation
* kill/restart fixtures reload content-addressed checkpoints
* truncated and corrupt journals fail closed
* revision mismatch and foreign/stale worktrees reject
* process exit is never inferred as completion
* identical completed runs return the same terminal receipt identity
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.gui_optimizer.artifact_store import (
    ArtifactKind,
    default_evidence_artifact_store,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.run_journal import (
    GUI_RESUME_DECISION_INTERFACE,
    GUI_RUN_CHECKPOINT_INTERFACE,
    GUI_RUN_JOURNAL_INTERFACE,
    GuiRunJournal,
    GuiRunJournalError,
    JournalPhase,
    JournalReasonCode,
    PhaseRecordStatus,
    ResumeAction,
    RunStatus,
    default_run_journal,
    terminal_receipt_identity,
)

REVISION = "b" * 40
OTHER_REVISION = "c" * 40


def _open_kwargs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": "run:agent-supervisor-label",
        "application_id": "app:agent-supervisor",
        "screen_id": "screen:agent-supervisor",
        "objective_id": "objective:accessible-name",
        "source_revision": REVISION,
        "canonical_branch": "main",
        "canonical_revision": REVISION,
        "canonical_porcelain": "",
        "proposal_id": "proposal:label-form",
        "attempt": 1,
    }
    payload.update(overrides)
    return payload


def _resume_kwargs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": "run:agent-supervisor-label",
        "source_revision": REVISION,
        "canonical_branch": "main",
        "canonical_revision": REVISION,
        "canonical_porcelain": "",
        "process_alive": False,
    }
    payload.update(overrides)
    return payload


def _receipt(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "interface": "GuiJournalTerminalReceipt@1",
        "decision": "accept",
        "proposal_id": "proposal:label-form",
        "run_id": "run:agent-supervisor-label",
        "source_revision": REVISION,
        "verification_status": "verified",
    }
    payload.update(overrides)
    return payload


def test_open_append_and_heartbeat_are_content_addressed(tmp_path: Path) -> None:
    journal = default_run_journal(tmp_path / "runtime")
    opened = journal.open_run(**_open_kwargs())
    assert opened.interface == GUI_RUN_CHECKPOINT_INTERFACE
    assert opened.status is RunStatus.OPEN
    assert opened.phase is JournalPhase.BASELINE
    first = journal.append_phase(
        run_id=opened.run_id,
        phase=JournalPhase.BASELINE,
        effect_id="effect:baseline:1",
        payload={"violations": ["missing-name"], "attempt": 1},
    )
    second = journal.append_phase(
        run_id=opened.run_id,
        phase=JournalPhase.BASELINE,
        effect_id="effect:baseline:1",
        payload={"violations": ["missing-name"], "attempt": 1},
    )
    assert second.cid == first.cid
    assert second.effect_ids == ("effect:baseline:1",)
    beat = journal.heartbeat(opened.run_id)
    assert beat.heartbeat_seq == first.heartbeat_seq + 1
    assert beat.prev_checkpoint_cid == first.cid
    reopened = journal.open_run(**_open_kwargs())
    assert reopened.cid == beat.cid


def test_kill_restart_resumes_without_duplicate_effect(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    first = default_run_journal(root)
    opened = first.open_run(**_open_kwargs())
    first.append_phase(
        run_id=opened.run_id,
        phase=JournalPhase.IMPACT,
        effect_id="effect:impact:1",
        payload={"touched": ["comp:goal-form"]},
    )
    first.heartbeat(
        opened.run_id,
        worktree_path="/isolated/worktree",
        worktree_revision=REVISION,
        worktree_lease_id="lease:vgo-1",
    )
    first.mark_interrupted(opened.run_id)
    restarted = default_run_journal(root)
    decision = restarted.decide_resume(**_resume_kwargs())
    assert decision.action is ResumeAction.RESUME
    assert decision.interface == GUI_RESUME_DECISION_INTERFACE
    assert JournalReasonCode.PROCESS_EXIT_NOT_COMPLETION.value in decision.reason_codes
    assert decision.checkpoint is not None
    assert decision.checkpoint.status is RunStatus.INTERRUPTED
    replayed = restarted.append_phase(
        run_id=opened.run_id,
        phase=JournalPhase.IMPACT,
        effect_id="effect:impact:1",
        payload={"touched": ["comp:goal-form"]},
    )
    assert replayed.effect_ids.count("effect:impact:1") == 1
    continued = restarted.append_phase(
        run_id=opened.run_id,
        phase=JournalPhase.CONTEXT_PACK,
        effect_id="effect:context:1",
        payload={"pack_id": "pack:label-form"},
    )
    assert continued.effect_ids == ("effect:impact:1", "effect:context:1")


def test_truncated_and_corrupt_journals_fail_closed(tmp_path: Path) -> None:
    journal = default_run_journal(tmp_path / "runtime")
    opened = journal.open_run(**_open_kwargs())
    head = journal.host_root / "journals" / opened.run_id / "head.json"
    head.write_text("", encoding="utf-8")
    with pytest.raises(GuiRunJournalError) as truncated:
        journal.require_checkpoint(opened.run_id)
    assert truncated.value.reason_code == JournalReasonCode.TRUNCATED_JOURNAL.value
    head.write_text("{not-json", encoding="utf-8")
    with pytest.raises(GuiRunJournalError) as corrupt_head:
        journal.require_checkpoint(opened.run_id)
    assert corrupt_head.value.reason_code == JournalReasonCode.CORRUPT_JOURNAL.value
    fresh = default_run_journal(tmp_path / "runtime-b")
    sealed = fresh.open_run(**_open_kwargs())
    blob = fresh.store.host_path_for_cid(sealed.cid)
    blob.write_bytes(b'{"tampered":true}\n')
    with pytest.raises(GuiRunJournalError) as corrupt_checkpoint:
        fresh.require_checkpoint(sealed.run_id)
    assert (
        corrupt_checkpoint.value.reason_code
        == JournalReasonCode.CORRUPT_JOURNAL.value
    )


def test_revision_mismatch_and_canonical_mutation_reject(tmp_path: Path) -> None:
    journal = default_run_journal(tmp_path / "runtime")
    opened = journal.open_run(**_open_kwargs())
    journal.append_phase(
        run_id=opened.run_id,
        phase=JournalPhase.PROPOSAL,
        effect_id="effect:proposal:1",
        payload={"proposal_id": "proposal:label-form"},
    )
    mismatch = journal.decide_resume(
        **_resume_kwargs(source_revision=OTHER_REVISION)
    )
    assert mismatch.action is ResumeAction.REJECT
    assert JournalReasonCode.REVISION_MISMATCH.value in mismatch.reason_codes
    mutated = journal.decide_resume(
        **_resume_kwargs(canonical_revision=OTHER_REVISION)
    )
    assert mutated.action is ResumeAction.REJECT
    assert (
        JournalReasonCode.CANONICAL_MUTATION_DETECTED.value in mutated.reason_codes
    )
    dirty = journal.decide_resume(**_resume_kwargs(canonical_porcelain=" M file.js"))
    assert dirty.action is ResumeAction.REJECT
    assert (
        JournalReasonCode.CANONICAL_MUTATION_DETECTED.value in dirty.reason_codes
    )


def test_stale_and_foreign_worktrees_fail_closed(tmp_path: Path) -> None:
    journal = default_run_journal(tmp_path / "runtime")
    opened = journal.open_run(
        **_open_kwargs(
            worktree_path="/isolated/worktree-a",
            worktree_revision=REVISION,
            worktree_lease_id="lease:vgo-1",
        )
    )
    journal.append_phase(
        run_id=opened.run_id,
        phase=JournalPhase.ISOLATED_WORKTREE,
        effect_id="effect:worktree:apply",
        payload={"applied": True, "promoted": False},
        status=PhaseRecordStatus.COMPLETED,
    )
    journal.heartbeat(
        opened.run_id,
        worktree_path="/isolated/worktree-a",
        worktree_revision=REVISION,
        worktree_lease_id="lease:vgo-1",
    )
    foreign = journal.decide_resume(
        **_resume_kwargs(
            worktree_path="/isolated/worktree-b",
            worktree_revision=REVISION,
            worktree_lease_id="lease:vgo-1",
        )
    )
    assert foreign.action is ResumeAction.REJECT
    assert JournalReasonCode.FOREIGN_WORKTREE.value in foreign.reason_codes
    stale = journal.decide_resume(
        **_resume_kwargs(
            worktree_path="/isolated/worktree-a",
            worktree_revision=OTHER_REVISION,
            worktree_lease_id="lease:vgo-1",
        )
    )
    assert stale.action is ResumeAction.REJECT
    assert JournalReasonCode.STALE_WORKTREE.value in stale.reason_codes
    missing = journal.decide_resume(**_resume_kwargs())
    assert missing.action is ResumeAction.REJECT
    assert JournalReasonCode.STALE_WORKTREE.value in missing.reason_codes


def test_identical_completed_runs_return_same_terminal_receipt(
    tmp_path: Path,
) -> None:
    journal = default_run_journal(tmp_path / "runtime")
    opened = journal.open_run(**_open_kwargs())
    store = journal.store
    shot = store.put(
        b"baseline-png",
        kind=ArtifactKind.SCREENSHOT,
        binding={
            "repository_id": "repository:verified-gui-optimizer",
            "repository_revision": REVISION,
            "component_id": "comp:goal-form",
            "scenario_id": "scenario:keyboard-desktop",
            "extractor_id": "extractor:playwright@1",
            "extractor_version": "playwright@1.0.0",
            "checker_id": "checker:visual-regression@1",
            "checker_version": "visual-regression@1.0.0",
        },
    )
    journal.append_phase(
        run_id=opened.run_id,
        phase=JournalPhase.DECISION,
        effect_id="effect:decision:accept",
        payload={"decision": "accept"},
        artifact_cids=[shot.cid],
    )
    journal.bind_manifest(opened.run_id, [shot])
    receipt = _receipt()
    expected = terminal_receipt_identity(receipt)
    first = journal.commit_terminal_receipt(opened.run_id, receipt)
    assert first.status is RunStatus.COMPLETED
    assert first.terminal_receipt_digest == expected.digest
    second = journal.commit_terminal_receipt(opened.run_id, receipt)
    assert second.cid == first.cid
    assert second.terminal_receipt_cid == first.terminal_receipt_cid
    replay = default_run_journal(tmp_path / "runtime")
    decision = replay.decide_resume(**_resume_kwargs(process_alive=False))
    assert decision.action is ResumeAction.RETURN_COMPLETED
    assert decision.terminal_receipt_cid == first.terminal_receipt_cid
    assert decision.terminal_receipt_digest == expected.digest
    with pytest.raises(GuiRunJournalError) as mutated:
        replay.commit_terminal_receipt(opened.run_id, _receipt(decision="reject"))
    assert (
        mutated.value.reason_code
        == JournalReasonCode.COMPLETED_RECEIPT_MISMATCH.value
    )


def test_missing_run_restarts_and_closed_inputs_reject(tmp_path: Path) -> None:
    journal = default_run_journal(tmp_path / "runtime")
    missing = journal.decide_resume(**_resume_kwargs())
    assert missing.action is ResumeAction.RESTART
    with pytest.raises(GuiRunJournalError) as unknown:
        journal.open_from_mapping({**_open_kwargs(), "vendor": "hidden"})
    assert unknown.value.reason_code == JournalReasonCode.UNKNOWN_FIELD.value
    with pytest.raises(GuiRunJournalError) as path_key:
        journal.append_from_mapping(
            {
                "run_id": "run:agent-supervisor-label",
                "phase": "baseline",
                "effect_id": "effect:x",
                "payload": {"ok": True},
                "host_path": "/tmp/escape",
            }
        )
    assert path_key.value.reason_code == JournalReasonCode.BROWSER_PATH_FORBIDDEN.value
    with pytest.raises(GuiRunJournalError) as null_field:
        journal.decide_from_mapping({**_resume_kwargs(), "run_id": None})
    assert (
        null_field.value.reason_code
        == JournalReasonCode.INVALID_RUN_JOURNAL_INPUT.value
    )
    assert journal.interface == GUI_RUN_JOURNAL_INTERFACE
    assert GuiRunJournal(store=default_evidence_artifact_store(tmp_path / "alt")).store.host_root.exists()


def test_duplicate_effect_with_different_payload_fails_closed(tmp_path: Path) -> None:
    journal = default_run_journal(tmp_path / "runtime")
    opened = journal.open_run(**_open_kwargs())
    journal.append_phase(
        run_id=opened.run_id,
        phase=JournalPhase.COMPARE,
        effect_id="effect:compare:1",
        payload={"delta": 1},
    )
    with pytest.raises(GuiRunJournalError) as exc:
        journal.append_phase(
            run_id=opened.run_id,
            phase=JournalPhase.COMPARE,
            effect_id="effect:compare:1",
            payload={"delta": 2},
        )
    assert exc.value.reason_code == JournalReasonCode.DUPLICATE_EFFECT.value
