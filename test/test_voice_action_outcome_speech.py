"""Tests for action outcome speech selection (VOICE-ACTION-026).

Acceptance:
- After execute/deny, spoken text prefers the library outcome frame.
- Missing library rows fall back safely.
- Transfer success is never invented without a succeeded receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pytest

from ipfs_accelerate_py.action_runtime.contracts import (
    ActionReceipt,
    ActionStatus,
)
from ipfs_accelerate_py.action_runtime.outcome_speech import (
    HANDOFF_LOGICAL_ACTIONS,
    LIBRARY_ROLE_DENY,
    LIBRARY_ROLE_FAIL,
    LIBRARY_ROLE_SUCCESS,
    OUTCOME_ROLE_DENIED,
    OUTCOME_ROLE_FAILED,
    OUTCOME_ROLE_SUCCESS,
    OUTCOME_ROLE_UNKNOWN,
    REASON_LIBRARY_HIT,
    REASON_LIBRARY_MISSING,
    REASON_NO_RECEIPT,
    REASON_TRANSFER_SUCCESS_BLOCKED,
    OutcomeSpeechFrame,
    OutcomeSpeechLibrary,
    allows_spoken_success,
    claims_transfer_success,
    default_action_speech_frames_path,
    library_role_for_outcome,
    outcome_frame_id,
    safe_fallback_spoken_text,
    select_outcome_speech,
    select_outcome_speech_from_default_library,
    spoken_outcome_role,
)


def _receipt(
    status: ActionStatus,
    *,
    receipt_id: str = "rcpt-test-1",
    proposal_id: str = "prop-test-1",
) -> ActionReceipt:
    return ActionReceipt(
        receipt_id=receipt_id,
        status=status,
        proposal_id=proposal_id,
        decision_id="dec-test-1",
        descriptor_id="voice.cli.open_app_surface.v1",
        adapter="cli",
        interface_identity="test",
    )


def _pilot_library() -> OutcomeSpeechLibrary:
    """Minimal in-memory library covering pilot open_app_surface + handoff."""

    rows = [
        OutcomeSpeechFrame(
            frame_id="frame.action.outcome.open_app_surface.success.v1",
            logical_action="open_app_surface",
            role=LIBRARY_ROLE_SUCCESS,
            spoken_text="I opened the requested app screen.",
        ),
        OutcomeSpeechFrame(
            frame_id="frame.action.outcome.open_app_surface.denied.v1",
            logical_action="open_app_surface",
            role=LIBRARY_ROLE_DENY,
            spoken_text="Okay, I will not open that app screen.",
        ),
        OutcomeSpeechFrame(
            frame_id="frame.action.outcome.open_app_surface.failed.v1",
            logical_action="open_app_surface",
            role=LIBRARY_ROLE_FAIL,
            spoken_text=(
                "I could not open that app screen right now. "
                "Please try again in a moment."
            ),
        ),
        OutcomeSpeechFrame(
            frame_id="frame.action.outcome.handoff_live_agent.success.v1",
            logical_action="handoff_live_agent",
            role=LIBRARY_ROLE_SUCCESS,
            spoken_text=(
                "Your request to speak with a live specialist has been submitted. "
                "I will not treat the transfer as complete until a provider confirms it."
            ),
        ),
        OutcomeSpeechFrame(
            frame_id="frame.action.outcome.handoff_live_agent.denied.v1",
            logical_action="handoff_live_agent",
            role=LIBRARY_ROLE_DENY,
            spoken_text=(
                "Okay, I will not request a live specialist right now. "
                "We can continue here, or you can ask again later."
            ),
        ),
        OutcomeSpeechFrame(
            frame_id="frame.action.outcome.handoff_live_agent.failed.v1",
            logical_action="handoff_live_agent",
            role=LIBRARY_ROLE_FAIL,
            spoken_text=(
                "I could not complete the live specialist request just now. "
                "Please try again, or stay on the line for more options."
            ),
        ),
    ]
    return OutcomeSpeechLibrary(rows)


@dataclass
class _FakeResolution:
    status: str
    reason: str
    audio: Optional[bytes] = None
    artifact: Any = None

    @property
    def hit(self) -> bool:
        return self.status == "hit"


class _FakeResolver:
    """Minimal exact-match resolver stub for audio hit/miss coverage."""

    def __init__(self, hits: Mapping[str, bytes]) -> None:
        self._hits = dict(hits)
        self.calls: list[dict[str, Any]] = []

    def resolve(
        self,
        spoken_text: str,
        identity: Any,
        *,
        template_id: str | None = None,
        response_id: str | None = None,
    ) -> _FakeResolution:
        self.calls.append(
            {
                "spoken_text": spoken_text,
                "identity": identity,
                "template_id": template_id,
                "response_id": response_id,
            }
        )
        audio = self._hits.get(spoken_text)
        if audio:
            return _FakeResolution(status="hit", reason="exact_match", audio=audio, artifact=object())
        return _FakeResolution(status="miss", reason="spoken_text_mismatch")


# ---------------------------------------------------------------------------
# Role / gate helpers
# ---------------------------------------------------------------------------


def test_spoken_outcome_role_maps_execute_and_deny_statuses() -> None:
    assert spoken_outcome_role(ActionStatus.SUCCEEDED) == OUTCOME_ROLE_SUCCESS
    assert spoken_outcome_role(ActionStatus.DENIED) == OUTCOME_ROLE_DENIED
    assert spoken_outcome_role(ActionStatus.FAILED) == OUTCOME_ROLE_FAILED
    assert spoken_outcome_role(ActionStatus.CANCELLED) == "cancelled"
    assert spoken_outcome_role(ActionStatus.ACCEPTED) == OUTCOME_ROLE_UNKNOWN
    assert spoken_outcome_role(ActionStatus.STARTED) == OUTCOME_ROLE_UNKNOWN
    assert spoken_outcome_role(ActionStatus.UNKNOWN) == OUTCOME_ROLE_UNKNOWN
    assert spoken_outcome_role(None) == OUTCOME_ROLE_UNKNOWN
    assert spoken_outcome_role(_receipt(ActionStatus.DENIED)) == OUTCOME_ROLE_DENIED


def test_allows_spoken_success_only_for_succeeded() -> None:
    for status in (
        ActionStatus.ACCEPTED,
        ActionStatus.STARTED,
        ActionStatus.UNKNOWN,
        ActionStatus.FAILED,
        ActionStatus.DENIED,
        ActionStatus.CANCELLED,
        None,
        "accepted",
    ):
        assert allows_spoken_success(status) is False
        assert spoken_outcome_role(status) != OUTCOME_ROLE_SUCCESS
    assert allows_spoken_success(ActionStatus.SUCCEEDED) is True
    assert allows_spoken_success("succeeded") is True


def test_library_role_mapping() -> None:
    assert library_role_for_outcome("success") == LIBRARY_ROLE_SUCCESS
    assert library_role_for_outcome("denied") == LIBRARY_ROLE_DENY
    assert library_role_for_outcome("failed") == LIBRARY_ROLE_FAIL
    assert library_role_for_outcome("cancelled") is None
    assert library_role_for_outcome("unknown") is None


def test_claims_transfer_success_detects_false_warmth() -> None:
    assert claims_transfer_success("You're connected to a live specialist.") is True
    assert claims_transfer_success("I've connected you to a live agent.") is True
    assert claims_transfer_success("The transfer is complete.") is True
    assert (
        claims_transfer_success(
            "Your request to speak with a live specialist has been submitted. "
            "I will not treat the transfer as complete until a provider confirms it."
        )
        is False
    )
    assert claims_transfer_success("Okay, I will not open that app screen.") is False


# ---------------------------------------------------------------------------
# Prefer library outcome frames after execute / deny
# ---------------------------------------------------------------------------


def test_execute_success_prefers_library_outcome_frame() -> None:
    library = _pilot_library()
    selection = select_outcome_speech(
        logical_action="open_app_surface",
        receipt=_receipt(ActionStatus.SUCCEEDED),
        library=library,
    )
    assert selection.source == "library"
    assert selection.reason == REASON_LIBRARY_HIT
    assert selection.outcome_role == OUTCOME_ROLE_SUCCESS
    assert selection.library_role == LIBRARY_ROLE_SUCCESS
    assert selection.spoken_text == "I opened the requested app screen."
    assert selection.frame_id == "frame.action.outcome.open_app_surface.success.v1"
    assert selection.spoken_success_allowed is True
    assert selection.audio_hit is False


def test_deny_prefers_library_outcome_frame() -> None:
    library = _pilot_library()
    selection = select_outcome_speech(
        logical_action="open_app_surface",
        receipt=ActionStatus.DENIED,
        library=library,
    )
    assert selection.source == "library"
    assert selection.reason == REASON_LIBRARY_HIT
    assert selection.outcome_role == OUTCOME_ROLE_DENIED
    assert selection.library_role == LIBRARY_ROLE_DENY
    assert selection.spoken_text == "Okay, I will not open that app screen."
    assert selection.frame_id == outcome_frame_id(
        "open_app_surface", LIBRARY_ROLE_DENY
    )
    assert selection.spoken_success_allowed is False


def test_failed_execute_prefers_library_fail_frame() -> None:
    library = _pilot_library()
    selection = select_outcome_speech(
        logical_action="open_app_surface",
        receipt="failed",
        library=library,
    )
    assert selection.source == "library"
    assert selection.outcome_role == OUTCOME_ROLE_FAILED
    assert selection.library_role == LIBRARY_ROLE_FAIL
    assert "could not open that app screen" in selection.spoken_text.lower()


# ---------------------------------------------------------------------------
# Safe fallback
# ---------------------------------------------------------------------------


def test_missing_library_frame_falls_back_safely() -> None:
    empty = OutcomeSpeechLibrary()
    selection = select_outcome_speech(
        logical_action="open_app_surface",
        receipt=ActionStatus.SUCCEEDED,
        library=empty,
    )
    assert selection.source == "safe_fallback"
    assert selection.reason == REASON_LIBRARY_MISSING
    assert selection.spoken_text == safe_fallback_spoken_text(
        logical_action="open_app_surface",
        outcome_role=OUTCOME_ROLE_SUCCESS,
    )
    assert selection.frame_id is None
    assert claims_transfer_success(selection.spoken_text) is False


def test_mapping_library_helper_and_none_library() -> None:
    # Compact mapping form.
    mapping_lib = {
        ("open_app_surface", "deny"): "Okay, I will not open that app screen.",
    }
    selection = select_outcome_speech(
        logical_action="open_app_surface",
        receipt=ActionStatus.DENIED,
        library=mapping_lib,
    )
    assert selection.source == "library"
    assert selection.spoken_text == "Okay, I will not open that app screen."

    # No library at all → safe fallback.
    selection2 = select_outcome_speech(
        logical_action="open_app_surface",
        receipt=ActionStatus.DENIED,
        library=None,
    )
    assert selection2.source == "safe_fallback"
    assert selection2.reason == REASON_LIBRARY_MISSING
    assert "will not take that action" in selection2.spoken_text.lower()


def test_unknown_and_cancelled_use_safe_fallback_without_library_roles() -> None:
    library = _pilot_library()
    unknown = select_outcome_speech(
        logical_action="open_app_surface",
        receipt=ActionStatus.STARTED,
        library=library,
    )
    assert unknown.outcome_role == OUTCOME_ROLE_UNKNOWN
    assert unknown.source == "safe_fallback"
    assert unknown.spoken_success_allowed is False
    assert claims_transfer_success(unknown.spoken_text) is False

    cancelled = select_outcome_speech(
        logical_action="open_app_surface",
        receipt=ActionStatus.CANCELLED,
        library=library,
    )
    assert cancelled.outcome_role == "cancelled"
    assert cancelled.source == "safe_fallback"
    assert "cancel" in cancelled.spoken_text.lower()


def test_missing_receipt_is_unknown_and_never_success() -> None:
    library = _pilot_library()
    selection = select_outcome_speech(
        logical_action="open_app_surface",
        receipt=None,
        library=library,
    )
    assert selection.outcome_role == OUTCOME_ROLE_UNKNOWN
    assert selection.reason == REASON_NO_RECEIPT
    assert selection.spoken_success_allowed is False
    assert selection.source == "safe_fallback"
    assert claims_transfer_success(selection.spoken_text) is False


# ---------------------------------------------------------------------------
# Never invent transfer success
# ---------------------------------------------------------------------------


def test_handoff_accepted_never_invents_transfer_success() -> None:
    library = _pilot_library()
    # Request creation (accepted) must not play the success frame or claim connection.
    selection = select_outcome_speech(
        logical_action="handoff_live_agent",
        receipt=_receipt(ActionStatus.ACCEPTED),
        library=library,
    )
    assert selection.outcome_role == OUTCOME_ROLE_UNKNOWN
    assert selection.spoken_success_allowed is False
    assert selection.source == "safe_fallback"
    assert claims_transfer_success(selection.spoken_text) is False
    assert "connected" not in selection.spoken_text.lower()
    assert "transfer is complete" not in selection.spoken_text.lower()
    # Must not reuse the success library frame for accepted.
    assert selection.frame_id != "frame.action.outcome.handoff_live_agent.success.v1"


@pytest.mark.parametrize(
    "status",
    [
        ActionStatus.ACCEPTED,
        ActionStatus.STARTED,
        ActionStatus.UNKNOWN,
        ActionStatus.FAILED,
        ActionStatus.DENIED,
        ActionStatus.CANCELLED,
    ],
)
def test_handoff_non_succeeded_statuses_never_claim_transfer(status: ActionStatus) -> None:
    library = _pilot_library()
    selection = select_outcome_speech(
        logical_action="handoff_live_agent",
        receipt=status,
        library=library,
    )
    assert selection.spoken_success_allowed is False
    assert claims_transfer_success(selection.spoken_text) is False
    assert selection.outcome_role != OUTCOME_ROLE_SUCCESS or status is ActionStatus.SUCCEEDED
    assert "you're connected" not in selection.spoken_text.lower()
    assert "i have connected you" not in selection.spoken_text.lower()


def test_handoff_succeeded_may_use_library_success_without_false_warmth() -> None:
    library = _pilot_library()
    selection = select_outcome_speech(
        logical_action="handoff_live_agent",
        receipt=ActionStatus.SUCCEEDED,
        library=library,
    )
    assert selection.spoken_success_allowed is True
    assert selection.source == "library"
    assert selection.outcome_role == OUTCOME_ROLE_SUCCESS
    assert selection.frame_id == "frame.action.outcome.handoff_live_agent.success.v1"
    # Library success for handoff is cautious (request submitted), not false warmth.
    assert "provider confirms" in selection.spoken_text.lower()
    assert claims_transfer_success(selection.spoken_text) is False


def test_blocks_library_frame_that_invents_transfer_success() -> None:
    """Even a malicious library row cannot invent transfer success without receipt."""

    bad = OutcomeSpeechLibrary(
        [
            OutcomeSpeechFrame(
                frame_id="frame.action.outcome.handoff_live_agent.success.v1",
                logical_action="handoff_live_agent",
                role=LIBRARY_ROLE_SUCCESS,
                spoken_text="You're connected to a live specialist.",
            )
        ]
    )
    # Force a path where role would be success only if succeeded — use succeeded
    # with a bad claim is allowed by gate (spoken_success_allowed True). So test
    # the blocked path with a non-success status that somehow had a claimy deny
    # frame is less relevant. Instead: inject claimy text for denied? Better:
    # use unknown status with a mapping that shouldn't use success — already covered.
    # Explicitly: when status is accepted, role is unknown so library success
    # is not consulted. To exercise REASON_TRANSFER_SUCCESS_BLOCKED, use a
    # deny frame that invents connection (pathological content).
    pathological = OutcomeSpeechLibrary(
        [
            OutcomeSpeechFrame(
                frame_id="frame.action.outcome.handoff_live_agent.denied.v1",
                logical_action="handoff_live_agent",
                role=LIBRARY_ROLE_DENY,
                spoken_text="You're connected to a live specialist.",
            )
        ]
    )
    selection = select_outcome_speech(
        logical_action="handoff_live_agent",
        receipt=ActionStatus.DENIED,
        library=pathological,
    )
    assert selection.source == "safe_fallback"
    assert selection.reason == REASON_TRANSFER_SUCCESS_BLOCKED
    assert claims_transfer_success(selection.spoken_text) is False
    assert selection.spoken_text == safe_fallback_spoken_text(
        logical_action="handoff_live_agent",
        outcome_role=OUTCOME_ROLE_DENIED,
    )
    # Succeeded with claimy success is authorized by the gate (provider confirmed).
    allowed = select_outcome_speech(
        logical_action="handoff_live_agent",
        receipt=ActionStatus.SUCCEEDED,
        library=bad,
    )
    assert allowed.spoken_success_allowed is True
    assert allowed.source == "library"
    assert allowed.spoken_text == "You're connected to a live specialist."


def test_handoff_logical_action_set() -> None:
    assert "handoff_live_agent" in HANDOFF_LOGICAL_ACTIONS


# ---------------------------------------------------------------------------
# Precomputed audio preference
# ---------------------------------------------------------------------------


def test_prefers_precomputed_audio_when_resolver_hits() -> None:
    library = _pilot_library()
    spoken = "I opened the requested app screen."
    audio = b"RIFF....wav-fixture"
    resolver = _FakeResolver({spoken: audio})
    selection = select_outcome_speech(
        logical_action="open_app_surface",
        receipt=ActionStatus.SUCCEEDED,
        library=library,
        audio_resolver=resolver,
        synthesis_identity={"provider": "fixture", "model": "m", "voice": "abby"},
    )
    assert selection.source == "library"
    assert selection.audio_hit is True
    assert selection.audio == audio
    assert resolver.calls
    assert resolver.calls[0]["spoken_text"] == spoken
    assert resolver.calls[0]["template_id"] == (
        "frame.action.outcome.open_app_surface.success.v1"
    )


def test_audio_miss_still_returns_spoken_text() -> None:
    library = _pilot_library()
    resolver = _FakeResolver({})
    selection = select_outcome_speech(
        logical_action="open_app_surface",
        receipt=ActionStatus.SUCCEEDED,
        library=library,
        audio_resolver=resolver,
        synthesis_identity={"provider": "fixture"},
    )
    assert selection.audio_hit is False
    assert selection.audio is None
    assert selection.spoken_text == "I opened the requested app screen."


# ---------------------------------------------------------------------------
# Default corpus integration (when present in workspace)
# ---------------------------------------------------------------------------


def test_default_corpus_load_when_present() -> None:
    path = default_action_speech_frames_path()
    if path is None:
        pytest.skip("action speech-frame corpus not present in this checkout")
    library = OutcomeSpeechLibrary.from_jsonl_path(path)
    assert library.frame_count >= 30  # 10 actions × 3 outcome roles
    selection = select_outcome_speech_from_default_library(
        logical_action="open_app_surface",
        receipt=ActionStatus.DENIED,
        frames_path=path,
    )
    assert selection.source == "library"
    assert selection.spoken_text == "Okay, I will not open that app screen."
    assert selection.frame_id == "frame.action.outcome.open_app_surface.denied.v1"


def test_from_records_skips_confirm_by_default() -> None:
    library = OutcomeSpeechLibrary.from_records(
        [
            {
                "frame_id": "frame.action.confirm.open_app_surface.v1",
                "logical_action": "open_app_surface",
                "role": "confirm",
                "spoken_text": "Say yes to continue.",
            },
            {
                "frame_id": "frame.action.outcome.open_app_surface.denied.v1",
                "logical_action": "open_app_surface",
                "role": "deny",
                "spoken_text": "Okay, I will not open that app screen.",
            },
        ]
    )
    assert library.frame_count == 1
    assert library.get("open_app_surface", "deny") is not None
    assert library.get("open_app_surface", "confirm") is None


def test_outcome_frame_ids_preference_when_present() -> None:
    library = _pilot_library()
    selection = select_outcome_speech(
        logical_action="open_app_surface",
        receipt=ActionStatus.DENIED,
        library=library,
        outcome_frame_ids={
            "denied": "frame.action.outcome.open_app_surface.denied.v1",
        },
    )
    assert selection.source == "library"
    assert selection.frame_id == "frame.action.outcome.open_app_surface.denied.v1"
    assert selection.metadata.get("preferred_frame_id") == (
        "frame.action.outcome.open_app_surface.denied.v1"
    )


def test_selection_to_dict_is_receipt_safe() -> None:
    selection = select_outcome_speech(
        logical_action="open_app_surface",
        receipt=ActionStatus.SUCCEEDED,
        library=_pilot_library(),
    )
    payload = selection.to_dict()
    assert payload["schema"].startswith("voice-action/")
    assert payload["task_id"] == "VOICE-ACTION-026"
    assert payload["spoken_text"]
    assert payload["source"] == "library"
    assert "command" not in payload
    assert "argv" not in payload


def test_requires_logical_action() -> None:
    with pytest.raises(ValueError, match="logical_action"):
        select_outcome_speech(logical_action="", receipt=ActionStatus.DENIED)
