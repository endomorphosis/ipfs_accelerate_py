# Exact native producer from 5abf57b9f6a3c998784401b9f191b1e955adc218.
# This fixture reproduces the retained historical disposition conflict.
def _finalize_failed_attempt(
    self,
    attempt: "DatabaseTaskAttempt",
    *,
    reason: str,
    force_block: bool = False,
    unknown_authority: bool = False,
    attempt_consumed: bool = True,
    reconciliation_evidence: Mapping[str, Any] | None = None,
) -> tuple["DatabaseTaskAttempt", dict[str, Any]]:
    """Close a failed attempt without a redispatch/crash window.

    Canonical retry/blocked state is committed before the lane-local claim
    is released and before the execution projection becomes terminal.  If
    any later write fails, the still-running projection and dispatch
    journal make the operation replayable without invoking callbacks.
    """

    current = self.get_attempt(attempt.attempt_id) or attempt
    budget = dict(current.body.get("retry_budget") or {})
    retained_one_shot_consumed = self._retained_recovery_pair_binds_attempt(
        current,
        admission=budget.get("retained_recovery_admission"),
        consumption=budget.get("retained_recovery_consumption"),
    )
    if retained_one_shot_consumed:
        # The canonical retrying -> in_progress CAS spent this exact
        # operator credit before Portal/provider construction.  Neither
        # an unlimited ordinary retry policy nor a pre-provider deferral
        # may recreate retrying authority for the consumed one-shot.
        attempt_consumed = True
    provider_route_cooldown: dict[str, int] = {}
    if not attempt_consumed:
        callback_state = self._database_callback_boundary_state(current)
        provider_dispatch = callback_state["provider_dispatch"]
        provider_body = dict(
            (provider_dispatch or {}).get("body") or {}
        )
        if not (
            callback_state["safe_provider_route_deferred"]
            and not callback_state["callback_boundary_crossed"]
            and not callback_state["callback_authority_incomplete"]
            and callback_state["durable_provider_result"] is None
            and callback_state["durable_effect_result"] is None
            and not current.phase_committed(ATTEMPT_PHASE_PROVIDER)
            and not current.phase_committed(ATTEMPT_PHASE_EFFECT)
            and _database_provider_route_backoff_is_bounded(
                provider_body.get("backoff_seconds")
            )
            and type(provider_body.get("retry_not_before_ms")) is int
            and provider_body["retry_not_before_ms"] > 0
        ):
            raise DatabaseImplementationConflictError(
                "non-consuming retry lacks exact provider-route cooldown "
                "authority"
            )
        provider_route_cooldown = {
            "backoff_seconds": int(provider_body["backoff_seconds"]),
            "retry_not_before_ms": int(
                provider_body["retry_not_before_ms"]
            ),
        }
    task = self.task_source.get(current.task_cid)
    if task is not None and not self._retained_recovery_pair_is_exact_for_attempt(
        task,
        current,
    ):
        raise DatabaseImplementationConflictError(
            "retained recovery task and attempt chains differ"
        )
    try:
        claimed_attempts_used = max(
            1,
            int(
                budget.get("attempts_used")
                or (
                    self._retry_budget_state(task)["attempts_used"]
                    if task is not None
                    else 0
                )
                or current.attempt_number
            ),
        )
        attempts_used = (
            claimed_attempts_used
            if attempt_consumed
            else max(0, claimed_attempts_used - 1)
        )
    except (TypeError, ValueError):
        attempts_used = (
            max(1, int(current.attempt_number))
            if attempt_consumed
            else max(0, int(current.attempt_number) - 1)
        )
    try:
        retry_cap = int(
            budget.get("max_task_attempts", self.max_task_attempts)
        )
    except (TypeError, ValueError):
        retry_cap = self.max_task_attempts
    retry_exhausted = bool(
        force_block
        or retained_one_shot_consumed
        or (retry_cap > 0 and attempts_used >= retry_cap)
    )
    target_status = "blocked" if retry_exhausted else "retrying"
    binding = dict(current.body.get("control_claim") or {})
    task_status = (
        str(task.status or "").strip().lower() if task is not None else ""
    )
    task_receipt = (
        dict(task.body.get("completion_receipt") or {})
        if task is not None
        else {}
    )
    already_finalized = bool(
        task is not None
        and task_status in {"retrying", "blocked"}
        and task_receipt.get("schema") == DATABASE_RETRY_BUDGET_SCHEMA
        and str(task_receipt.get("attempt_id") or "") == current.attempt_id
        and str(task_receipt.get("claim_id") or "") == current.claim_id
        and str(task_receipt.get("lease_id") or "") == current.lease_id
        and str(task_receipt.get("owner_session_id") or "")
        == current.owner_session_id
        and int(task_receipt.get("attempt_number") or 0)
        == int(current.attempt_number)
        and int(task_receipt.get("fencing_token") or -1)
        == int(current.fencing_token)
        and int(task_receipt.get("fence_epoch") or -1)
        == int(current.fence_epoch)
    )
    exact_control = self._database_attempt_has_exact_control(current, task)
    if already_finalized:
        receipt = task_receipt
        if (
            not attempt_consumed
            and (
                not self._exact_provider_route_nonconsuming_receipt(
                    receipt,
                    current,
                )
                or not self._provider_route_nonconsuming_task_epoch_is_current(
                    task,
                    current,
                )
            )
            and not self._extra_gate_retry_receipt_is_claimable(
                task,
                receipt,
            )
        ):
            raise DatabaseImplementationConflictError(
                "non-consuming retry receipt changed its exact refund "
                "authority"
            )
        retry_exhausted = bool(
            receipt.get("retry_exhausted") or retained_one_shot_consumed
        )
        target_status = "blocked" if retry_exhausted else "retrying"
    elif exact_control:
        receipt = self._retry_budget_receipt(
            task,
            attempts_used=attempts_used,
            operation=(
                "database_unknown_outcome_blocked"
                if force_block
                else (
                    "database_retry_exhausted"
                    if retry_exhausted
                    else "database_retry_rearmed"
                )
            ),
            attempt=current,
            reason=(
                "provider_route_deferred_rearmed"
                if not attempt_consumed
                else reason
            ),
        )
        # A successor process may be the first observer able to prove
        # that a pre-crash provider/effect dispatch has no admissible
        # terminal evidence.  Preserve the process that actually claimed
        # and dispatched the exact fenced attempt as the blocking process;
        # otherwise the successor records itself here and can never use
        # the existing later-process rearm path without another unrelated
        # daemon restart.  Every binding below must match before the
        # origin is transferred.  Ambiguous records retain the current
        # process identity and therefore remain safely blocked.
        dispatch_origin_candidate = bool(
            force_block
            and task_receipt.get("schema")
            == DATABASE_RETRY_BUDGET_SCHEMA
            and str(task_receipt.get("attempt_id") or "")
            == current.attempt_id
            and str(task_receipt.get("claim_id") or "")
            == current.claim_id
            and bool(
                str(task_receipt.get("process_instance_id") or "").strip()
            )
        )
        try:
            dispatch_origin_matches = bool(
                dispatch_origin_candidate
                and int(task_receipt.get("fencing_token") or -1)
                == int(current.fencing_token)
                and int(task_receipt.get("fence_epoch") or -1)
                == int(current.fence_epoch)
            )
        except (TypeError, ValueError):
            dispatch_origin_matches = False
        if dispatch_origin_matches:
            receipt["process_instance_id"] = str(
                task_receipt["process_instance_id"]
            )
            receipt["owner_session_id"] = str(
                task_receipt.get("owner_session_id")
                or current.owner_session_id
            )
            if self.process_instance_id != receipt["process_instance_id"]:
                receipt["reconciled_by_process_instance_id"] = (
                    self.process_instance_id
                )
    else:
        # A replacement revision/validation epoch owns the canonical row.
        # Retire only this stale attempt and never debit or block the new
        # epoch with an old callback outcome.
        receipt = {
            "schema": DATABASE_RETRY_BUDGET_SCHEMA,
            "operation": "database_superseded_attempt_revoked",
            "task_cid": current.task_cid,
            "validation_spec_cid": str(
                binding.get("validation_spec_cid") or ""
            ),
            "attempts_used": attempts_used,
            "max_task_attempts": retry_cap,
            "retry_exhausted": False,
            "attempt_id": current.attempt_id,
            "claim_id": current.claim_id,
            "superseded": True,
            "reason": str(reason)[:256],
        }
        retry_exhausted = False
        target_status = "superseded"
    receipt["retry_exhausted"] = retry_exhausted
    if force_block:
        receipt["forced_block"] = True
    if unknown_authority:
        receipt["authority_outcome"] = "unknown"
    if not attempt_consumed:
        if already_finalized and any(
            receipt.get(name) != value
            for name, value in provider_route_cooldown.items()
        ):
            raise DatabaseImplementationConflictError(
                "non-consuming retry receipt changed its durable cooldown"
            )
        receipt["attempt_consumed"] = False
        receipt.update(provider_route_cooldown)
    if reconciliation_evidence:
        receipt["terminal_reconciliation"] = dict(
            reconciliation_evidence
        )
    if (
        retained_one_shot_consumed
        and already_finalized
        and task is not None
        and task_status == "retrying"
    ):
        self._cas_task_status_database(
            task.task_cid,
            expected_revision=int(task.revision),
            new_status="blocked",
            receipt=receipt,
        )
    if exact_control and task_status == "in_progress":
        self._cas_task_status_database(
            task.task_cid,
            expected_revision=int(task.revision),
            new_status=target_status,
            receipt=receipt,
        )
    elif exact_control and not already_finalized and task_status not in {
        target_status,
        "ready",
        "todo",
        "open",
        "pending",
        "queued",
        "proposed",
        "admitted",
        "cancelled",
        "canceled",
        "quarantined",
    }:
        raise DatabaseImplementationConflictError(
            f"cannot terminalize failed attempt against task status "
            f"{task_status!r}"
        )

    claim = self.coordinator.get_task_claim(current.claim_id)
    if claim is not None:
        claim_state = str(
            getattr(getattr(claim, "state", ""), "value", claim.state)
            or ""
        )
        if claim_state == "accepted":
            now = self._now_ms()
            if int(claim.expires_at_ms) <= now:
                expire_claim = getattr(
                    self.coordinator,
                    "expire_task_claim",
                    None,
                )
                if not callable(expire_claim):
                    raise DatabaseImplementationAuthorityError(
                        "coordinator cannot persist elapsed task-claim expiry"
                    )
                expire_claim(claim, now_ms=now)
            else:
                self.coordinator.release(
                    claim.as_fenced_lease(),
                    reason=(
                        "effect_outcome_unknown"
                        if force_block
                        else "implementation_attempt_failed"
                    ),
                    expected_fencing_token=int(claim.fencing_token),
                    expected_fence_epoch=int(claim.fence_epoch),
                    now_ms=now,
                )
    current = self.get_attempt(current.attempt_id) or current
    if current.status == "running":
        actual_database_disposition = (
            "superseded_attempt_revoked"
            if receipt.get("operation")
            == "database_superseded_attempt_revoked"
            else (
                "blocked_unknown_outcome"
                if force_block
                else (
                    "provider_route_deferred_rearmed"
                    if not attempt_consumed
                    else "terminalized_for_retry"
                )
            )
        )
        failed_phase_body: dict[str, Any] = {
            "reason": str(reason)[:512],
            "retry_exhausted": retry_exhausted,
            "unknown_authority": unknown_authority,
            "database_disposition": actual_database_disposition,
        }
        if not attempt_consumed:
            failed_phase_body["attempt_consumed"] = False
            failed_phase_body.update(provider_route_cooldown)
        if reconciliation_evidence:
            failed_phase_body["terminal_reconciliation"] = dict(
                reconciliation_evidence
            )
        current = self.commit_phase(
            current,
            ATTEMPT_PHASE_FAILED,
            body=failed_phase_body,
            require_live_claim=False,
        )
    return current, receipt
