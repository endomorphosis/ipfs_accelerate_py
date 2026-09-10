"""Native producer/consumer methods over an explicitly admitted queue client.

This adapter opens no database, issues no grant and performs no recovery replay.
Native source/owner migration and recovery/settlement admission remain separate.
"""

from __future__ import annotations

import json
from typing import Mapping, Any

from .merge_queue import MergeQueueFenceError, MergeRequest
from .owner_merge_queue import OwnerMergeQueueClient, OwnerMergeQueueError


class OwnerMergeQueueAdapter:
    """Ordinary merge operations; no local files or implicit legacy fallback."""

    def __init__(self, client: OwnerMergeQueueClient):
        self._client = client
        self.target_repository_id = client.repository_id
        self.target_branch = client.target_branch
        self.require_target_binding = True

    def bind_target(self, target_repository_id, target_branch, *, required=True):
        if type(required) is not bool or (
            target_repository_id,
            target_branch,
            required,
        ) != (
            self._client.repository_id,
            self._client.target_branch,
            True,
        ):
            raise MergeQueueFenceError("adapter cannot change its admitted target")

    def _consumer(self, consumer_id):
        if consumer_id not in (None, self._client.consumer_id):
            raise MergeQueueFenceError("consumer differs from admitted queue client")

    def _decode(self, data):
        if not isinstance(data, dict):
            raise OwnerMergeQueueError("queue response must contain a request object")
        request = MergeRequest.from_dict(data)
        if (
            not request.has_target_binding
            or request.target_repository_id != self._client.repository_id
            or request.target_branch != self._client.target_branch
            or json.dumps(request.to_dict(), sort_keys=True, allow_nan=False)
            != json.dumps(data, sort_keys=True, allow_nan=False)
        ):
            raise OwnerMergeQueueError("queue response request identity differs")
        return request

    def _request(self, operation, **arguments):
        result = self._client.call(operation, **arguments)
        value = result["request_json"]
        return None if value is None else self._decode(json.loads(value))

    def enqueue(
        self,
        *,
        branch_name,
        task_id,
        priority="P2",
        lane_id="",
        attempt=1,
        metadata=None,
        commit_sha="",
        canonical_task_id="",
        canonical_task_key="",
        canonical_task_cid="",
        target_repository_id="",
        target_branch="",
    ):
        # The closed producer operation creates attempt 1; recovery must use
        # its separately admitted exact claim transition, never reset history.
        if type(attempt) is not int or attempt != 1:
            raise OwnerMergeQueueError("enqueue cannot import an attempt history")
        if target_repository_id or target_branch:
            self.bind_target(target_repository_id, target_branch)
        if canonical_task_cid:
            if canonical_task_id and canonical_task_id != canonical_task_cid:
                raise OwnerMergeQueueError("canonical task identifiers differ")
            canonical_task_id = canonical_task_cid
        return self._request(
            "enqueue",
            branch_name=branch_name,
            task_id=task_id,
            priority=priority,
            lane_id=lane_id,
            commit_sha=commit_sha,
            canonical_task_id=canonical_task_id,
            canonical_task_key=canonical_task_key,
            metadata_json=json.dumps(
                {} if metadata is None else metadata, allow_nan=False
            ),
        )

    def get(self, request_id):
        return self._request("get", request_id=request_id)

    def _snapshot(self, operation, *, limit, **arguments):
        result = self._client.call(operation, limit=limit, **arguments)
        rows = json.loads(result["requests_json"])
        if not isinstance(rows, list) or len(rows) > limit:
            raise OwnerMergeQueueError(
                "queue response page differs from requested bound"
            )
        return tuple(self._decode(row) for row in rows)

    def pending_requests(self, *, limit=32, after_request_id=None):
        return self._snapshot(
            "pending_requests", limit=limit, after_request_id=after_request_id
        )

    def processing_requests(self, *, limit=32, after_request_id=None):
        return self._snapshot(
            "processing_requests", limit=limit, after_request_id=after_request_id
        )

    def quarantined_requests(self, *, limit=32, after_request_id=None):
        return self._snapshot(
            "quarantined_requests", limit=limit, after_request_id=after_request_id
        )

    def completed_requests(
        self,
        *,
        limit=32,
        metadata_schema="",
        require_completion_absent=False,
        completion_schema="",
        completion_reason="",
        canonical_task_id="",
        database_task_cid="",
        reopen_schema="",
        reopen_reason="",
        before_request_id="",
        ordered_by_request_id=False,
    ):
        return self._snapshot(
            "completed_requests",
            limit=limit,
            metadata_schema=metadata_schema,
            require_completion_absent=require_completion_absent,
            completion_schema=completion_schema,
            completion_reason=completion_reason,
            canonical_task_id=canonical_task_id,
            database_task_cid=database_task_cid,
            reopen_schema=reopen_schema,
            reopen_reason=reopen_reason,
            before_request_id=before_request_id,
            ordered_by_request_id=ordered_by_request_id,
        )

    def has_pending_for_task(self, task_id, *, commit_sha=None):
        value = self._client.call(
            "has_pending_for_task", task_id=task_id, commit_sha=commit_sha
        )["has_pending"]
        if type(value) is not bool:
            raise OwnerMergeQueueError("queue response has no active task observation")
        return value

    def dequeue(self, *, consumer_id=None):
        self._consumer(consumer_id)
        return self._request("dequeue")

    def claim_pending_request(self, request, *, consumer_id=None):
        self._consumer(consumer_id)
        if isinstance(request, MergeRequest):
            self._decode(request.to_dict())
            request = request.request_id
        return self._request("claim", request_id=request)

    def _claim(self, request):
        if not isinstance(request, MergeRequest):
            raise MergeQueueFenceError("a complete owner claim is required")
        self._decode(request.to_dict())
        if request.consumer_id != self._client.consumer_id:
            raise MergeQueueFenceError("request claim belongs to a different consumer")
        return {
            key: getattr(request, key)
            for key in ("request_id", "claim_token", "claim_generation")
        }

    def owns_claim(self, request, *, consumer_id=None):
        self._consumer(consumer_id)
        value = self._client.call("owns_claim", **self._claim(request))["owns_claim"]
        if type(value) is not bool:
            raise OwnerMergeQueueError("queue response has no claim observation")
        return value

    def _transition(
        self, operation, request, metadata: Mapping[str, Any] | None, **arguments
    ):
        return self._request(
            operation,
            **self._claim(request),
            **arguments,
            metadata_json=json.dumps(
                {} if metadata is None else dict(metadata), allow_nan=False
            ),
        )

    def complete(self, request, metadata=None):
        self._transition("complete", request, metadata)

    def requeue(self, request, reason="", *, metadata=None):
        result = self._transition("requeue", request, metadata, reason=reason)
        # The admitted owner does not materialize legacy JSON receipt paths.
        return None if result is not None and result.status == "quarantined" else result

    def quarantine(self, request, reason="", *, metadata=None):
        self._transition("quarantine", request, metadata, reason=reason)

    def fail(self, request, reason="", *, retryable=False, metadata=None):
        if type(retryable) is not bool:
            raise OwnerMergeQueueError("retryable must be a boolean")
        if retryable:
            self.requeue(request, reason=reason, metadata=metadata)
        else:
            self.quarantine(request, reason=reason, metadata=metadata)

    def defer(self, request, reason="", *, delay_seconds, metadata=None):
        return self._transition(
            "defer",
            request,
            metadata,
            reason=reason,
            delay_seconds_json=json.dumps(delay_seconds, allow_nan=False),
        )
