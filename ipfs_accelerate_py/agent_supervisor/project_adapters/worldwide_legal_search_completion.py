"""Native completion verification for the worldwide legal search campaign.

A done flag, a zero process exit, a stale receipt, skipped work, an unresolved
source blocker, or absent release evidence cannot establish completion. The
verifier reads only the observation it is given. It does not fetch the future
public release, execute validation, or grant completion authority.

WWL-106 is the implementation of this checker. Fixture release evidence is
enough to exercise it. The later public release tasks are not inputs.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from .worldwide_legal_search_progress import digest

OBSERVATION_SCHEMA = "wwl.native-completion-observation@1"
VERDICT_SCHEMA = "wwl.native-completion-verification@1"
TASK_RECEIPT_SCHEMA = "wwl.native-task-completion-receipt@1"
RELEASE_SCHEMA = "wwl.native-release-evidence@1"
SOURCE_SCHEMA = "wwl.native-source-disposition@1"
PROOF_SCHEMA = "wwl.native-current-tree-completion@1"
IMPLEMENTATION_SCHEMA = "wwl.completion-verifier-implementation@1"

TASK_ID = "WWL-106"
FRESHNESS_SECONDS = 3600
FUTURE_SKEW_SECONDS = 30
MAX_TASKS = 256
MAX_SOURCES = 256
MAX_EXPANSION = 64
MAX_OUTPUTS = 32
MAX_ARGV = 16
MAX_TEXT = 4096

NON_AUTHORITATIVE = (
    "done_flag",
    "zero_exit",
    "stale_receipt",
    "skipped_work",
    "unresolved_source_blocker",
    "absent_release_evidence",
)
REJECTION_ORDER = (
    "completion_authority_refused",
    "self_certified",
    "skipped_work",
    "unresolved_source_blocker",
    "stale_receipt",
    "absent_release_evidence",
    "done_flag",
    "zero_exit",
    "work_remains",
    "hold",
    "effects_unknown",
    "incomplete_evidence",
)
VERIFIED_REASON = "current_fixture_completion_verified"
DONE_STATUSES = frozenset({"done", "complete", "completed"})
SKIPPED_STATUSES = frozenset({"skipped", "skip", "omitted"})
STATUSES = frozenset({"", "ready", "running", "blocked", *DONE_STATUSES, *SKIPPED_STATUSES})
TERMINAL_SOURCE_DISPOSITIONS = frozenset({"ingested", "excluded"})
SOURCE_DISPOSITIONS = frozenset({*TERMINAL_SOURCE_DISPOSITIONS, "unresolved", "skipped"})
UNRESOLVED_BLOCKERS = frozenset({
    "source-outage",
    "gated-input",
    "credentials",
    "permission",
    "unknown-rights",
    "hold",
    "unresolved",
})
BLOCKER_KINDS = frozenset({"", "skipped", *UNRESOLVED_BLOCKERS})
OBSERVATION_FIELDS = (
    "schema", "run_id", "goal_id", "accepted_tree", "admitted_plan_revision",
    "inventory_revision", "source_revision", "observed_at", "required_task_ids",
    "tasks", "sources", "release", "hold", "effects_unknown", "runnable_count",
    "active_task_id", "expansion_child_ids",
)
OPTIONAL_CLAIMS = (
    "completion_verified", "completion_receipt", "done", "process_exit_code",
    "exit_code", "prose", "model_output", "status", "completion_authority",
    "task_retry_authority", "release_required", "future_release_required", "returncode",
)
TASK_FIELDS = ("task_id", "status", "exit_code", "skipped", "receipt")
RECEIPT_FIELDS = (
    "schema", "task_id", "run_id", "goal_id", "accepted_tree", "admitted_plan_revision",
    "source_revision", "validation_argv", "validation_exit_code", "validation_result",
    "validation_skipped", "output_hashes", "merge_receipt", "observed_at", "independent",
    "completion_authority",
)
RELEASE_FIELDS = (
    "schema", "run_id", "goal_id", "accepted_tree", "admitted_plan_revision", "commit",
    "artifact_digest", "public_url", "rollback_preserved", "observed_at", "fixture", "live_fetch",
)
SOURCE_FIELDS = (
    "schema", "source_id", "inventory_revision", "disposition", "blocker_kind",
    "reason", "ingest_receipt", "resolved_revision",
)
IGNORED_CLAIM_ORDER = (
    "done_flag", "zero_exit", "release_opt_out", "completion_authority", "prose",
)

_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")
_TASK = re.compile(r"^(?:WWL-[0-9]{3}|WWL-SRC-[0-9a-f]{16}-(?:normalize|index|geo|publish|verify)-[0-9]{4})$")
_GOAL = re.compile(r"^WWL-G[0-9]{3}(?:\.S[0-9]+)?$")
_RUN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SOURCE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
_REASON = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:/-]{0,160}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RESULT = re.compile(r"^[1-9][0-9]* passed$")
_URL = re.compile(r"^https://[A-Za-z0-9._~:/?#\[\]!$&'()*+,;=%-]{1,200}$")


class CompletionContractError(ValueError):
    """Observation is not a completion-verification envelope."""


def sealed(fields: tuple[str, ...], body: Mapping[str, Any]) -> dict[str, Any]:
    """Bind receipt_id to the canonical field subset."""
    missing = [key for key in fields if key not in body]
    if missing:
        raise CompletionContractError("sealed evidence is missing a field")
    payload = {key: body[key] for key in fields}
    try:
        receipt_id = digest(payload)
    except (TypeError, ValueError) as error:
        raise CompletionContractError("sealed evidence is not canonical") from error
    return {**payload, "receipt_id": receipt_id}


def implementation_obligations() -> dict[str, Any]:
    """State what WWL-106 itself requires.

    The checker is finished when fixture proofs pass. It does not consult the
    future public release and it does not establish campaign completion.
    """
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "task_id": TASK_ID,
        "goal_id": "WWL-G090",
        "subgoal_id": "WWL-G090.S2",
        "root_goal_id": "WWL-G000",
        "depends_on": ["WWL-105"],
        "future_release_required": False,
        "live_release_consulted": False,
        "required_release_tasks": [],
        "campaign_completion_authority": False,
        "completion_authority": False,
        "establishes_campaign_completion": False,
        "evidence": "fixtures",
        "non_authoritative": list(NON_AUTHORITATIVE),
    }


def verify_completion(
    observation: Mapping[str, Any],
    *,
    now: float,
    freshness_seconds: int = FRESHNESS_SECONDS,
) -> dict[str, Any]:
    """Return a fail-closed verdict for one completion observation.

    ``completion_verified`` is true only when every required task receipt, source
    disposition and release record is current and bound. Claims listed in
    ``NON_AUTHORITATIVE`` never flip that bit by themselves.
    """
    _require_clock(now)
    _require_freshness(freshness_seconds)
    if type(observation) is not dict:
        raise CompletionContractError("completion observation must be an object")
    _require_observation_keys(observation)
    _screen_identity(observation)
    tasks = _screen_tasks(observation)
    sources = _screen_sources(observation)
    _screen_release(observation.get("release"))
    _screen_claims(observation)

    found: set[str] = set()
    saw_done = _claims_done(observation, tasks)
    saw_zero = _claims_zero(observation, tasks)
    if observation.get("completion_authority") is True or observation.get("task_retry_authority") is True:
        found.add("completion_authority_refused")
    if observation["hold"] is True:
        found.add("hold")
    if observation["effects_unknown"] is True:
        found.add("effects_unknown")
    if observation["runnable_count"] != 0 or observation["active_task_id"] != "":
        found.add("work_remains")
    if not _fresh(observation["observed_at"], now, freshness_seconds):
        found.add("stale_receipt")
    required = list(observation["required_task_ids"])
    by_id = {task["task_id"]: task for task in tasks}
    if any(task_id not in by_id for task_id in required):
        found.add("skipped_work")
    expansion = list(observation["expansion_child_ids"])
    if any(child not in required for child in expansion):
        found.add("skipped_work")
    for task in tasks:
        found |= _task_rejections(task, observation, now, freshness_seconds, saw_done, saw_zero)
    found |= _source_rejections(sources, observation)
    found |= _release_rejections(observation.get("release"), observation, now, freshness_seconds)

    ignored = _ignored_claims(observation, tasks, found, saw_done, saw_zero)
    verified = not found
    proof = _proof(observation, tasks, sources) if verified else None
    reason = VERIFIED_REASON if verified else _ordered(found)[0]
    return _verdict(
        run_id=observation["run_id"],
        goal_id=observation["goal_id"],
        accepted_tree=observation["accepted_tree"],
        verified=verified,
        reason=reason,
        rejections=_ordered(found),
        ignored=ignored,
        proof=proof,
    )


def as_watchdog_verifier(
    snapshot: Mapping[str, Any],
    *,
    now: float | None = None,
    freshness_seconds: int = FRESHNESS_SECONDS,
) -> dict[str, Any]:
    """Adapt a watchdog snapshot. Snapshot flags and receipt CIDs are not evidence.

    The watchdog calls this with the snapshot only. A normal status projection
    has no source disposition and no release record, so it cannot verify.
    """
    clock = _clock(now)
    if type(snapshot) is not dict:
        return _rejected("", "", "", "incomplete_evidence", ["incomplete_evidence"])
    if snapshot.get("schema") == OBSERVATION_SCHEMA:
        try:
            return verify_completion(snapshot, now=clock, freshness_seconds=freshness_seconds)
        except CompletionContractError:
            tree = snapshot.get("accepted_tree")
            run_id = snapshot.get("run_id")
            goal_id = snapshot.get("goal_id")
            return _rejected(
                run_id if isinstance(run_id, str) else "",
                goal_id if isinstance(goal_id, str) else "",
                tree if isinstance(tree, str) else "",
                "incomplete_evidence",
                ["incomplete_evidence"],
            )
    return _snapshot_claims(snapshot)


def _clock(now: float | None) -> float:
    if now is None:
        import time
        return time.time()
    _require_clock(now)
    return now


def _rejected(run_id: str, goal_id: str, tree: str, reason: str, rejections: list[str]) -> dict[str, Any]:
    return _verdict(
        run_id=run_id, goal_id=goal_id, accepted_tree=tree, verified=False, reason=reason,
        rejections=rejections, ignored=[], proof=None,
    )


def _snapshot_claims(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    found: set[str] = set()
    required = snapshot.get("required_task_ids")
    statuses = snapshot.get("task_statuses")
    if not isinstance(required, list):
        required = []
    if not isinstance(statuses, Mapping):
        statuses = {}
    done_status = any(statuses.get(task_id) in DONE_STATUSES for task_id in required)
    if snapshot.get("completion_verified") is True or snapshot.get("done") is True or done_status or bool(snapshot.get("completion_receipt")):
        found.add("done_flag")
    if snapshot.get("process_exit_code") == 0 or snapshot.get("exit_code") == 0 or snapshot.get("returncode") == 0:
        found.add("zero_exit")
    if any(statuses.get(task_id) in SKIPPED_STATUSES for task_id in required):
        found.add("skipped_work")
    receipts = snapshot.get("native_completion_receipts_by_alias")
    if isinstance(receipts, Mapping) and any(isinstance(value, str) and value for value in receipts.values()):
        found.add("stale_receipt")
    sources = snapshot.get("sources")
    if not isinstance(sources, list) or not sources:
        found.add("unresolved_source_blocker")
    release = snapshot.get("release")
    if type(release) is not dict or release.get("schema") != RELEASE_SCHEMA:
        found.add("absent_release_evidence")
    if snapshot.get("hold") is True:
        found.add("hold")
    if snapshot.get("effects_unknown") is True:
        found.add("effects_unknown")
    runnable = snapshot.get("runnable_count")
    active = snapshot.get("task_id")
    if runnable not in (0, None) or (isinstance(active, str) and active != ""):
        found.add("work_remains")
    if snapshot.get("completion_authority") is True or snapshot.get("task_retry_authority") is True:
        found.add("completion_authority_refused")
    if not found:
        found.add("incomplete_evidence")
    tree = snapshot.get("accepted_tree")
    run_id = snapshot.get("run_id")
    goal_id = snapshot.get("goal_id")
    ordered = _ordered(found)
    return _rejected(
        run_id if isinstance(run_id, str) else "",
        goal_id if isinstance(goal_id, str) else "",
        tree if isinstance(tree, str) else "",
        ordered[0],
        ordered,
    )


def _verdict(
    *,
    run_id: str,
    goal_id: str,
    accepted_tree: str,
    verified: bool,
    reason: str,
    rejections: list[str],
    ignored: list[str],
    proof: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": VERDICT_SCHEMA,
        "run_id": run_id,
        "goal_id": goal_id,
        "accepted_tree": accepted_tree,
        "completion_verified": bool(verified) and proof is not None,
        "completion_receipt": digest(proof) if verified and proof is not None else "",
        "completion_authority": False,
        "task_retry_authority": False,
        "live_release_consulted": False,
        "reason": reason,
        "rejections": list(rejections),
        "ignored_claims": list(ignored),
        "proof": proof if verified else None,
    }


def _ordered(found: set[str]) -> list[str]:
    return [code for code in REJECTION_ORDER if code in found]


def _require_clock(now: float) -> None:
    if isinstance(now, bool) or type(now) not in (int, float) or not math.isfinite(now) or now < 0:
        raise CompletionContractError("completion clock is not a finite time")


def _require_freshness(freshness_seconds: int) -> None:
    if type(freshness_seconds) is not int or not 1 <= freshness_seconds <= 86400:
        raise CompletionContractError("completion freshness must be a positive bound")


def _require_observation_keys(observation: Mapping[str, Any]) -> None:
    keys = set(observation)
    required = set(OBSERVATION_FIELDS)
    allowed = required | set(OPTIONAL_CLAIMS)
    if not required <= keys:
        raise CompletionContractError("completion observation is missing a field")
    if not keys <= allowed:
        raise CompletionContractError("completion observation has an unknown field")


def _screen_identity(observation: Mapping[str, Any]) -> None:
    if observation.get("schema") != OBSERVATION_SCHEMA:
        raise CompletionContractError("completion observation schema is not supported")
    if not isinstance(observation.get("run_id"), str) or _RUN.fullmatch(observation["run_id"]) is None:
        raise CompletionContractError("completion run id is invalid")
    if not isinstance(observation.get("goal_id"), str) or _GOAL.fullmatch(observation["goal_id"]) is None:
        raise CompletionContractError("completion goal id is invalid")
    for key in ("accepted_tree", "admitted_plan_revision", "inventory_revision", "source_revision"):
        if not isinstance(observation.get(key), str) or _SHA.fullmatch(observation[key]) is None:
            raise CompletionContractError("completion identity is not a sha256 digest")
    if not _time_value(observation.get("observed_at")):
        raise CompletionContractError("completion observation time is invalid")
    if type(observation.get("hold")) is not bool or type(observation.get("effects_unknown")) is not bool:
        raise CompletionContractError("hold and unknown-effect flags must be booleans")
    runnable = observation.get("runnable_count")
    if type(runnable) is not int or runnable < 0 or runnable > MAX_TASKS:
        raise CompletionContractError("runnable count is invalid")
    active = observation.get("active_task_id")
    if not isinstance(active, str) or (active != "" and _TASK.fullmatch(active) is None):
        raise CompletionContractError("active task id is invalid")
    required = observation.get("required_task_ids")
    if not _task_list(required, allow_empty=False):
        raise CompletionContractError("required task set is invalid")
    if active != "" and active not in required:
        raise CompletionContractError("active task is outside the required set")
    expansion = observation.get("expansion_child_ids")
    if not _task_list(expansion, allow_empty=True) or len(expansion) > MAX_EXPANSION:
        raise CompletionContractError("expanded child task set is invalid")


def _task_list(value: Any, *, allow_empty: bool) -> bool:
    if not isinstance(value, list) or len(value) > MAX_TASKS:
        return False
    if not value and not allow_empty:
        return False
    if len(set(value)) != len(value):
        return False
    return all(isinstance(item, str) and _TASK.fullmatch(item) is not None for item in value)


def _screen_tasks(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = observation.get("tasks")
    if not isinstance(tasks, list) or len(tasks) > MAX_TASKS:
        raise CompletionContractError("task evidence list is invalid")
    required = set(observation["required_task_ids"])
    seen: set[str] = set()
    screened: list[dict[str, Any]] = []
    for task in tasks:
        if type(task) is not dict or set(task) != set(TASK_FIELDS):
            raise CompletionContractError("task evidence keys are not the receipt contract")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or _TASK.fullmatch(task_id) is None or task_id not in required:
            raise CompletionContractError("task is outside the required set")
        if task_id in seen:
            raise CompletionContractError("task evidence repeats a task")
        seen.add(task_id)
        if not isinstance(task.get("status"), str) or task["status"] not in STATUSES:
            raise CompletionContractError("task status is invalid")
        exit_code = task.get("exit_code")
        if exit_code is not None and (isinstance(exit_code, bool) or type(exit_code) is not int):
            raise CompletionContractError("task exit code is invalid")
        if type(task.get("skipped")) is not bool:
            raise CompletionContractError("task skip flag must be a boolean")
        receipt = task.get("receipt")
        if receipt is not None:
            if type(receipt) is not dict:
                raise CompletionContractError("task receipt must be an object or null")
            if receipt.get("schema") == TASK_RECEIPT_SCHEMA:
                _require_exact_keys(receipt, (*RECEIPT_FIELDS, "receipt_id"), "task receipt")
                _screen_receipt_types(receipt)
        screened.append(task)
    return screened


def _screen_sources(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = observation.get("sources")
    if not isinstance(sources, list) or len(sources) > MAX_SOURCES:
        raise CompletionContractError("source disposition list is invalid")
    seen: set[str] = set()
    for source in sources:
        if type(source) is not dict:
            raise CompletionContractError("source disposition must be an object")
        _require_exact_keys(source, SOURCE_FIELDS, "source disposition")
        if source.get("schema") != SOURCE_SCHEMA:
            raise CompletionContractError("source disposition schema is not supported")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or _SOURCE.fullmatch(source_id) is None:
            raise CompletionContractError("source id is invalid")
        if source_id in seen:
            raise CompletionContractError("source disposition repeats a source")
        seen.add(source_id)
        if not isinstance(source.get("inventory_revision"), str) or _SHA.fullmatch(source["inventory_revision"]) is None:
            raise CompletionContractError("source inventory revision is invalid")
        if source.get("disposition") not in SOURCE_DISPOSITIONS:
            raise CompletionContractError("source disposition is invalid")
        if source.get("blocker_kind") not in BLOCKER_KINDS:
            raise CompletionContractError("source blocker kind is invalid")
        for key in ("reason", "ingest_receipt", "resolved_revision"):
            if not isinstance(source.get(key), str):
                raise CompletionContractError("source disposition field is invalid")
    return list(sources)


def _screen_release(release: Any) -> None:
    if release is None:
        return
    if type(release) is not dict:
        raise CompletionContractError("release evidence must be an object or null")
    if release.get("schema") != RELEASE_SCHEMA:
        return
    _require_exact_keys(release, (*RELEASE_FIELDS, "receipt_id"), "release")
    for key in ("run_id", "goal_id", "accepted_tree", "admitted_plan_revision", "commit", "artifact_digest", "public_url", "receipt_id"):
        if not isinstance(release.get(key), str):
            raise CompletionContractError("release evidence field is invalid")
    if not _time_value(release.get("observed_at")):
        raise CompletionContractError("release evidence time is invalid")
    for key in ("rollback_preserved", "fixture", "live_fetch"):
        if type(release.get(key)) is not bool:
            raise CompletionContractError("release evidence flag is invalid")


def _screen_claims(observation: Mapping[str, Any]) -> None:
    if "completion_verified" in observation and type(observation["completion_verified"]) is not bool:
        raise CompletionContractError("done flag must be a boolean")
    if "done" in observation and type(observation["done"]) is not bool:
        raise CompletionContractError("done flag must be a boolean")
    if "status" in observation and (not isinstance(observation["status"], str) or observation["status"] not in STATUSES):
        raise CompletionContractError("done flag status is invalid")
    if "completion_receipt" in observation and (not isinstance(observation["completion_receipt"], str) or len(observation["completion_receipt"]) > 80):
        raise CompletionContractError("claimed completion receipt is invalid")
    for key in ("process_exit_code", "exit_code", "returncode"):
        if key in observation and (isinstance(observation[key], bool) or type(observation[key]) is not int):
            raise CompletionContractError("claimed exit code is invalid")
    for key in ("prose", "model_output"):
        if key in observation and (not isinstance(observation[key], str) or len(observation[key]) > MAX_TEXT):
            raise CompletionContractError("model prose is invalid")
    for key in ("completion_authority", "task_retry_authority", "release_required", "future_release_required"):
        if key in observation and type(observation[key]) is not bool:
            raise CompletionContractError("completion claim flag must be a boolean")


def _screen_receipt_types(receipt: Mapping[str, Any]) -> None:
    for key in ("task_id", "run_id", "goal_id", "accepted_tree", "admitted_plan_revision", "source_revision", "validation_result", "merge_receipt", "receipt_id"):
        if not isinstance(receipt.get(key), str):
            raise CompletionContractError("task receipt field is invalid")
    if not isinstance(receipt.get("validation_argv"), list) or not isinstance(receipt.get("output_hashes"), dict):
        raise CompletionContractError("task receipt evidence is invalid")
    if any(not isinstance(item, str) for item in receipt["validation_argv"]):
        raise CompletionContractError("task receipt argv is invalid")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in receipt["output_hashes"].items()):
        raise CompletionContractError("task receipt output hashes are invalid")
    for key in ("validation_exit_code", "validation_skipped"):
        if isinstance(receipt.get(key), bool) or type(receipt.get(key)) is not int:
            raise CompletionContractError("task receipt counter is invalid")
    if not _time_value(receipt.get("observed_at")):
        raise CompletionContractError("task receipt time is invalid")
    for key in ("independent", "completion_authority"):
        if type(receipt.get(key)) is not bool:
            raise CompletionContractError("task receipt flag is invalid")


def _require_exact_keys(value: Mapping[str, Any], fields: tuple[str, ...], label: str) -> None:
    if set(value) != set(fields):
        raise CompletionContractError(label + " keys are not the receipt contract")


def _time_value(value: Any) -> bool:
    return not isinstance(value, bool) and type(value) in (int, float) and math.isfinite(value)


def _fresh(observed: Any, now: float, freshness_seconds: int) -> bool:
    return _time_value(observed) and now - freshness_seconds <= observed <= now + FUTURE_SKEW_SECONDS


def _claims_done(observation: Mapping[str, Any], tasks: list[dict[str, Any]]) -> bool:
    return bool(
        observation.get("done") is True
        or observation.get("completion_verified") is True
        or observation.get("status") in DONE_STATUSES
        or observation.get("completion_receipt")
        or any(task["status"] in DONE_STATUSES for task in tasks)
    )


def _claims_zero(observation: Mapping[str, Any], tasks: list[dict[str, Any]]) -> bool:
    claimed = any(observation.get(key) == 0 for key in ("process_exit_code", "exit_code", "returncode"))
    return claimed or any(task["exit_code"] == 0 for task in tasks)


def _task_rejections(
    task: Mapping[str, Any],
    observation: Mapping[str, Any],
    now: float,
    freshness_seconds: int,
    saw_done: bool,
    saw_zero: bool,
) -> set[str]:
    found: set[str] = set()
    if task["skipped"] is True or task["status"] in SKIPPED_STATUSES:
        found.add("skipped_work")
    receipt = task["receipt"]
    if receipt is None:
        if "skipped_work" in found:
            return found
        if task["status"] in DONE_STATUSES or saw_done:
            found.add("done_flag")
        if task["exit_code"] == 0 or saw_zero:
            found.add("zero_exit")
        if "done_flag" not in found and "zero_exit" not in found:
            found.add("incomplete_evidence")
        return found
    if receipt.get("schema") != TASK_RECEIPT_SCHEMA:
        found.add("stale_receipt")
        return found
    if receipt["receipt_id"] != digest({key: receipt[key] for key in RECEIPT_FIELDS}):
        found.add("stale_receipt")
    if (
        receipt["task_id"] != task["task_id"]
        or receipt["run_id"] != observation["run_id"]
        or receipt["goal_id"] != observation["goal_id"]
        or receipt["accepted_tree"] != observation["accepted_tree"]
        or receipt["admitted_plan_revision"] != observation["admitted_plan_revision"]
        or receipt["source_revision"] != observation["source_revision"]
        or not _fresh(receipt["observed_at"], now, freshness_seconds)
    ):
        found.add("stale_receipt")
    if receipt["completion_authority"] is not False:
        found.add("completion_authority_refused")
    if receipt["independent"] is not True:
        found.add("self_certified")
    found |= _validation_rejections(receipt)
    return found


def _validation_rejections(receipt: Mapping[str, Any]) -> set[str]:
    result = receipt["validation_result"]
    skipped = receipt["validation_skipped"]
    if skipped != 0 or (isinstance(result, str) and "skip" in result.lower()):
        return {"skipped_work"}
    evidence_ok = (
        _argv_ok(receipt["validation_argv"])
        and isinstance(result, str)
        and _RESULT.fullmatch(result) is not None
        and _outputs_ok(receipt["output_hashes"])
        and isinstance(receipt["merge_receipt"], str)
        and _SHA.fullmatch(receipt["merge_receipt"]) is not None
        and receipt["validation_exit_code"] == 0
    )
    if evidence_ok:
        return set()
    if receipt["validation_exit_code"] == 0 and receipt["independent"] is True and receipt["completion_authority"] is False:
        return {"zero_exit"}
    return {"incomplete_evidence"}


def _argv_ok(argv: Any) -> bool:
    return (
        isinstance(argv, list)
        and 1 <= len(argv) <= MAX_ARGV
        and all(isinstance(item, str) and 1 <= len(item) <= 200 and "\x00" not in item for item in argv)
    )


def _outputs_ok(hashes: Any) -> bool:
    if not isinstance(hashes, dict) or not hashes or len(hashes) > MAX_OUTPUTS:
        return False
    return all(_relative_output(path) and _SHA.fullmatch(digest_value) is not None for path, digest_value in hashes.items())


def _relative_output(path: str) -> bool:
    if not isinstance(path, str) or not path or len(path) > 180 or path.startswith(("/", "\\")) or "\\" in path:
        return False
    return all(part not in ("", ".", "..") for part in path.split("/"))


def _source_rejections(sources: list[dict[str, Any]], observation: Mapping[str, Any]) -> set[str]:
    if not sources:
        return {"unresolved_source_blocker"}
    found: set[str] = set()
    for source in sources:
        disposition = source["disposition"]
        blocker = source["blocker_kind"]
        if disposition == "skipped" or blocker == "skipped":
            found.add("skipped_work")
            continue
        terminal = disposition in TERMINAL_SOURCE_DISPOSITIONS and blocker == ""
        if not terminal or disposition == "unresolved" or blocker in UNRESOLVED_BLOCKERS:
            found.add("unresolved_source_blocker")
            continue
        if source["inventory_revision"] != observation["inventory_revision"]:
            found.add("stale_receipt")
        if disposition == "excluded" and _REASON.fullmatch(source["reason"]) is None:
            found.add("unresolved_source_blocker")
        if disposition == "ingested" and (
            _SHA.fullmatch(source["ingest_receipt"]) is None or _SHA.fullmatch(source["resolved_revision"]) is None
        ):
            found.add("unresolved_source_blocker")
    return found


def _release_rejections(
    release: Any,
    observation: Mapping[str, Any],
    now: float,
    freshness_seconds: int,
) -> set[str]:
    # A live_fetch flag is not a fetch. This function never opens a URL.
    if type(release) is not dict or release.get("schema") != RELEASE_SCHEMA:
        return {"absent_release_evidence"}
    shape_ok = (
        release["live_fetch"] is False
        and release["rollback_preserved"] is True
        and _COMMIT.fullmatch(release["commit"]) is not None
        and _SHA.fullmatch(release["artifact_digest"]) is not None
        and _SHA.fullmatch(release["receipt_id"]) is not None
        and _URL.fullmatch(release["public_url"]) is not None
        and _RUN.fullmatch(release["run_id"]) is not None
        and _GOAL.fullmatch(release["goal_id"]) is not None
        and _SHA.fullmatch(release["accepted_tree"]) is not None
        and _SHA.fullmatch(release["admitted_plan_revision"]) is not None
    )
    if not shape_ok:
        return {"absent_release_evidence"}
    stale = (
        release["receipt_id"] != digest({key: release[key] for key in RELEASE_FIELDS})
        or release["run_id"] != observation["run_id"]
        or release["goal_id"] != observation["goal_id"]
        or release["accepted_tree"] != observation["accepted_tree"]
        or release["admitted_plan_revision"] != observation["admitted_plan_revision"]
        or not _fresh(release["observed_at"], now, freshness_seconds)
    )
    if stale:
        return {"stale_receipt"}
    return set()


def _ignored_claims(
    observation: Mapping[str, Any],
    tasks: list[dict[str, Any]],
    found: set[str],
    saw_done: bool,
    saw_zero: bool,
) -> list[str]:
    ignored: list[str] = []
    if saw_done and "done_flag" not in found:
        ignored.append("done_flag")
    if saw_zero and "zero_exit" not in found:
        ignored.append("zero_exit")
    if observation.get("release_required") is False or observation.get("future_release_required") is False:
        ignored.append("release_opt_out")
    if "completion_authority" in found or observation.get("completion_authority") is True or any(
        isinstance(task.get("receipt"), dict) and task["receipt"].get("completion_authority") is True for task in tasks
    ):
        if "completion_authority_refused" in found:
            ignored.append("completion_authority")
    if observation.get("prose") or observation.get("model_output"):
        ignored.append("prose")
    return [code for code in IGNORED_CLAIM_ORDER if code in ignored]


def _proof(
    observation: Mapping[str, Any],
    tasks: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {task["task_id"]: task for task in tasks}
    release = observation["release"]
    return {
        "schema": PROOF_SCHEMA,
        "run_id": observation["run_id"],
        "goal_id": observation["goal_id"],
        "accepted_tree": observation["accepted_tree"],
        "admitted_plan_revision": observation["admitted_plan_revision"],
        "inventory_revision": observation["inventory_revision"],
        "source_revision": observation["source_revision"],
        "required_task_ids": list(observation["required_task_ids"]),
        "verified_task_ids": list(observation["required_task_ids"]),
        "task_receipts": [
            {"task_id": task_id, "receipt_id": by_id[task_id]["receipt"]["receipt_id"]}
            for task_id in observation["required_task_ids"]
        ],
        "sources": [
            {
                "source_id": source["source_id"],
                "disposition": source["disposition"],
                "inventory_revision": source["inventory_revision"],
                "ingest_receipt": source["ingest_receipt"],
                "resolved_revision": source["resolved_revision"],
                "reason": source["reason"],
                "blocker_kind": source["blocker_kind"],
            }
            for source in sources
        ],
        "release_receipt": release["receipt_id"],
        "expansion_child_ids": list(observation["expansion_child_ids"]),
        "runnable_count": 0,
        "task_id": "",
        "effects_unknown": False,
        "hold": False,
        "live_release_consulted": False,
    }
