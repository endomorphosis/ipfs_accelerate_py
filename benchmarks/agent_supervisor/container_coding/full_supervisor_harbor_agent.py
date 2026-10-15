"""Harbor adapter for the isolated admitted native supervisor, one task per container.

Cold planning, index construction, and coding share the selected agent budget.
Installation is setup time, measured by Harbor separately for both harnesses.
"""
from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.runtime.grok_coding_budget import (
    DEFAULT_GROK_CODING_MAX_TURNS, validate_grok_coding_max_turns, grok_coding_selection,
    require_grok_coding_archive, grok_coding_runtime_matches,
    grok_coding_not_applicable_provider_free,
)

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import stat
from time import monotonic
import uuid

from harbor.agents.base import BaseAgent
from harbor.models.agent.context import AgentContext

from .container_worker_deployment import (deploy_worker_boundary, doctor_dispatch_arguments,
    prompt_tokenizer_arguments, semantic_context_arguments, semantic_decision_arguments,
    validate_legacy_semantic_route)
from .semantic_decision_nomination_delivery import (capture_nomination_snapshot,
    nomination_delivery_arguments, nomination_delivery_command, read_nomination_source,
    verify_nomination_snapshot)
from .terminal_deployment import (
    ROOT, PYTHON, CANONICAL_CVE_PATH, SECURITY_INITIALIZER_DESCRIPTOR,
    SECURITY_INITIALIZER_PATH, SECURITY_CHECKPOINT_PATH, SECURITY_CHECKPOINT_HUB, deploy_supervisor, runtime_environment,
    SECURITY_FORMULA_PATH, SECURITY_FORMULA_DESCRIPTOR, SECURITY_HEADER_PROTOCOL,
    INTENT_CHECKPOINT_PATH, INTENT_ROUNDTRIP_PATH, INTENT_CHECKPOINT_DESCRIPTOR,
    INTENT_PROJECTION_REQUEST_PATH, INTENT_ACTION_384_CONFIG, validate_intent_action_384_binding,
    SOURCE384_CONFIG, validate_source384_binding,
)
from .benchmark_resource_profile import PROFILES, execution_budget, admission_environment
from .native_codex_baseline import MODEL, CLI_VERSION
from .benchmark_provider_profile import require_runtime_provider_profile, resolve_provider_profile
from .full_supervisor_benchmark import _intent_selection, validate_header_planning_selection, _prompt_options
from .terminal_public_outputs import capture_public_inputs, export_public_outputs
from .terminal_retrieval_selection import require_retrieval_revision
from .terminal_semantic_transport_policy import (
    DEFAULT_SEMANTIC_TRANSPORT_SCHEMA, validate_semantic_transport_schema,
    semantic_transport_selection, require_semantic_transport_archive,
)
from .terminal_coding_reply_policy import (
    DEFAULT_CODING_REPLY_MODE, validate_coding_reply_mode, require_coding_reply_archive,
    coding_reply_selection,
)
from .terminal_source_input_policy import (
    DEFAULT_SOURCE_INPUT_POLICY, validate_source_input_policy,
    source_input_selection, require_source_input_archive,
)
from .terminal_query_reference_policy import (
    DEFAULT_QUERY_REFERENCE_POLICY, validate_query_reference_policy,
    query_reference_selection, require_query_reference_archive,
)
from .terminal_repository_source_policy import (
    DEFAULT_REPOSITORY_SOURCE_POLICY, validate_repository_source_policy,
    repository_source_selection, require_repository_source_archive,
)
from .terminal_source_view_policy import (
    DEFAULT_SOURCE_VIEW_POLICY, validate_source_view_advisory_policy,
    source_view_advisory_selection, require_source_view_advisory_archive,
)
from .terminal_tool_output_policy import (
    DEFAULT_TOOL_OUTPUT_POLICY, validate_tool_output_policy,
    tool_output_selection, require_tool_output_archive,
)
from .terminal_semantic_metadata_policy import (
    DEFAULT_SEMANTIC_METADATA_VIEW, validate_semantic_metadata_view,
    semantic_metadata_observation, require_semantic_metadata_archive,
)
from .terminal_capsule_selection_policy import (
    DEFAULT_CAPSULE_SELECTION, validate_capsule_selection,
    capsule_selection_observation, require_capsule_selection_archive,
    DEFAULT_CAPSULE_BUDGET, validate_capsule_budget,
    capsule_budget_observation, require_capsule_budget_archive,
)
from .terminal_planning_policy import (
    DEFAULT_PLANNING_MODE, validate_planning_mode,
    planning_observation, require_planning_archive,
)
from .terminal_semantic_working_set_policy import (
    DEFAULT_SEMANTIC_WORKING_SET, SEMANTIC_WORKING_SETS, validate_semantic_working_set,
    semantic_working_set_selection, semantic_working_set_observation, require_semantic_working_set_archive,
)
from .terminal_source_edit_policy import (
    DEFAULT_SOURCE_EDIT_MODE, SOURCE_EDIT_MODES, validate_source_edit_mode,
    source_edit_selection, source_edit_observation, require_source_edit_archive,
)
from .terminal_source_view_policy import (
    DEFAULT_SOURCE_VIEW_PROFILE, validate_source_view_profile,
    source_view_observation, require_source_view_archive, validate_source_view_policy, source_view_policy_argv,
)
from .terminal_semantic_context_policy import (
    DEFAULT_SEMANTIC_CONTEXT_PROFILE, validate_semantic_context_profile,
    semantic_context_selection, require_semantic_context_archive,
)
from .terminal_numerical_policy import (
    validate_numerical_profile, numerical_observation, require_numerical_archive,
)
from .terminal_numerical_selection_deployment import (
    numerical_selection_asset_arguments,
)
from .terminal_provider_stop import provider_stop_observation
from .timeout_policy import MIN_DRIVER_TIMEOUT_SECONDS

PUBLIC_EXPORT_SECONDS = 5
ADAPTER_RECEIPT_MARGIN_SECONDS = 5
ADAPTER_FINALIZATION_SECONDS = 4 * PUBLIC_EXPORT_SECONDS + ADAPTER_RECEIPT_MARGIN_SECONDS


class _AdapterBudgetExhausted(TimeoutError):
    """The selected profile's original deadline cannot accommodate this stage."""


class _AdapterRunDeadline:
    def __init__(self, metadata, *, resource_profile=None, started=None,
                 agent_timeout_seconds=None):
        limits = execution_budget(resource_profile)
        seconds = limits["harbor_seconds"]
        # The optional legacy helper argument can confirm this profile, never
        # replace its independently selected physical execution limit.
        if agent_timeout_seconds is not None and (
                type(agent_timeout_seconds) is not int or agent_timeout_seconds != seconds):
            raise ValueError("adapter timeout must equal the selected resource profile")
        now = monotonic()
        if started is not None and (type(started) not in (int, float)
                or not math.isfinite(started) or started > now):
            raise ValueError("finite elapsed adapter start required")
        self.started = now if started is None else started
        self.deadline = self.started + seconds
        self.driver_limit = limits["driver_seconds"]
        self.exec_grace = limits["exec_seconds"] - self.driver_limit
        self.receipt = {
            "schema": "terminal-supervisor-adapter-deadline@1",
            "resource_profile": resource_profile,
            "budget_seconds": seconds,
            "profile_driver_seconds": self.driver_limit,
            "profile_exec_seconds": limits["exec_seconds"],
            "driver_exec_grace_seconds": self.exec_grace,
            "finalization_reserve_seconds": ADAPTER_FINALIZATION_SECONDS,
            "receipt_margin_seconds": ADAPTER_RECEIPT_MARGIN_SECONDS,
            "budget_exhausted": False, "stages": {},
        }
        metadata["adapter_deadline"] = self.receipt

    def remaining(self):
        return max(0.0, self.deadline - monotonic())

    def exhausted(self, stage, reason):
        if not self.receipt["budget_exhausted"]:
            self.receipt.update(budget_exhausted=True, exhausted_stage=stage,
                                exhaustion_reason=reason)
        return _AdapterBudgetExhausted("adapter budget exhausted before " + stage)

    def driver_seconds(self):
        seconds = min(self.driver_limit, int(self.remaining()
            - ADAPTER_FINALIZATION_SECONDS - self.exec_grace))
        if seconds < MIN_DRIVER_TIMEOUT_SECONDS:
            self.receipt["stages"]["driver"] = {
                "status": "not_started_budget_exhausted",
                "available_driver_seconds": max(0, seconds),
            }
            raise self.exhausted("driver", "insufficient_driver_budget")
        self.receipt["driver_timeout_seconds"] = seconds
        return seconds

    async def stage(self, name, operation, *, maximum=None,
                    reserve=ADAPTER_RECEIPT_MARGIN_SECONDS):
        # Do not construct an environment coroutine when its reserve is gone.
        available = self.remaining() - reserve
        if available <= 0:
            self.receipt["stages"][name] = {"status": "not_started_budget_exhausted"}
            raise self.exhausted(name, "reserved_finalization_time_unavailable")
        timeout = min(available, maximum) if maximum is not None else available
        started = monotonic()
        receipt = {"status": "running", "timeout_seconds": timeout}
        self.receipt["stages"][name] = receipt
        try:
            result = await asyncio.wait_for(operation(), timeout=timeout)
        except BaseException as error:
            receipt.update(status="cancelled" if isinstance(error, asyncio.CancelledError)
                           else "timed_out" if isinstance(error, TimeoutError) else "failed",
                           error_type=type(error).__name__)
            raise
        else:
            receipt["status"] = "completed"
            return result
        finally:
            receipt["seconds"] = max(0.0, monotonic() - started)

    def finish(self):
        self.receipt.update(seconds=max(0.0, monotonic() - self.started),
                            remaining_seconds=self.remaining(),
                            deadline_exhausted=self.remaining() <= 0)


def _retain_nomination_publisher_result(result, *, operation: str, artifact: Path) -> None:
    """Retain only this standalone publisher's bounded private diagnostics."""
    if operation not in {"prepare", "publish"}:
        raise ValueError("closed nomination publisher operation required")
    return_code = getattr(result, "return_code", None)
    record = {
        "schema": "terminal-semantic-decision-nomination-command@1", "operation": operation,
        "return_code": (return_code if type(return_code) is int
                        and -(2 ** 31) <= return_code < 2 ** 31 else None),
        "provider_called": False, "completion_authority": False,
    }
    for field in ("stdout", "stderr"):
        value = getattr(result, field, None)
        raw = value.encode("utf-8", errors="replace") if type(value) is str else None
        record[field] = raw[-4096:].decode("utf-8", errors="replace") if raw is not None else None
        record[field + "_bytes"] = len(raw) if raw is not None else None
        record[field + "_truncated"] = len(raw) > 4096 if raw is not None else None
    raw_record = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    if len(raw_record) > 65_536:
        raise ValueError("nomination publisher diagnostic exceeds its byte bound")
    descriptor = os.open(artifact, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw_record)
            stream.flush()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _nomination_delivery_receipt(result, expected: dict) -> dict:
    """Accept only the bounded root publisher's exact transport receipt."""
    if type(result.return_code) is not int or result.return_code != 0:
        raise ValueError("nomination delivery command failed")
    if type(result.stdout) is not str or len(result.stdout.encode("utf-8")) > 4096:
        raise ValueError("bounded nomination delivery receipt required")

    def closed_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate nomination delivery receipt field")
            value[key] = item
        return value

    receipt = json.loads(result.stdout, object_pairs_hook=closed_object)
    if (type(receipt) is not dict or set(receipt) != set(expected)
            or any(type(receipt[key]) is not type(value) or receipt[key] != value
                   for key, value in expected.items())):
        raise ValueError("nomination delivery receipt differs from pinned transport")
    return receipt


async def _download_context_replay_evidence(environment, *, state: str,
                                            receipt: dict, output: Path) -> dict:
    """Receive only the attested sibling archive through the closed verifier."""
    from .terminal_context_evidence import validate_evidence_receipt, verify_evidence_archive

    checked = validate_evidence_receipt(receipt, state=Path(state))
    destination = output / "context-replay-evidence.tar"
    receipt_destination = output / "context-replay-evidence-receipt.json"
    if any(os.path.lexists(path) for path in (destination, receipt_destination)):
        raise ValueError("fresh context replay evidence destination required")
    temporary = output / (".context-replay-evidence-" + uuid.uuid4().hex + ".tar")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.close(descriptor)
    published = receipt_identity = None
    completed = False

    def identity(path):
        info = path.lstat()
        return (info.st_dev, info.st_ino, info.st_size, info.st_mode,
                info.st_mtime_ns, info.st_ctime_ns)

    def remove_owned(path, held):
        if held is not None:
            try:
                current = path.lstat()
            except FileNotFoundError:
                return
            if (current.st_dev, current.st_ino) == held:
                path.unlink()

    try:
        await environment.download_file(str(state) + "-context-evidence.tar", temporary)
        info = temporary.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_size != checked["bytes"]):
            raise ValueError("context replay evidence size or file type differs")
        before = identity(temporary)
        # The worker only reads a bounded archive. Cancellation can unlink its
        # temporary pathname; the worker can never publish any output.
        manifest = await asyncio.to_thread(verify_evidence_archive, archive=temporary,
            expected_sha256=checked["sha256"], manifest_sha256=checked["manifest_sha256"])
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()
        if (identity(temporary) != before or len(encoded) != checked["manifest_bytes"]
                or hashlib.sha256(encoded).hexdigest() != checked["manifest_sha256"]
                or len(manifest["files"]) != checked["file_count"]
                or manifest["content_bytes"] != checked["content_bytes"]
                or any(manifest[key] != checked[key] for key in (
                    "repository_origin", "state_origin", "workspace_root_origin", "profile"))):
            raise ValueError("context replay evidence manifest differs from receipt")
        temporary.chmod(0o444)
        os.link(temporary, destination, follow_symlinks=False)
        info = destination.lstat()
        published = (info.st_dev, info.st_ino)
        temporary.unlink()
        result = {**checked, "artifact": str(destination), "status": "delivered", "verified": True}
        descriptor = os.open(receipt_destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            info = os.fstat(stream.fileno())
            receipt_identity = (info.st_dev, info.st_ino)
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        completed = True
        return result
    finally:
        temporary.unlink(missing_ok=True)
        if not completed:
            remove_owned(receipt_destination, receipt_identity)
            remove_owned(destination, published)


def _measured_native_usage(report: dict) -> dict:
    """Count observed native totals once, retaining provider-specific semantics."""
    from .native_usage_accounting import native_usage_contract

    rows = report.get("provider_invocations", [])
    unique = {}
    for row in rows:
        identity = row.get("invocation_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("provider invocation identity is unavailable")
        if identity in unique and unique[identity] != row:
            raise ValueError("conflicting provider receipts")
        unique[identity] = row
    known = bool(unique) and not report.get("unreceipted_provider_attempt")
    result = {}
    for key in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
        values = [(row.get("native_rollout_usage") or {}).get("usage", {}).get(key)
                  for row in unique.values()]
        result[key] = sum(values) if known and all(type(v) is int and v >= 0 for v in values) else None
    result["provider_calls"] = len(unique)
    result["observed_complete_sessions"] = known and all(
        (row.get("native_rollout_usage") or {}).get("task_complete_observed") is True
        for row in unique.values()
    ) if unique else False
    contracts = [native_usage_contract(row, row.get("native_rollout_usage") or {})
                 for row in unique.values()]
    grok = any(row.get("provider") == "grok_cli"
        or (row.get("native_rollout_usage") or {}).get("schema") == "native-grok-final-usage@1"
        for row in unique.values())
    cache_semantics = [contract[2] for contract in contracts]
    result["cache_included_in_input"] = (cache_semantics[0] if cache_semantics
        and all(type(value) is bool and value == cache_semantics[0] for value in cache_semantics)
        else None)
    result["source"] = "observed_native_final_totals" if grok else "observed_native_cumulative_totals"
    if grok:
        writes = [(row.get("native_rollout_usage") or {}).get("usage", {}).get("cache_write_input_tokens")
                  for row in unique.values()]
        result["cache_write_input_tokens"] = (sum(writes)
            if known and all(type(value) is int and value >= 0 for value in writes) else None)
        completeness = [(row.get("native_rollout_usage") or {}).get("usage_complete_observed")
                        for row in unique.values()]
        result["usage_complete_observed"] = (False if any(value is False for value in completeness)
            else True if known and all(value is True for value in completeness) else None)
    result["all_invocations_receipted"] = known
    complete = [contract[0] for contract in contracts]
    result["native_scoped_usage_complete_observed"] = (
        False if any(value is False for value in complete)
        else True if known and all(value is True for value in complete) else None)
    scopes = {contract[1] for contract in contracts}
    result["token_count_scope"] = next(iter(scopes)) if len(scopes) == 1 else None
    if result["token_count_scope"] is None:
        result["source"] = "observed_native_mixed_or_unknown_totals"
    result["harbor_usage_exportable"] = (
        result["native_scoped_usage_complete_observed"] is True
        and result["token_count_scope"] is not None
        and type(result["cache_included_in_input"]) is bool
        and all(type(result.get(key)) is int and result[key] >= 0 for key in (
            "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")))
    # A native task_complete event can close a failed session with no token
    # counters. Preserve that event separately from complete accounting.
    result["usage_accounting_complete"] = result["harbor_usage_exportable"]
    result["billing_total_verified"] = False
    result["dollar_cost"] = None
    return result


def measured_usage(report: dict) -> dict:
    """Apply decision coverage only when its explicit report scope is selected."""
    from .terminal_decision_usage_projection import decision_usage_selected, measured_decision_usage

    if decision_usage_selected(report):
        return measured_decision_usage(report)
    return _measured_native_usage(report)


def populate_harbor_usage(context: AgentContext, usage: dict) -> None:
    """Export qualified native-scope counters, never infer usage from success.

    Observed incomplete counters remain in the supervisor receipt. A qualified
    native scope still excludes billing or unreported auxiliary-model usage.
    """
    context.n_input_tokens = context.n_cache_tokens = context.n_output_tokens = None
    if usage.get("harbor_usage_exportable") is not True:
        return
    value = usage["input_tokens"] if usage.get("cache_included_in_input") is True else None
    if usage.get("cache_included_in_input") is False:
        categories = [usage.get(key) for key in ("input_tokens", "cached_input_tokens",
                                                "cache_write_input_tokens", "output_tokens", "total_tokens")]
        if (all(type(item) is int and item >= 0 for item in categories)
                and sum(categories[:4]) == categories[4]):
            value = sum(categories[:3])
    if value is None:
        return
    context.n_input_tokens = value
    context.n_cache_tokens = usage["cached_input_tokens"]
    context.n_output_tokens = usage["output_tokens"]


def security_asset_arguments(manifest: dict, arm: str) -> tuple[list[str], dict]:
    """Expose admitted portable assets only to the indexed benchmark arm."""
    if arm != "full":
        return [], {}
    arguments, observation = [], {}
    initializer = manifest.get("security_initializer")
    checkpoint = manifest.get("security_checkpoint")
    formula = manifest.get("formula_decoder")
    protocol = manifest.get("header_protocol")
    if formula is not None and checkpoint is None:
        raise ValueError("formula decoder requires the frozen security checkpoint profile")
    if protocol is not None and formula is None:
        raise ValueError("reviewed header protocol requires a formula decoder")
    if checkpoint is not None:
        if initializer is not None or manifest.get("canonical_cve_training") is not None:
            raise ValueError("frozen checkpoint and local training are mutually exclusive")
        descriptor = checkpoint.get("descriptor", {})
        if (checkpoint.get("path") != SECURITY_CHECKPOINT_PATH
                or descriptor.get("output") != ROOT + "/" + SECURITY_CHECKPOINT_PATH
                or descriptor.get("manifest_sha256") != checkpoint.get("manifest_sha256")
                or checkpoint.get("mode") != "frozen_inference"
                or checkpoint.get("runtime_training_steps") != 0 or checkpoint.get("runtime_download_calls") != 0):
            raise ValueError("runtime frozen checkpoint location or mode differs")
        arguments += ["--security-checkpoint", ROOT + "/" + SECURITY_CHECKPOINT_PATH,
                      "--security-checkpoint-manifest-sha256", checkpoint["manifest_sha256"]]
        if checkpoint.get("hub") is not None:
            if checkpoint.get("hub_descriptor_path") != SECURITY_CHECKPOINT_HUB:
                raise ValueError("runtime frozen checkpoint Hub provenance location differs")
            arguments += ["--security-checkpoint-hub-descriptor", ROOT + "/" + SECURITY_CHECKPOINT_HUB]
        observation["security_checkpoint"] = {"manifest_sha256": checkpoint["manifest_sha256"],
            "checkpoint_sha256": descriptor["checkpoint_sha256"], "hub": checkpoint.get("hub"),
            "mode": "frozen_inference", "training_steps": 0, "runtime_download_calls": 0}
    if formula is not None:
        descriptor = formula.get("descriptor", {})
        if (formula.get("path") != SECURITY_FORMULA_PATH
                or formula.get("descriptor_path") != SECURITY_FORMULA_DESCRIPTOR
                or descriptor.get("output") != ROOT + "/" + SECURITY_FORMULA_PATH
                or descriptor.get("manifest_sha256") != formula.get("manifest_sha256")
                or formula.get("mode") != "frozen_production_inference"
                or descriptor.get("mode") != "frozen_production_inference"
                or formula.get("source_training_data_included") is not False
                or formula.get("runtime_training_steps") != 0 or formula.get("runtime_download_calls") != 0):
            raise ValueError("runtime formula decoder location, mode or lineage differs")
        arguments += ["--formula-decoder-descriptor", ROOT + "/" + SECURITY_FORMULA_DESCRIPTOR]
        observation["formula_decoder"] = {"manifest_sha256": descriptor["manifest_sha256"],
            "weights_sha256": descriptor["weights_sha256"], "mode": "frozen_production_inference",
            "runtime_training_steps": 0, "runtime_download_calls": 0, "proof_authority": False}
    if protocol is not None:
        import hashlib
        reviewed = protocol.get("protocol")
        # The builder and container use the native reviewed-protocol contract.
        # Harbor only checks its bounded, pinned transport envelope; its
        # controller need not import the datasets runtime or scientific stack.
        if (type(reviewed) is not dict or set(reviewed) != {"review_ref", "callback_parameter"}
                or type(reviewed["review_ref"]) is not str or not reviewed["review_ref"].strip()
                or len(reviewed["review_ref"]) > 512
                or type(reviewed["callback_parameter"]) is not str
                or not reviewed["callback_parameter"].isidentifier()):
            raise ValueError("explicit closed reviewed header protocol required")
        digest = hashlib.sha256(json.dumps(reviewed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if protocol.get("path") != SECURITY_HEADER_PROTOCOL or protocol.get("sha256") != digest:
            raise ValueError("runtime reviewed header protocol binding differs")
        arguments += ["--header-protocol-descriptor", ROOT + "/" + SECURITY_HEADER_PROTOCOL]
        observation["header_protocol"] = {"sha256": digest, "review_ref": reviewed["review_ref"],
            "security_specification_inferred": False}
    if initializer is not None:
        if (initializer.get("path") != SECURITY_INITIALIZER_PATH
                or initializer.get("descriptor_path") != SECURITY_INITIALIZER_DESCRIPTOR
                or initializer.get("source_checkpoint_included") is not False
                or initializer["descriptor"].get("output") != ROOT + "/" + SECURITY_INITIALIZER_PATH):
            raise ValueError("runtime security initializer location or portability differs")
        arguments += ["--security-initializer", ROOT + "/" + SECURITY_INITIALIZER_DESCRIPTOR]
        observation["security_initializer"] = {key: initializer["descriptor"][key]
            for key in ("manifest_sha256", "initializer_sha256", "source_checkpoint_sha256")}
    training = manifest.get("canonical_cve_training")
    if training is not None:
        if training.get("path") != CANONICAL_CVE_PATH or training.get("raw_source_included") is not False:
            raise ValueError("runtime canonical CVE location or portability differs")
        arguments += ["--canonical-cve-export", ROOT + "/" + CANONICAL_CVE_PATH,
                      "--canonical-cve-manifest-sha256", training["manifest_sha256"]]
        observation["canonical_cve_training"] = {key: training[key] for key in (
            "manifest_sha256", "canonical_dataset_cid", "canonical_record_count", "training_pair_count")}
    return arguments, observation


def intent_asset_arguments(manifest: dict, *, enabled: bool = True) -> tuple[list[str], dict]:
    """Both supervisor arms share optional instruction preprocessing."""
    if type(enabled) is not bool:
        raise ValueError("explicit boolean Intent preprocessing selection required")
    if not enabled:
        return ["--disable-intent-autoencoder"], {"enabled": False, "status": "disabled"}
    selected = validate_intent_action_384_binding(manifest)
    if selected is not None:
        if manifest.get("intent_checkpoint") is not None or manifest.get("intent_projection_request") is not None:
            raise ValueError("Intent384 and legacy Intent selections are mutually exclusive")
        return ["--intent-action-384-config", ROOT + "/" + INTENT_ACTION_384_CONFIG], {
            "enabled": True, "sha256": selected["checkpoint_sha256"], "config_sha256": selected["config_sha256"],
            "embedding_revision": selected["embedding_revision"], "mode": selected["mode"],
            "execution_authority": False}
    binding = manifest.get("intent_checkpoint")
    if binding is None:
        if manifest.get("intent_projection_request") is not None:
            raise ValueError("runtime Intent projection request has no selected checkpoint")
        return [], {"enabled": True, "status": "fail_open_no_checkpoint"}
    import re
    descriptor = binding.get("descriptor")
    variants = {"intent-projection-feature-checkpoint/v1": (INTENT_CHECKPOINT_PATH, "frozen_structural_feature_inference"),
                "intent-roundtrip-checkpoint/v1": (INTENT_ROUNDTRIP_PATH, "frozen_semantic_roundtrip_inference"),
                "intent-copy-roundtrip-checkpoint/v1": (INTENT_ROUNDTRIP_PATH, "frozen_copy_roundtrip_inference")}
    if type(descriptor) is not dict or set(descriptor) != {"schema", "path", "sha256"} or descriptor.get("schema") not in variants:
        raise ValueError("runtime Intent checkpoint schema differs")
    checkpoint_path, mode = variants[descriptor["schema"]]
    if (descriptor["path"] != ROOT + "/" + checkpoint_path
            or type(descriptor["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])
            or binding.get("sha256") != descriptor["sha256"]
            or binding.get("path") != checkpoint_path
            or binding.get("descriptor_path") != INTENT_CHECKPOINT_DESCRIPTOR
            or binding.get("mode") != mode
            or type(binding.get("runtime_training_steps")) is not int or binding["runtime_training_steps"] != 0
            or type(binding.get("runtime_download_calls")) is not int or binding["runtime_download_calls"] != 0
            or binding.get("source_training_data_included") is not False):
        raise ValueError("runtime Intent checkpoint binding differs")
    arguments = ["--intent-checkpoint-descriptor", ROOT + "/" + INTENT_CHECKPOINT_DESCRIPTOR]
    observation = {
        "enabled": True, "sha256": descriptor["sha256"],
        "mode": mode, "execution_authority": False}
    request = manifest.get("intent_projection_request")
    if request is not None:
        fields = {"schema", "path", "sha256", "bytes", "request_sha256", "checkpoint_sha256",
                  "instruction_sha256", "source_ir_sha256"}
        digests = ("sha256", "request_sha256", "checkpoint_sha256", "instruction_sha256", "source_ir_sha256")
        if (type(request) is not dict or set(request) != fields
                or request["schema"] != "intent-projection-request/v1"
                or request["path"] != INTENT_PROJECTION_REQUEST_PATH
                or type(request["bytes"]) is not int or not 0 < request["bytes"] <= 65536
                or any(type(request[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", request[key])
                       for key in digests)
                or descriptor["schema"] not in {"intent-roundtrip-checkpoint/v1", "intent-copy-roundtrip-checkpoint/v1"}
                or request["checkpoint_sha256"] != descriptor["sha256"]):
            raise ValueError("runtime Intent projection request binding differs")
        arguments += ["--intent-projection-request", ROOT + "/" + INTENT_PROJECTION_REQUEST_PATH,
                      "--intent-projection-request-sha256", request["sha256"]]
        observation["projection_request"] = {key: request[key] for key in ("sha256", "request_sha256",
            "instruction_sha256", "source_ir_sha256", "checkpoint_sha256")}
    return arguments, observation


def source384_asset_arguments(manifest, arm, resource_profile=None):
    """Make selected parent use explicit; the no-index ablation never invokes it."""
    if arm not in {"full", "no-index"}:
        raise ValueError("unknown Source384 ablation")
    binding = validate_source384_binding(manifest)
    enabled = binding is not None and arm == "full"
    if enabled and resource_profile not in PROFILES:
        raise ValueError("Source384 requires an explicit common resource profile")
    return (["--source384-config", ROOT + "/" + SOURCE384_CONFIG] if enabled else []), {
        "selected": binding is not None, "enabled": enabled,
        "disabled_reason": "no_index_ablation" if binding is not None and not enabled else None,
        "config_sha256": binding["config_sha256"] if binding else None,
        "checkpoint_sha256": binding["config"]["checkpoint_sha256"] if binding else None,
        "training_steps": 0, "download_calls": 0, "execution_authority": False,
    }


class FullSupervisorAgent(BaseAgent):
    def __init__(self, *args, runtime_archive: str, arm="full", auth_json: str | None = None,
                 model_revision="", disable_intent_autoencoder: bool = False,
                 intent_requirement_contract: dict | None = None, resource_profile=None,
                 setup_cache_selection: dict | None = None, task_profile: dict | None = None,
                 provider_profile: str | None = None,
                 semantic_transport_schema: str = DEFAULT_SEMANTIC_TRANSPORT_SCHEMA,
                 coding_reply_mode: str = DEFAULT_CODING_REPLY_MODE,
                 tool_output_policy: str = DEFAULT_TOOL_OUTPUT_POLICY,
                 source_view_advisory_policy: str = DEFAULT_SOURCE_VIEW_POLICY,
                 repository_source_policy: str = DEFAULT_REPOSITORY_SOURCE_POLICY,
                 source_input_policy: str = DEFAULT_SOURCE_INPUT_POLICY,
                 query_reference_policy: str = DEFAULT_QUERY_REFERENCE_POLICY,
                 semantic_metadata_view: str = DEFAULT_SEMANTIC_METADATA_VIEW,
                 semantic_working_set: str = DEFAULT_SEMANTIC_WORKING_SET,
                 grok_coding_max_turns: int = DEFAULT_GROK_CODING_MAX_TURNS,
                 capsule_selection: str = DEFAULT_CAPSULE_SELECTION,
                 capsule_budget: str = DEFAULT_CAPSULE_BUDGET,
                 source_view_profile: str = DEFAULT_SOURCE_VIEW_PROFILE,
                 source_view_policy: dict | None = None,
                 source_edit_mode: str = DEFAULT_SOURCE_EDIT_MODE,
                 semantic_context_profile: str = DEFAULT_SEMANTIC_CONTEXT_PROFILE,
                 numerical_formalization_profile: str | None = None,
                 planning_mode: str = DEFAULT_PLANNING_MODE,
                 metadata_census_only: bool = False,
                 prompt_tokenizer_artifact: str | Path | None = None,
                 prompt_tokenizer_sha256: str | None = None,
                 max_input_tokens: int | None = None,
                 semantic_slot_policy_revision: str = "legacy@1",
                 semantic_context_projection: bool = False,
                 semantic_native_input: bool = False,
                 semantic_decision_policy_artifact: str | Path | None = None,
                 semantic_decision_policy_sha256: str | None = None,
                 semantic_decision_nomination_artifact: str | Path | None = None,
                 semantic_decision_nomination_source: str | Path | None = None,
                 semantic_decision_nomination_sha256: str | None = None,
                 disable_semantic_typed_nomination: bool = False,
                 semantic_retrieval_query_reference: bool = False,
                 semantic_decision_max_calls: int = 3,
                 semantic_decision_max_total_input_tokens: int | None = None,
                 doctor_dispatch_policy: str = "auto",
                 **kwargs):
        doctor_dispatch_arguments(arm=arm, doctor_dispatch_policy=doctor_dispatch_policy)
        prompt_options = _prompt_options(arm,
            prompt_tokenizer_artifact=prompt_tokenizer_artifact,
            prompt_tokenizer_sha256=prompt_tokenizer_sha256,
            max_input_tokens=max_input_tokens,
            semantic_slot_policy_revision=semantic_slot_policy_revision,
            semantic_context_projection=semantic_context_projection,
            semantic_native_input=semantic_native_input,
            semantic_decision_policy_artifact=semantic_decision_policy_artifact,
            semantic_decision_policy_sha256=semantic_decision_policy_sha256,
            semantic_decision_nomination_artifact=semantic_decision_nomination_artifact,
            semantic_decision_nomination_source=semantic_decision_nomination_source,
            semantic_decision_nomination_sha256=semantic_decision_nomination_sha256,
            disable_semantic_typed_nomination=disable_semantic_typed_nomination,
            **({"semantic_retrieval_query_reference": semantic_retrieval_query_reference}
               if semantic_retrieval_query_reference is not False else {}),
            semantic_decision_max_calls=semantic_decision_max_calls,
            semantic_decision_max_total_input_tokens=semantic_decision_max_total_input_tokens)
        validate_legacy_semantic_route(policyoptions=prompt_options, rendereroptions={
            'semantic_transport_schema': semantic_transport_schema,
            'semantic_metadata_view': semantic_metadata_view,
            'semantic_working_set': semantic_working_set,
            'source_view_profile': source_view_profile,
            'source_edit_mode': source_edit_mode,
            'source_view_policy': source_view_policy,
            'coding_reply_mode': coding_reply_mode,
            'tool_output_policy': tool_output_policy,
            'source_view_advisory_policy': source_view_advisory_policy,
            'repository_source_policy': repository_source_policy,
            'source_input_policy': source_input_policy,
            'query_reference_policy': query_reference_policy,
            'semantic_context_profile': semantic_context_profile,
            'capsule_selection': capsule_selection,
            'capsule_budget': capsule_budget,
            'metadata_census_only': metadata_census_only,
            "provider": resolve_provider_profile(provider_profile)["provider"],
        })
        super().__init__(*args, **kwargs)
        selected_provider = resolve_provider_profile(provider_profile)
        if self.model_name != selected_provider["model"] or arm not in {"full", "no-index"}:
            raise ValueError("the isolated comparison requires the pinned model and explicit arm")
        self.planning_mode = validate_planning_mode(planning_mode,
            intent_requirement_contract=intent_requirement_contract, task_profile=task_profile)
        self._frozen_planning_mode = self.planning_mode
        self.capsule_selection = validate_capsule_selection(capsule_selection, arm=arm)
        self.capsule_budget = validate_capsule_budget(capsule_budget, arm=arm)
        self._frozen_capsule_budget = self.capsule_budget
        if self.capsule_budget != DEFAULT_CAPSULE_BUDGET:
            from .terminal_task_profile import MULTITASK_SCHEMA
            if isinstance(task_profile, dict) and task_profile.get("schema") == MULTITASK_SCHEMA:
                raise ValueError("bounded capsules support one independent signed task")
        self.source_view_profile = validate_source_view_profile(source_view_profile, arm=arm)
        self.source_edit_mode = validate_source_edit_mode(source_edit_mode, arm=arm,
            source_view_profile=self.source_view_profile)
        self._frozen_source_edit_mode = self.source_edit_mode
        self.source_view_policy = validate_source_view_policy(source_view_policy,
            profile=self.source_view_profile, arm=arm)
        self._source_view_policy_frozen_json = json.dumps(
            source_view_observation(self.source_view_profile, self.source_view_policy),
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        self.semantic_context_profile = validate_semantic_context_profile(semantic_context_profile, arm=arm)
        from .terminal_source_view_policy import validate_source_context_combination
        validate_source_context_combination(source_view_profile=source_view_profile,
            source_view_advisory_policy=source_view_advisory_policy, repository_source_policy=repository_source_policy,
            source_input_policy=source_input_policy, query_reference_policy=query_reference_policy,
            capsule_selection=capsule_selection, semantic_context_profile=semantic_context_profile)
        self.semantic_transport_schema = validate_semantic_transport_schema(semantic_transport_schema, arm=arm)
        self.source_input_policy = validate_source_input_policy(source_input_policy,
            arm=arm, provider=selected_provider["provider"], semantic_metadata_view=semantic_metadata_view)
        self.query_reference_policy = validate_query_reference_policy(query_reference_policy,
            arm=arm, provider=selected_provider["provider"], semantic_metadata_view=semantic_metadata_view, source_input_policy=source_input_policy)
        self.repository_source_policy = validate_repository_source_policy(repository_source_policy,
            arm=arm, provider=selected_provider["provider"], source_view_advisory_policy=source_view_advisory_policy)
        self.source_view_advisory_policy = validate_source_view_advisory_policy(source_view_advisory_policy,
            arm=arm, provider=selected_provider["provider"])
        self.tool_output_policy = validate_tool_output_policy(tool_output_policy,
            arm=arm, provider=selected_provider["provider"])
        self.coding_reply_mode = validate_coding_reply_mode(coding_reply_mode,
            arm=arm, provider=selected_provider["provider"])
        self.semantic_metadata_view = validate_semantic_metadata_view(semantic_metadata_view,
            arm=arm, transport_schema=self.semantic_transport_schema, provider=selected_provider["provider"])
        self.semantic_working_set = validate_semantic_working_set(semantic_working_set, arm=arm,
            source_view_profile=self.source_view_profile, transport_schema=self.semantic_transport_schema,
            metadata_view=self.semantic_metadata_view)
        self._frozen_semantic_working_set = self.semantic_working_set
        self.grok_coding_max_turns = validate_grok_coding_max_turns(
            grok_coding_max_turns, provider=selected_provider["provider"])
        self._frozen_grok_coding_max_turns = self.grok_coding_max_turns
        if self.grok_coding_max_turns != DEFAULT_GROK_CODING_MAX_TURNS and metadata_census_only:
            raise ValueError("nondefault coding turns require generation, not metadata census")
        if type(metadata_census_only) is not bool:
            raise ValueError("explicit Boolean metadata census selection required")
        if metadata_census_only and self.source_edit_mode != DEFAULT_SOURCE_EDIT_MODE:
            raise ValueError("mapped source edits require a coding worker, not metadata census")
        if metadata_census_only and self.planning_mode != DEFAULT_PLANNING_MODE:
            raise ValueError("live metadata census requires model planning")
        if metadata_census_only and (arm != "full" or self.semantic_metadata_view not in {"common-bindings@1", "common-bindings-instruction@1"}
                or self.coding_reply_mode != "ordinary-completion@1"):
            raise ValueError("live census requires the full common-bindings ordinary-completion route")
        if metadata_census_only and self.source_input_policy != DEFAULT_SOURCE_INPUT_POLICY:
            raise ValueError("source input policy requires an executed coding route")
        if metadata_census_only and self.query_reference_policy != DEFAULT_QUERY_REFERENCE_POLICY:
            raise ValueError("query reference policy requires an executed coding route")
        if metadata_census_only and self.repository_source_policy != DEFAULT_REPOSITORY_SOURCE_POLICY:
            raise ValueError("repository source policy requires an executed coding route")
        if metadata_census_only and self.source_view_advisory_policy != DEFAULT_SOURCE_VIEW_POLICY:
            raise ValueError("source view policy requires an executed coding route")
        if metadata_census_only and self.tool_output_policy != DEFAULT_TOOL_OUTPUT_POLICY:
            raise ValueError("tool output policy requires an executed coding route")
        self.metadata_census_only = metadata_census_only
        self.numerical_formalization_profile = validate_numerical_profile(numerical_formalization_profile)
        if type(disable_intent_autoencoder) is not bool:
            raise ValueError("Intent ablation switch must be boolean")
        self.runtime_archive = Path(runtime_archive).resolve(strict=True)
        self.provider_profile = provider_profile
        self.auth_json = Path(auth_json or Path.home() / (".codex/auth.json"
            if selected_provider["provider"] == "codex_cli" else ".grok/auth.json"))
        self.arm = arm
        self.model_revision = model_revision
        self.prompt_tokenizer_artifact = prompt_tokenizer_artifact
        self.prompt_tokenizer_sha256 = prompt_tokenizer_sha256
        self.max_input_tokens = max_input_tokens
        self.semantic_slot_policy_revision = semantic_slot_policy_revision
        self.semantic_context_projection = semantic_context_projection
        self.semantic_native_input = semantic_native_input
        self.semantic_decision_policy_artifact = semantic_decision_policy_artifact
        self.semantic_decision_policy_sha256 = semantic_decision_policy_sha256
        self.semantic_decision_nomination_artifact = semantic_decision_nomination_artifact
        self.semantic_decision_nomination_source = semantic_decision_nomination_source
        self.semantic_decision_nomination_sha256 = semantic_decision_nomination_sha256
        self.disable_semantic_typed_nomination = disable_semantic_typed_nomination
        self.semantic_retrieval_query_reference = semantic_retrieval_query_reference
        self.semantic_decision_max_calls = semantic_decision_max_calls
        self.semantic_decision_max_total_input_tokens = semantic_decision_max_total_input_tokens
        self.doctor_dispatch_policy = doctor_dispatch_policy
        if resource_profile not in (None, *PROFILES):
            raise ValueError("unknown benchmark resource profile")
        self.resource_profile = resource_profile
        from .terminal_setup_cache_advice import validate_setup_cache_selection, validate_setup_cache_prerequisites
        if selected_provider["provider"] != "codex_cli" and setup_cache_selection is not None:
            raise ValueError("Codex setup cache is incompatible with Grok")
        validate_setup_cache_selection(self.runtime_archive, setup_cache_selection)
        validate_setup_cache_prerequisites(setup_cache_selection, install_codex=selected_provider["provider"] == "codex_cli",
            auth_json=self.auth_json, arm=arm, resource_profile=resource_profile)
        self.setup_cache_selection = json.loads(json.dumps(setup_cache_selection))
        self.disable_intent_autoencoder = disable_intent_autoencoder
        if intent_requirement_contract is not None:
            if type(intent_requirement_contract) is not dict:
                raise ValueError("Intent coverage requires an explicit requirement contract object")
            encoded = json.dumps(intent_requirement_contract, allow_nan=False)
            if len(encoded.encode("utf-8")) > 4 * 1024 * 1024:
                raise ValueError("Intent requirement contract exceeds its byte bound")
            intent_requirement_contract = json.loads(encoded)
        self.intent_requirement_contract = intent_requirement_contract
        from .terminal_task_profile import validate_task_profile
        self.task_profile = validate_task_profile(task_profile) if task_profile is not None else None
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def name():
        return "ipfs-admitted-supervisor"

    def version(self):
        selected = resolve_provider_profile(getattr(self, "provider_profile", None))
        return "isolated-native-v1+" + selected["provider"].removesuffix("_cli") + "-" + selected["cli_version"]

    def _checked_planning_mode(self):
        mode = validate_planning_mode(getattr(self, "planning_mode", DEFAULT_PLANNING_MODE),
            intent_requirement_contract=getattr(self, "intent_requirement_contract", None),
            task_profile=getattr(self, "task_profile", None))
        if mode != DEFAULT_PLANNING_MODE and getattr(self, "metadata_census_only", False):
            raise ValueError("live metadata census requires model planning")
        if mode != getattr(self, "_frozen_planning_mode", DEFAULT_PLANNING_MODE):
            raise ValueError("prepared planning mode changed")
        return mode

    def _checked_capsule_budget(self):
        value = validate_capsule_budget(getattr(self, "capsule_budget", DEFAULT_CAPSULE_BUDGET),
            arm=self.arm)
        if value != getattr(self, "_frozen_capsule_budget", DEFAULT_CAPSULE_BUDGET):
            raise ValueError("prepared capsule budget changed")
        if value != DEFAULT_CAPSULE_BUDGET:
            from .terminal_task_profile import MULTITASK_SCHEMA
            task_profile = getattr(self, "task_profile", None)
            if isinstance(task_profile, dict) and task_profile.get("schema") == MULTITASK_SCHEMA:
                raise ValueError("bounded capsules support one independent signed task")
        return value

    def _checked_grok_coding_max_turns(self):
        value = validate_grok_coding_max_turns(
            getattr(self, "grok_coding_max_turns", DEFAULT_GROK_CODING_MAX_TURNS),
            provider=resolve_provider_profile(getattr(self, "provider_profile", None))["provider"])
        if value != DEFAULT_GROK_CODING_MAX_TURNS and getattr(self, "metadata_census_only", False):
            raise ValueError("nondefault coding turns require generation, not metadata census")
        if value != getattr(self, "_frozen_grok_coding_max_turns", DEFAULT_GROK_CODING_MAX_TURNS):
            raise ValueError("prepared Grok coding turn bound changed")
        return value

    def _checked_semantic_working_set(self):
        mode = validate_semantic_working_set(getattr(self, "semantic_working_set", DEFAULT_SEMANTIC_WORKING_SET),
            arm=self.arm, source_view_profile=getattr(self, "source_view_profile", DEFAULT_SOURCE_VIEW_PROFILE),
            transport_schema=getattr(self, "semantic_transport_schema", DEFAULT_SEMANTIC_TRANSPORT_SCHEMA),
            metadata_view=getattr(self, "semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW))
        if mode != DEFAULT_SEMANTIC_WORKING_SET and getattr(self, "metadata_census_only", False):
            raise ValueError("semantic working set requires a coding worker")
        if mode != getattr(self, "_frozen_semantic_working_set", DEFAULT_SEMANTIC_WORKING_SET):
            raise ValueError("prepared semantic working set changed")
        return mode

    def _checked_source_edit_mode(self, profile):
        mode = validate_source_edit_mode(getattr(self, "source_edit_mode", DEFAULT_SOURCE_EDIT_MODE),
            arm=self.arm, source_view_profile=profile)
        if mode != getattr(self, "_frozen_source_edit_mode", DEFAULT_SOURCE_EDIT_MODE):
            raise ValueError("prepared source edit mode changed")
        if mode != DEFAULT_SOURCE_EDIT_MODE and getattr(self, "metadata_census_only", False):
            raise ValueError("mapped source edits require a coding worker, not metadata census")
        return mode

    def _checked_source_view_policy(self, profile):
        selected = validate_source_view_policy(getattr(self, "source_view_policy", None),
            profile=profile, arm=self.arm)
        frozen = getattr(self, "_source_view_policy_frozen_json", None)
        current = json.dumps(source_view_observation(profile, selected), sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        if frozen is not None and frozen != current:
            raise ValueError("prepared source view policy changed")
        return selected

    async def setup(self, environment):
        from .terminal_setup_cache_advice import (
            validate_setup_cache_selection, validate_setup_cache_prerequisites, apply_setup_cache_advice,
        )
        selection = getattr(self, "setup_cache_selection", None)
        manifest, _ = validate_setup_cache_selection(self.runtime_archive, selection)
        from .terminal_worker_capability import require_worker_capability
        require_worker_capability(manifest, getattr(self, "resource_profile", None))
        require_planning_archive(manifest, self._checked_planning_mode())
        require_numerical_archive(manifest, getattr(self, "numerical_formalization_profile", None))
        numerical_selection_asset_arguments(manifest, self.arm, getattr(self, "resource_profile", None))
        if selection is not None and manifest.get("numerical_selection_384") is not None:
            raise ValueError("setup cache is not qualified for learned numerical selection")
        require_semantic_transport_archive(manifest,
            getattr(self, "semantic_transport_schema", DEFAULT_SEMANTIC_TRANSPORT_SCHEMA))
        selected_provider = require_runtime_provider_profile(manifest, getattr(self, "provider_profile", None))
        reply_mode = validate_coding_reply_mode(getattr(self, "coding_reply_mode", DEFAULT_CODING_REPLY_MODE),
            arm=self.arm, provider=selected_provider["provider"])
        require_coding_reply_archive(manifest, reply_mode)
        tool_output = validate_tool_output_policy(
            getattr(self, "tool_output_policy", DEFAULT_TOOL_OUTPUT_POLICY), arm=self.arm,
            provider=resolve_provider_profile(getattr(self, "provider_profile", None))["provider"])
        require_tool_output_archive(manifest, tool_output)
        source_view_advisory = validate_source_view_advisory_policy(
            getattr(self, "source_view_advisory_policy", DEFAULT_SOURCE_VIEW_POLICY), arm=self.arm,
            provider=resolve_provider_profile(getattr(self, "provider_profile", None))["provider"])
        require_source_view_advisory_archive(manifest, source_view_advisory)
        repository_source = validate_repository_source_policy(
            getattr(self, "repository_source_policy", DEFAULT_REPOSITORY_SOURCE_POLICY), arm=self.arm,
            provider=resolve_provider_profile(getattr(self, "provider_profile", None))["provider"], source_view_advisory_policy=source_view_advisory)
        require_repository_source_archive(manifest, repository_source, archive=self.runtime_archive / "runtime.tar.gz")
        source_input = validate_source_input_policy(
            getattr(self, "source_input_policy", DEFAULT_SOURCE_INPUT_POLICY), arm=self.arm,
            provider=resolve_provider_profile(getattr(self, "provider_profile", None))["provider"],
            semantic_metadata_view=getattr(self, "semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW))
        query_reference = validate_query_reference_policy(
            getattr(self, "query_reference_policy", DEFAULT_QUERY_REFERENCE_POLICY), arm=self.arm,
            provider=resolve_provider_profile(getattr(self, "provider_profile", None))["provider"],
            semantic_metadata_view=getattr(self, "semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW),
            source_input_policy=getattr(self, "source_input_policy", DEFAULT_SOURCE_INPUT_POLICY))
        require_source_input_archive(manifest, source_input)
        require_query_reference_archive(manifest, query_reference)
        metadata_view = validate_semantic_metadata_view(
            getattr(self, "semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW), arm=self.arm,
            transport_schema=getattr(self, "semantic_transport_schema", DEFAULT_SEMANTIC_TRANSPORT_SCHEMA),
            provider=selected_provider["provider"])
        require_semantic_metadata_archive(manifest, metadata_view)
        working_set = self._checked_semantic_working_set()
        require_semantic_working_set_archive(manifest, working_set)
        grok_turns = self._checked_grok_coding_max_turns()
        require_grok_coding_archive(manifest, grok_turns)
        capsule_selection = validate_capsule_selection(
            getattr(self, "capsule_selection", DEFAULT_CAPSULE_SELECTION), arm=self.arm)
        require_capsule_selection_archive(manifest, capsule_selection)
        capsule_budget = self._checked_capsule_budget()
        require_capsule_budget_archive(manifest, capsule_budget)
        source_view = validate_source_view_profile(
            getattr(self, "source_view_profile", DEFAULT_SOURCE_VIEW_PROFILE), arm=self.arm)
        self._checked_source_view_policy(source_view)
        require_source_edit_archive(manifest, self._checked_source_edit_mode(source_view))
        require_source_view_archive(manifest, source_view, getattr(self, "resource_profile", None))
        semantic_profile = validate_semantic_context_profile(
            getattr(self, "semantic_context_profile", DEFAULT_SEMANTIC_CONTEXT_PROFILE), arm=self.arm)
        require_semantic_context_archive(manifest, semantic_profile)
        require_retrieval_revision(manifest, self.arm, getattr(self, "model_revision", ""))
        validate_header_planning_selection(validate_source384_binding(manifest),
            getattr(self, "intent_requirement_contract", None), self.arm)
        validate_setup_cache_prerequisites(selection, install_codex=selected_provider["provider"] == "codex_cli", auth_json=self.auth_json,
            arm=self.arm, resource_profile=getattr(self, "resource_profile", None))
        task_profile = getattr(self, "task_profile", None)
        if task_profile is not None:
            from .terminal_task_bootstrap import bootstrap_task_repository
            self.task_bootstrap = await bootstrap_task_repository(environment, profile=task_profile,
                output=self.logs_dir / "task-bootstrap")
        provider_options = ({"provider": "grok_cli", "install_codex": False}
            if selected_provider["provider"] == "grok_cli" else {})
        await deploy_supervisor(environment, archive_dir=self.runtime_archive,
                                output=self.logs_dir / "deployment", auth_json=self.auth_json, **provider_options,
                                **({"repository_source_policy": repository_source}
                                   if repository_source != DEFAULT_REPOSITORY_SOURCE_POLICY else {}),
                                **({"isolated_uv_bootstrap": True} if task_profile is not None else {}),
                                **({"setup_cache_selection": selection} if selection is not None else {}))
        boundary = self.logs_dir / "worker-boundary"
        deployed_boundary = await deploy_worker_boundary(environment, output=boundary,
            **({"provider": "grok_cli"} if selected_provider["provider"] == "grok_cli" else {}))
        if getattr(self, "metadata_census_only", False):
            if validate_source384_binding(manifest) is None:
                raise ValueError("live census requires current Source384 assets")
            # This is the actual independent namespace/permission/termination
            # qualification, protected by container root before owner use.
            custody = self.logs_dir / "census-worker-qualification.json"
            custody.write_text(json.dumps(deployed_boundary, sort_keys=True, allow_nan=False) + "\n")
            target = ROOT + "/census-worker-qualification.json"
            await environment.upload_file(custody, target)
            protected = await environment.exec(command="chown root:root " + target + " && chmod 0444 " + target,
                                               user="root", cwd="/", timeout_sec=5)
            if protected.return_code:
                raise RuntimeError("live census qualification protection failed")
        if selected_provider["provider"] == "grok_cli":
            from .terminal_grok_deployment import require_grok_credential_readiness
            await require_grok_credential_readiness(environment,
                output=self.logs_dir / "provider-readiness")
        if selection is not None:
            self.setup_cache_receipt = await apply_setup_cache_advice(environment,
                archive_dir=self.runtime_archive, expected=selection,
                boundary_output=boundary, output=self.logs_dir / "setup-cache",
                resource_profile=getattr(self, "resource_profile", None))

    async def run(self, instruction: str, environment, context: AgentContext):
        run_started = monotonic()
        doctor_arguments = doctor_dispatch_arguments(arm=self.arm,
            doctor_dispatch_policy=getattr(self, "doctor_dispatch_policy", "auto"))
        nomination_source = getattr(self, "semantic_decision_nomination_source", None)
        nomination_sha256 = getattr(self, "semantic_decision_nomination_sha256", None)
        prompt_options = _prompt_options(self.arm,
            prompt_tokenizer_artifact=getattr(self, "prompt_tokenizer_artifact", None),
            prompt_tokenizer_sha256=getattr(self, "prompt_tokenizer_sha256", None),
            max_input_tokens=getattr(self, "max_input_tokens", None),
            semantic_slot_policy_revision=getattr(self, "semantic_slot_policy_revision", "legacy@1"),
            semantic_context_projection=getattr(self, "semantic_context_projection", False),
            semantic_native_input=getattr(self, "semantic_native_input", False),
            semantic_decision_policy_artifact=getattr(self, "semantic_decision_policy_artifact", None),
            semantic_decision_policy_sha256=getattr(self, "semantic_decision_policy_sha256", None),
            semantic_decision_nomination_artifact=getattr(self, "semantic_decision_nomination_artifact", None),
            semantic_decision_nomination_source=nomination_source,
            semantic_decision_nomination_sha256=nomination_sha256,
            disable_semantic_typed_nomination=getattr(self, "disable_semantic_typed_nomination", False),
            **({"semantic_retrieval_query_reference": getattr(self, "semantic_retrieval_query_reference", False)}
               if getattr(self, "semantic_retrieval_query_reference", False) is not False else {}),
            semantic_decision_max_calls=getattr(self, "semantic_decision_max_calls", 3),
            semantic_decision_max_total_input_tokens=getattr(self, "semantic_decision_max_total_input_tokens", None))
        validate_legacy_semantic_route(policyoptions=prompt_options, rendereroptions={
            'semantic_transport_schema': getattr(self, 'semantic_transport_schema', 'supervisor-semantic-router-input@1'),
            'semantic_metadata_view': getattr(self, 'semantic_metadata_view', 'legacy'),
            'semantic_working_set': getattr(self, 'semantic_working_set', 'legacy'),
            'source_view_profile': getattr(self, 'source_view_profile', 'disabled'),
            'source_edit_mode': getattr(self, 'source_edit_mode', 'disabled'),
            'source_view_policy': getattr(self, 'source_view_policy', None),
            'coding_reply_mode': getattr(self, 'coding_reply_mode', 'legacy'),
            'tool_output_policy': getattr(self, 'tool_output_policy', 'legacy'),
            'source_view_advisory_policy': getattr(self, 'source_view_advisory_policy', 'legacy'),
            'repository_source_policy': getattr(self, 'repository_source_policy', 'legacy'),
            'source_input_policy': getattr(self, 'source_input_policy', 'legacy'),
            'query_reference_policy': getattr(self, 'query_reference_policy', 'legacy'),
            'semantic_context_profile': getattr(self, 'semantic_context_profile', 'legacy-bounded@1'),
            'capsule_selection': getattr(self, 'capsule_selection', 'lexical-query-symbol-name@1'),
            'capsule_budget': getattr(self, 'capsule_budget', 'legacy'),
            'metadata_census_only': getattr(self, 'metadata_census_only', False),
            "provider": resolve_provider_profile(getattr(self, "provider_profile", None))["provider"],
        })
        semantic_arguments = semantic_context_arguments(
            semantic_slot_policy_revision=getattr(self, "semantic_slot_policy_revision", "legacy@1"),
            semantic_context_projection=getattr(self, "semantic_context_projection", False),
            **({"semantic_native_input": True} if getattr(self, "semantic_native_input", False) else {}))
        decision_arguments = [] if nomination_source is not None else semantic_decision_arguments(
            semantic_decision_policy_artifact=getattr(self, "semantic_decision_policy_artifact", None),
            semantic_decision_policy_sha256=getattr(self, "semantic_decision_policy_sha256", None),
            semantic_decision_nomination_artifact=getattr(self, "semantic_decision_nomination_artifact", None),
            semantic_decision_nomination_sha256=nomination_sha256,
            disable_semantic_typed_nomination=getattr(self, "disable_semantic_typed_nomination", False),
            **({"semantic_retrieval_query_reference": True}
               if getattr(self, "semantic_retrieval_query_reference", False) else {}),
            semantic_decision_max_calls=getattr(self, "semantic_decision_max_calls", 3),
            semantic_decision_max_total_input_tokens=getattr(self, "semantic_decision_max_total_input_tokens", None))

        from .terminal_setup_cache_advice import validate_setup_cache_selection
        manifest, _ = validate_setup_cache_selection(self.runtime_archive,
            getattr(self, "setup_cache_selection", None))
        planning_mode = self._checked_planning_mode()
        require_planning_archive(manifest, planning_mode)
        retrieval_revision = require_retrieval_revision(manifest, self.arm,
            getattr(self, "model_revision", ""))
        transport = validate_semantic_transport_schema(
            getattr(self, "semantic_transport_schema", DEFAULT_SEMANTIC_TRANSPORT_SCHEMA), arm=self.arm)
        require_semantic_transport_archive(manifest, transport)
        reply_mode = validate_coding_reply_mode(getattr(self, "coding_reply_mode", DEFAULT_CODING_REPLY_MODE),
            arm=self.arm, provider=resolve_provider_profile(getattr(self, "provider_profile", None))["provider"])
        require_coding_reply_archive(manifest, reply_mode)
        tool_output = validate_tool_output_policy(
            getattr(self, "tool_output_policy", DEFAULT_TOOL_OUTPUT_POLICY), arm=self.arm,
            provider=resolve_provider_profile(getattr(self, "provider_profile", None))["provider"])
        require_tool_output_archive(manifest, tool_output)
        source_view_advisory = validate_source_view_advisory_policy(
            getattr(self, "source_view_advisory_policy", DEFAULT_SOURCE_VIEW_POLICY), arm=self.arm,
            provider=resolve_provider_profile(getattr(self, "provider_profile", None))["provider"])
        require_source_view_advisory_archive(manifest, source_view_advisory)
        repository_source = validate_repository_source_policy(
            getattr(self, "repository_source_policy", DEFAULT_REPOSITORY_SOURCE_POLICY), arm=self.arm,
            provider=resolve_provider_profile(getattr(self, "provider_profile", None))["provider"], source_view_advisory_policy=source_view_advisory)
        require_repository_source_archive(manifest, repository_source, archive=self.runtime_archive / "runtime.tar.gz")
        source_input = validate_source_input_policy(
            getattr(self, "source_input_policy", DEFAULT_SOURCE_INPUT_POLICY), arm=self.arm,
            provider=resolve_provider_profile(getattr(self, "provider_profile", None))["provider"],
            semantic_metadata_view=getattr(self, "semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW))
        query_reference = validate_query_reference_policy(
            getattr(self, "query_reference_policy", DEFAULT_QUERY_REFERENCE_POLICY), arm=self.arm,
            provider=resolve_provider_profile(getattr(self, "provider_profile", None))["provider"],
            semantic_metadata_view=getattr(self, "semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW),
            source_input_policy=getattr(self, "source_input_policy", DEFAULT_SOURCE_INPUT_POLICY))
        require_source_input_archive(manifest, source_input)
        require_query_reference_archive(manifest, query_reference)
        metadata_view = validate_semantic_metadata_view(
            getattr(self, "semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW), arm=self.arm,
            transport_schema=transport,
            provider=resolve_provider_profile(getattr(self, "provider_profile", None))["provider"])
        require_semantic_metadata_archive(manifest, metadata_view)
        working_set = self._checked_semantic_working_set()
        require_semantic_working_set_archive(manifest, working_set)
        grok_turns = self._checked_grok_coding_max_turns()
        require_grok_coding_archive(manifest, grok_turns)
        capsule_selection = validate_capsule_selection(
            getattr(self, "capsule_selection", DEFAULT_CAPSULE_SELECTION), arm=self.arm)
        require_capsule_selection_archive(manifest, capsule_selection)
        capsule_budget = self._checked_capsule_budget()
        require_capsule_budget_archive(manifest, capsule_budget)
        source_view = validate_source_view_profile(
            getattr(self, "source_view_profile", DEFAULT_SOURCE_VIEW_PROFILE), arm=self.arm)
        source_policy = self._checked_source_view_policy(source_view)
        from .terminal_source_view_policy import validate_source_context_combination
        validate_source_context_combination(source_view_profile=source_view,
            source_view_advisory_policy=source_view_advisory, repository_source_policy=repository_source,
            source_input_policy=source_input, query_reference_policy=query_reference,
            capsule_selection=capsule_selection,
            semantic_context_profile=getattr(self, "semantic_context_profile", DEFAULT_SEMANTIC_CONTEXT_PROFILE))
        source_edit = self._checked_source_edit_mode(source_view)
        require_source_edit_archive(manifest, source_edit)
        require_source_view_archive(manifest, source_view, getattr(self, "resource_profile", None))
        semantic_profile = validate_semantic_context_profile(
            getattr(self, "semantic_context_profile", DEFAULT_SEMANTIC_CONTEXT_PROFILE), arm=self.arm)
        require_semantic_context_archive(manifest, semantic_profile)
        validate_header_planning_selection(validate_source384_binding(manifest),
            getattr(self, "intent_requirement_contract", None), self.arm)
        from .terminal_worker_capability import require_worker_capability
        require_worker_capability(manifest, getattr(self, "resource_profile", None))
        numerical_profile = validate_numerical_profile(getattr(self, "numerical_formalization_profile", None))
        require_numerical_archive(manifest, numerical_profile)
        numerical_arguments, numerical_assets = numerical_selection_asset_arguments(
            manifest, self.arm, getattr(self, "resource_profile", None))
        if getattr(self, "setup_cache_selection", None) is not None and numerical_assets["selected"]:
            raise ValueError("setup cache is not qualified for learned numerical selection")
        public = self.logs_dir / "instruction.md"
        public.write_text(instruction)
        instruction_path = ROOT + "/instruction.md"
        uploads = [("instruction_upload", public, instruction_path)]
        # Owner state contains signing keys. Export only the explicit public
        # result and bounded process diagnostics; never download the state tree.
        run_id = uuid.uuid4().hex
        state = ROOT + "/state/benchmark-" + run_id
        report_path = state + "-result.json"
        profile = getattr(self, "resource_profile", None)
        budget = execution_budget(profile)
        argv = [PYTHON, "-P", "-m", "benchmarks.agent_supervisor.container_coding.terminal_container_supervisor",
                "--instruction", instruction_path, "--state", state,
                "--arm", self.arm, "--timeout-seconds", str(budget["driver_seconds"])]
        argv += prompt_tokenizer_arguments(
            prompt_tokenizer_artifact=getattr(self, "prompt_tokenizer_artifact", None),
            prompt_tokenizer_sha256=getattr(self, "prompt_tokenizer_sha256", None),
            max_input_tokens=getattr(self, "max_input_tokens", None))
        argv += semantic_arguments + decision_arguments + doctor_arguments
        if getattr(self, "semantic_retrieval_query_reference", False):
            argv += ["--semantic-retrieval-query-reference"]
        provider_profile = getattr(self, "provider_profile", None)
        if provider_profile is not None:
            argv += ["--provider-profile", provider_profile]
        if source_edit != DEFAULT_SOURCE_EDIT_MODE:
            argv += ["--source-edit-mode", source_edit]
        if working_set != DEFAULT_SEMANTIC_WORKING_SET:
            argv += ["--semantic-working-set", working_set]
        if grok_turns != DEFAULT_GROK_CODING_MAX_TURNS:
            argv += ["--grok-coding-max-turns", str(grok_turns)]
        if planning_mode != DEFAULT_PLANNING_MODE:
            argv += ["--planning-mode", planning_mode]
        if numerical_profile is not None:
            argv += ["--numerical-formalization-profile", numerical_profile]
        if transport != DEFAULT_SEMANTIC_TRANSPORT_SCHEMA:
            argv += ["--semantic-transport-schema", transport]
        if tool_output != DEFAULT_TOOL_OUTPUT_POLICY:
            argv += ["--tool-output-policy", tool_output]
        if repository_source != DEFAULT_REPOSITORY_SOURCE_POLICY:
            argv += ["--repository-source-policy", repository_source]
        if source_view_advisory != DEFAULT_SOURCE_VIEW_POLICY:
            argv += ["--source-view-advisory-policy", source_view_advisory]
        if source_input != DEFAULT_SOURCE_INPUT_POLICY:
            argv += ["--source-input-policy", source_input]
        if query_reference != DEFAULT_QUERY_REFERENCE_POLICY:
            argv += ["--query-reference-policy", query_reference]
        if reply_mode != DEFAULT_CODING_REPLY_MODE:
            argv += ["--coding-reply-mode", reply_mode]
        if metadata_view != DEFAULT_SEMANTIC_METADATA_VIEW:
            argv += ["--semantic-metadata-view", metadata_view]
        if capsule_selection != DEFAULT_CAPSULE_SELECTION:
            argv += ["--capsule-selection", capsule_selection]
        if capsule_budget != DEFAULT_CAPSULE_BUDGET:
            argv += ["--capsule-budget", capsule_budget]
        if source_view != DEFAULT_SOURCE_VIEW_PROFILE:
            argv += ["--source-view-profile", source_view]
        argv += source_view_policy_argv(source_policy, profile=source_view)
        if semantic_profile != DEFAULT_SEMANTIC_CONTEXT_PROFILE:
            argv += ["--semantic-context-profile", semantic_profile]
        if getattr(self, "metadata_census_only", False):
            argv += ["--metadata-census-only"]
        task_profile = getattr(self, "task_profile", None)
        if task_profile is not None:
            from .terminal_task_profile import validate_task_profile
            task_profile = validate_task_profile(task_profile, instruction=instruction)
            artifact = self.logs_dir / "task-profile.json"
            artifact.write_text(json.dumps(task_profile, sort_keys=True, allow_nan=False) + "\n")
            uploads.append(("task_profile_upload", artifact, ROOT + "/task-profile.json"))
            argv += ["--task-profile", ROOT + "/task-profile.json"]
        if profile is not None:
            argv += ["--resource-profile", profile]
        requirement_contract = getattr(self, "intent_requirement_contract", None)
        if requirement_contract is not None:
            artifact = self.logs_dir / "intent-requirements.json"
            artifact.write_text(json.dumps(requirement_contract, sort_keys=True,
                                           separators=(",", ":"), allow_nan=False) + "\n")
            requirement_path = ROOT + "/intent-requirements.json"
            uploads.append(("requirement_upload", artifact, requirement_path))
            argv += ["--intent-requirement-contract", requirement_path]
        if retrieval_revision:
            argv += ["--model-snapshot", ROOT + "/models/embedding", "--model-revision", retrieval_revision]
        asset_arguments, asset_observation = security_asset_arguments(manifest, self.arm)
        argv += asset_arguments
        intent_arguments, intent_observation = intent_asset_arguments(manifest,
            enabled=not self.disable_intent_autoencoder)
        argv += intent_arguments
        source384_arguments, source384_observation = source384_asset_arguments(
            manifest, self.arm, getattr(self, "resource_profile", None))
        argv += source384_arguments
        argv += numerical_arguments
        context.metadata = {"adapter_run_id": run_id, "arm": self.arm, "task_completed": False, "official_reward": None,
                            "provider_profile": resolve_provider_profile(provider_profile),
                            **semantic_transport_selection(transport),
                            **coding_reply_selection(reply_mode),
                            **tool_output_selection(tool_output),
                            **source_view_advisory_selection(source_view_advisory),
                            **repository_source_selection(repository_source),
                            **source_input_selection(source_input),
                            **query_reference_selection(query_reference),
                            **semantic_metadata_observation(metadata_view),
                            **capsule_selection_observation(capsule_selection),
                            **capsule_budget_observation(capsule_budget),
                            **source_view_observation(source_view, source_policy),
                            **source_edit_observation(source_edit),
                            **semantic_context_selection(semantic_profile),
                            **semantic_working_set_observation(working_set),
                            **grok_coding_selection(grok_turns),
                            **numerical_observation(numerical_profile, self.arm),
                            **planning_observation(planning_mode),
                            **_intent_selection(requirement_contract, planning_mode=planning_mode),
                            "runtime_archive_sha256": manifest["archive_sha256"],
                            "security_training_assets": asset_observation,
                            "security_model_assets": asset_observation,
                            "intent_model_assets": intent_observation,
                            "source384_assets": source384_observation,
                            "numerical_selection_assets": numerical_assets,
                            "resource_profile": profile, "execution_budget": budget,
                            "admission_environment": admission_environment(profile),
                            **({"setup_cache": self.setup_cache_receipt} if hasattr(self, "setup_cache_receipt") else {}),
                            "planning_and_cold_index_charged_to_agent_time": True}
        if nomination_source is not None or getattr(self, "semantic_decision_nomination_artifact", None) is not None:
            context.metadata["disable_semantic_typed_nomination"] = getattr(
                self, "disable_semantic_typed_nomination", False)
            context.metadata["semantic_decision_typed_nomination_expected"] = not getattr(
                self, "disable_semantic_typed_nomination", False)
        if getattr(self, "semantic_native_input", False):
            context.metadata["semantic_native_input_requested"] = True
        if getattr(self, "semantic_retrieval_query_reference", False):
            context.metadata["semantic_retrieval_query_reference_requested"] = True
        if task_profile is not None:
            context.metadata["task_profile"] = task_profile
            context.metadata["public_output_capture"] = "not_selected_for_generic_profile"
        context.n_input_tokens = context.n_cache_tokens = context.n_output_tokens = None
        deadline = _AdapterRunDeadline(context.metadata, resource_profile=profile, started=run_started)
        error = None
        report = None
        driver_started = False
        nomination_stage = None
        try:
            if nomination_source is not None:
                nomination_stage = "nomination_capture"
                delivery = {
                    "schema": "terminal-semantic-decision-nomination-delivery@1",
                    "status": "preparing", "stage": nomination_stage, "published": False,
                    "sha256": nomination_sha256,
                    "provider_called": False, "completion_authority": False,
                }
                context.metadata["semantic_decision_nomination_delivery"] = delivery
                raw_nomination = read_nomination_source(
                    source=nomination_source, expected_sha256=nomination_sha256)
                snapshot_directory = self.logs_dir.absolute() / ("decision-nomination-" + run_id)
                snapshot_directory.mkdir(mode=0o755)
                snapshot = capture_nomination_snapshot(source=nomination_source,
                    expected_sha256=nomination_sha256, directory=snapshot_directory)
                if verify_nomination_snapshot(snapshot, directory=snapshot_directory) != raw_nomination:
                    raise ValueError("captured nomination changed before staging")
                locations = nomination_delivery_arguments(root=ROOT, run_id=run_id,
                    expected_sha256=nomination_sha256, expected_bytes=len(raw_nomination))
                expected_receipt = {
                    "schema": "terminal-semantic-decision-nomination-delivery@1",
                    "artifact": locations["artifact"], "sha256": nomination_sha256,
                    "bytes": len(raw_nomination), "provider_called": False,
                    "completion_authority": False,
                }
                delivery.update(artifact=locations["artifact"], bytes=len(raw_nomination))

                async def execute_nomination(operation):
                    command = nomination_delivery_command(python=PYTHON,
                        operation=operation, root=ROOT, run_id=run_id,
                        expected_sha256=nomination_sha256, expected_bytes=len(raw_nomination))
                    result = await environment.exec(command=shlex.join(command), cwd=ROOT,
                        user="root", env=runtime_environment(), timeout_sec=15)
                    _retain_nomination_publisher_result(result, operation=operation,
                        artifact=snapshot_directory / ("publisher-" + operation + ".json"))
                    return _nomination_delivery_receipt(result, dict(expected_receipt,
                        operation=operation, published=operation == "publish"))

                async def upload_nomination():
                    # Verify immediately before Harbor opens the immutable
                    # snapshot. The root publisher rechecks the uploaded bytes.
                    if verify_nomination_snapshot(snapshot, directory=snapshot_directory) != raw_nomination:
                        raise ValueError("captured nomination changed before upload")
                    return await environment.upload_file(Path(snapshot["artifact"]), locations["temporary"])

                nomination_stage = "nomination_prepare"
                delivery["stage"] = nomination_stage
                await deadline.stage(nomination_stage, lambda: execute_nomination("prepare"),
                    maximum=15, reserve=ADAPTER_FINALIZATION_SECONDS)
                nomination_stage = "nomination_upload"
                delivery["stage"] = nomination_stage
                await deadline.stage(nomination_stage, upload_nomination,
                    maximum=15, reserve=ADAPTER_FINALIZATION_SECONDS)
                nomination_stage = "nomination_publish"
                delivery["stage"] = nomination_stage
                receipt = await deadline.stage(nomination_stage, lambda: execute_nomination("publish"),
                    maximum=15, reserve=ADAPTER_FINALIZATION_SECONDS)
                delivery.update(receipt, status="published")
                argv += semantic_decision_arguments(
                    semantic_decision_nomination_artifact=receipt["artifact"],
                    semantic_decision_nomination_sha256=nomination_sha256,
                    disable_semantic_typed_nomination=getattr(self, "disable_semantic_typed_nomination", False),
                    **({"semantic_retrieval_query_reference": True}
                       if getattr(self, "semantic_retrieval_query_reference", False) else {}),
                    semantic_decision_max_calls=getattr(self, "semantic_decision_max_calls", 3),
                    semantic_decision_max_total_input_tokens=getattr(
                        self, "semantic_decision_max_total_input_tokens", None))
                nomination_stage = None
            for stage, source, target in uploads:
                await deadline.stage(stage,
                    lambda source=source, target=target: environment.upload_file(source, target),
                    reserve=ADAPTER_FINALIZATION_SECONDS)
            try:
                if task_profile is None:
                    context.metadata["public_input_capture"] = await deadline.stage(
                        "public_input_capture", lambda: capture_public_inputs(environment, self.logs_dir),
                        maximum=PUBLIC_EXPORT_SECONDS, reserve=ADAPTER_FINALIZATION_SECONDS)
            except Exception as exc:
                context.metadata["public_input_capture_error"] = type(exc).__name__
            driver_seconds = deadline.driver_seconds()
            argv[argv.index("--timeout-seconds") + 1] = str(driver_seconds)
            driver_started = True
            result = await deadline.stage("driver", lambda: environment.exec(
                command=shlex.join(argv), cwd="/app", user="supervisor",
                env={**runtime_environment(), **admission_environment(profile)},
                timeout_sec=driver_seconds + deadline.exec_grace),
                maximum=driver_seconds + deadline.exec_grace,
                reserve=ADAPTER_FINALIZATION_SECONDS)
            (self.logs_dir / "supervisor.stdout").write_text(result.stdout or "")
            (self.logs_dir / "supervisor.stderr").write_text(result.stderr or "")
            context.metadata["driver_returncode"] = result.return_code
        except BaseException as exc:
            error = exc
            context.metadata["adapter_error_type"] = type(exc).__name__
            if nomination_stage is not None:
                context.metadata["semantic_decision_nomination_delivery"].update(
                    status="failed", stage=nomination_stage, error_type=type(exc).__name__[:128])
            if driver_started:
                context.metadata["driver_error_type"] = type(exc).__name__
        finally:
            try:
                await deadline.stage("report_export", lambda: environment.download_file(
                    report_path, self.logs_dir / "supervisor-result.json"), maximum=PUBLIC_EXPORT_SECONDS,
                    # Retain native usage before optional replay evidence. If
                    # time is tight, evidence may abstain while admission and
                    # public outputs keep their original cleanup allowance.
                    reserve=2 * PUBLIC_EXPORT_SECONDS + ADAPTER_RECEIPT_MARGIN_SECONDS)
                report = json.loads((self.logs_dir / "supervisor-result.json").read_text())
                provider_stop = provider_stop_observation(report)
                if provider_stop is not None:
                    context.metadata["provider_stop"] = provider_stop
                if capsule_budget != DEFAULT_CAPSULE_BUDGET:
                    expected_budget = capsule_budget_observation(capsule_budget)
                    context.metadata["capsule_budget_runtime_declared_matches"] = all(
                        type(report.get(key)) is type(value) and report.get(key) == value
                        for key, value in expected_budget.items())
                    # A treatment mismatch does not erase provider usage. The
                    # outer collector separately refuses a complete matched receipt.
                if working_set != DEFAULT_SEMANTIC_WORKING_SET:
                    expected_working_set = semantic_working_set_observation(working_set)
                    context.metadata["semantic_working_set_runtime_declared_matches"] = all(
                        type(report.get(key)) is type(value) and report.get(key) == value
                        for key, value in expected_working_set.items())
                if grok_turns != DEFAULT_GROK_CODING_MAX_TURNS:
                    context.metadata["grok_coding_runtime_declared_matches"] = (
                        type(report.get("grok_coding_max_turns")) is int
                        and report["grok_coding_max_turns"] == grok_turns)
                    context.metadata["grok_coding_runtime_policy_matches"] = grok_coding_runtime_matches(report, grok_turns)
                    context.metadata["grok_coding_not_applicable_provider_free"] = (
                        grok_coding_not_applicable_provider_free(report, grok_turns))
                usage = measured_usage(report)
                populate_harbor_usage(context, usage)
                context.metadata.update(task_completed=report["task_completed"], usage=usage,
                                        supervisor_seconds=report["seconds"], phases=report["phases"],
                                        intent_preplanning=(report.get("planning") or {}).get(
                                            "intent_preplanning", report.get("intent_preplanning")),
                                        requirement_coverage=(report.get("planning") or {}).get("requirement_coverage"),
                                        error=report.get("error"), remaining_processes=report.get("remaining_processes"))
            except (Exception, asyncio.CancelledError) as exc:
                context.metadata["report_unavailable"] = type(exc).__name__
                if isinstance(exc, asyncio.CancelledError) and error is None:
                    error = exc
            receipt = report.get("context_replay_evidence") if type(report) is dict else None
            if receipt is not None:
                try:
                    if type(receipt) is not dict:
                        raise ValueError("context replay evidence receipt must be an object")
                    if receipt.get("status") == "exported":
                        context.metadata["context_replay_evidence"] = await deadline.stage(
                            "context_replay_evidence_export", lambda: _download_context_replay_evidence(
                                environment, state=state, receipt=receipt, output=self.logs_dir),
                            maximum=PUBLIC_EXPORT_SECONDS,
                            reserve=2 * PUBLIC_EXPORT_SECONDS + ADAPTER_RECEIPT_MARGIN_SECONDS)
                    else:
                        context.metadata["context_replay_evidence"] = {
                            "status": "unavailable", "reason": "owner_export_unavailable", "verified": False}
                except (Exception, asyncio.CancelledError) as exc:
                    context.metadata["context_replay_evidence"] = {
                        "status": "unavailable", "reason": "delivery_unavailable", "verified": False,
                        "error_type": type(exc).__name__}
                    if isinstance(exc, asyncio.CancelledError) and error is None:
                        error = exc
            # This envelope contains public task inputs, the admitted goal
            # graph, and signatures, never the profile's private signing key.
            # Retain it even when later native materialization fails.
            try:
                await deadline.stage("admission_export", lambda: environment.download_file(
                    state + "/admission.json", self.logs_dir / "admission.json"),
                    maximum=PUBLIC_EXPORT_SECONDS,
                    reserve=PUBLIC_EXPORT_SECONDS + ADAPTER_RECEIPT_MARGIN_SECONDS)
                context.metadata["signed_admission_exported"] = True
            except (Exception, asyncio.CancelledError) as exc:
                context.metadata["signed_admission_exported"] = False
                context.metadata["admission_export_error"] = type(exc).__name__
                if isinstance(exc, asyncio.CancelledError) and error is None:
                    error = exc
            try:
                if task_profile is None:
                    context.metadata["public_output_evidence"] = await deadline.stage(
                        "public_output_export", lambda: export_public_outputs(environment, self.logs_dir),
                        maximum=PUBLIC_EXPORT_SECONDS)
            except (Exception, asyncio.CancelledError) as exc:
                context.metadata["public_output_export_error"] = type(exc).__name__
                if isinstance(exc, asyncio.CancelledError) and error is None:
                    error = exc
            deadline.finish()
        if error is not None:
            raise error
