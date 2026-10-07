"""Harbor adapter for the isolated admitted native supervisor, one task per container.

Cold planning, index construction, and coding share the selected agent budget.
Installation is setup time, measured by Harbor separately for both harnesses.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shlex

from harbor.agents.base import BaseAgent
from harbor.models.agent.context import AgentContext

from .container_worker_deployment import deploy_worker_boundary
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
from .full_supervisor_benchmark import _intent_selection, validate_header_planning_selection
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
from .terminal_semantic_metadata_policy import (
    DEFAULT_SEMANTIC_METADATA_VIEW, validate_semantic_metadata_view,
    semantic_metadata_observation, require_semantic_metadata_archive,
)


def measured_usage(report: dict) -> dict:
    """Count observed native totals once, retaining provider-specific semantics."""
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
    grok = any(row.get("provider") == "grok_cli" for row in unique.values())
    cache_semantics = [(row.get("native_rollout_usage") or {}).get("cache_included_in_input")
                       if row.get("provider") == "grok_cli" else True
                       for row in unique.values()]
    result["cache_included_in_input"] = (cache_semantics[0] if cache_semantics
        and all(type(value) is bool and value == cache_semantics[0] for value in cache_semantics)
        else None) if grok else True
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
    result["billing_total_verified"] = False
    result["dollar_cost"] = None
    return result


def populate_harbor_usage(context: AgentContext, usage: dict) -> None:
    """Harbor input includes cache; retain native categories in the receipt."""
    value = usage["input_tokens"] if usage.get("cache_included_in_input") is True else None
    if usage.get("cache_included_in_input") is False:
        categories = [usage.get(key) for key in ("input_tokens", "cached_input_tokens",
                                                "cache_write_input_tokens", "output_tokens", "total_tokens")]
        if (all(type(item) is int and item >= 0 for item in categories)
                and sum(categories[:4]) == categories[4]):
            value = sum(categories[:3])
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
                 semantic_metadata_view: str = DEFAULT_SEMANTIC_METADATA_VIEW, **kwargs):
        super().__init__(*args, **kwargs)
        selected_provider = resolve_provider_profile(provider_profile)
        if self.model_name != selected_provider["model"] or arm not in {"full", "no-index"}:
            raise ValueError("the isolated comparison requires the pinned model and explicit arm")
        self.semantic_transport_schema = validate_semantic_transport_schema(semantic_transport_schema, arm=arm)
        self.coding_reply_mode = validate_coding_reply_mode(coding_reply_mode,
            arm=arm, provider=selected_provider["provider"])
        self.semantic_metadata_view = validate_semantic_metadata_view(semantic_metadata_view,
            arm=arm, transport_schema=self.semantic_transport_schema, provider=selected_provider["provider"])
        if type(disable_intent_autoencoder) is not bool:
            raise ValueError("Intent ablation switch must be boolean")
        self.runtime_archive = Path(runtime_archive).resolve(strict=True)
        self.provider_profile = provider_profile
        self.auth_json = Path(auth_json or Path.home() / (".codex/auth.json"
            if selected_provider["provider"] == "codex_cli" else ".grok/auth.json"))
        self.arm = arm
        self.model_revision = model_revision
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

    async def setup(self, environment):
        from .terminal_setup_cache_advice import (
            validate_setup_cache_selection, validate_setup_cache_prerequisites, apply_setup_cache_advice,
        )
        selection = getattr(self, "setup_cache_selection", None)
        manifest, _ = validate_setup_cache_selection(self.runtime_archive, selection)
        from .terminal_worker_capability import require_worker_capability
        require_worker_capability(manifest, getattr(self, "resource_profile", None))
        require_semantic_transport_archive(manifest,
            getattr(self, "semantic_transport_schema", DEFAULT_SEMANTIC_TRANSPORT_SCHEMA))
        selected_provider = require_runtime_provider_profile(manifest, getattr(self, "provider_profile", None))
        reply_mode = validate_coding_reply_mode(getattr(self, "coding_reply_mode", DEFAULT_CODING_REPLY_MODE),
            arm=self.arm, provider=selected_provider["provider"])
        require_coding_reply_archive(manifest, reply_mode)
        metadata_view = validate_semantic_metadata_view(
            getattr(self, "semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW), arm=self.arm,
            transport_schema=getattr(self, "semantic_transport_schema", DEFAULT_SEMANTIC_TRANSPORT_SCHEMA),
            provider=selected_provider["provider"])
        require_semantic_metadata_archive(manifest, metadata_view)
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
                                **({"isolated_uv_bootstrap": True} if task_profile is not None else {}),
                                **({"setup_cache_selection": selection} if selection is not None else {}))
        boundary = self.logs_dir / "worker-boundary"
        await deploy_worker_boundary(environment, output=boundary,
            **({"provider": "grok_cli"} if selected_provider["provider"] == "grok_cli" else {}))
        if selection is not None:
            self.setup_cache_receipt = await apply_setup_cache_advice(environment,
                archive_dir=self.runtime_archive, expected=selection,
                boundary_output=boundary, output=self.logs_dir / "setup-cache",
                resource_profile=getattr(self, "resource_profile", None))

    async def run(self, instruction: str, environment, context: AgentContext):
        from .terminal_setup_cache_advice import validate_setup_cache_selection
        manifest, _ = validate_setup_cache_selection(self.runtime_archive,
            getattr(self, "setup_cache_selection", None))
        retrieval_revision = require_retrieval_revision(manifest, self.arm,
            getattr(self, "model_revision", ""))
        transport = validate_semantic_transport_schema(
            getattr(self, "semantic_transport_schema", DEFAULT_SEMANTIC_TRANSPORT_SCHEMA), arm=self.arm)
        require_semantic_transport_archive(manifest, transport)
        reply_mode = validate_coding_reply_mode(getattr(self, "coding_reply_mode", DEFAULT_CODING_REPLY_MODE),
            arm=self.arm, provider=resolve_provider_profile(getattr(self, "provider_profile", None))["provider"])
        require_coding_reply_archive(manifest, reply_mode)
        metadata_view = validate_semantic_metadata_view(
            getattr(self, "semantic_metadata_view", DEFAULT_SEMANTIC_METADATA_VIEW), arm=self.arm,
            transport_schema=transport,
            provider=resolve_provider_profile(getattr(self, "provider_profile", None))["provider"])
        require_semantic_metadata_archive(manifest, metadata_view)
        validate_header_planning_selection(validate_source384_binding(manifest),
            getattr(self, "intent_requirement_contract", None), self.arm)
        from .terminal_worker_capability import require_worker_capability
        require_worker_capability(manifest, getattr(self, "resource_profile", None))
        public = self.logs_dir / "instruction.md"
        public.write_text(instruction)
        instruction_path = ROOT + "/instruction.md"
        await environment.upload_file(public, instruction_path)
        # Owner state contains signing keys. Export only the explicit public
        # result and bounded process diagnostics; never download the state tree.
        state = ROOT + "/state/benchmark"
        report_path = state + "-result.json"
        profile = getattr(self, "resource_profile", None)
        budget = execution_budget(profile)
        argv = [PYTHON, "-P", "-m", "benchmarks.agent_supervisor.container_coding.terminal_container_supervisor",
                "--instruction", instruction_path, "--state", state,
                "--arm", self.arm, "--timeout-seconds", str(budget["driver_seconds"])]
        provider_profile = getattr(self, "provider_profile", None)
        if provider_profile is not None:
            argv += ["--provider-profile", provider_profile]
        if transport != DEFAULT_SEMANTIC_TRANSPORT_SCHEMA:
            argv += ["--semantic-transport-schema", transport]
        if reply_mode != DEFAULT_CODING_REPLY_MODE:
            argv += ["--coding-reply-mode", reply_mode]
        if metadata_view != DEFAULT_SEMANTIC_METADATA_VIEW:
            argv += ["--semantic-metadata-view", metadata_view]
        task_profile = getattr(self, "task_profile", None)
        if task_profile is not None:
            from .terminal_task_profile import validate_task_profile
            task_profile = validate_task_profile(task_profile, instruction=instruction)
            artifact = self.logs_dir / "task-profile.json"
            artifact.write_text(json.dumps(task_profile, sort_keys=True, allow_nan=False) + "\n")
            await environment.upload_file(artifact, ROOT + "/task-profile.json")
            argv += ["--task-profile", ROOT + "/task-profile.json"]
        if profile is not None:
            argv += ["--resource-profile", profile]
        requirement_contract = getattr(self, "intent_requirement_contract", None)
        if requirement_contract is not None:
            artifact = self.logs_dir / "intent-requirements.json"
            artifact.write_text(json.dumps(requirement_contract, sort_keys=True,
                                           separators=(",", ":"), allow_nan=False) + "\n")
            requirement_path = ROOT + "/intent-requirements.json"
            await environment.upload_file(artifact, requirement_path)
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
        context.metadata = {"arm": self.arm, "task_completed": False, "official_reward": None,
                            "provider_profile": resolve_provider_profile(provider_profile),
                            **semantic_transport_selection(transport),
                            **coding_reply_selection(reply_mode),
                            **semantic_metadata_observation(metadata_view),
                            **_intent_selection(requirement_contract),
                            "runtime_archive_sha256": manifest["archive_sha256"],
                            "security_training_assets": asset_observation,
                            "security_model_assets": asset_observation,
                            "intent_model_assets": intent_observation,
                            "source384_assets": source384_observation,
                            "resource_profile": profile, "execution_budget": budget,
                            "admission_environment": admission_environment(profile),
                            **({"setup_cache": self.setup_cache_receipt} if hasattr(self, "setup_cache_receipt") else {}),
                            "planning_and_cold_index_charged_to_agent_time": True}
        if task_profile is not None:
            context.metadata["task_profile"] = task_profile
            context.metadata["public_output_capture"] = "not_selected_for_generic_profile"
        try:
            if task_profile is None:
                context.metadata["public_input_capture"] = await asyncio.wait_for(
                    capture_public_inputs(environment, self.logs_dir), timeout=5)
        except Exception as exc:
            context.metadata["public_input_capture_error"] = type(exc).__name__
        error = None
        try:
            result = await environment.exec(command=shlex.join(argv), cwd="/app", user="supervisor",
                                            env={**runtime_environment(), **admission_environment(profile)},
                                            timeout_sec=budget["exec_seconds"])
            (self.logs_dir / "supervisor.stdout").write_text(result.stdout or "")
            (self.logs_dir / "supervisor.stderr").write_text(result.stderr or "")
            context.metadata["driver_returncode"] = result.return_code
        except BaseException as exc:
            error = exc
            context.metadata["driver_error_type"] = type(exc).__name__
        finally:
            try:
                await asyncio.wait_for(environment.download_file(report_path, self.logs_dir / "supervisor-result.json"), timeout=5)
                report = json.loads((self.logs_dir / "supervisor-result.json").read_text())
                usage = measured_usage(report)
                populate_harbor_usage(context, usage)
                context.metadata.update(task_completed=report["task_completed"], usage=usage,
                                        supervisor_seconds=report["seconds"], phases=report["phases"],
                                        intent_preplanning=(report.get("planning") or {}).get(
                                            "intent_preplanning", report.get("intent_preplanning")),
                                        requirement_coverage=(report.get("planning") or {}).get("requirement_coverage"),
                                        error=report.get("error"), remaining_processes=report.get("remaining_processes"))
            except Exception as exc:
                context.metadata["report_unavailable"] = type(exc).__name__
            # This envelope contains public task inputs, the admitted goal
            # graph, and signatures, never the profile's private signing key.
            # Retain it even when later native materialization fails.
            try:
                await asyncio.wait_for(environment.download_file(state + "/admission.json",
                    self.logs_dir / "admission.json"), timeout=5)
                context.metadata["signed_admission_exported"] = True
            except Exception as exc:
                context.metadata["signed_admission_exported"] = False
                context.metadata["admission_export_error"] = type(exc).__name__
            try:
                if task_profile is None:
                    context.metadata["public_output_evidence"] = await asyncio.wait_for(
                        export_public_outputs(environment, self.logs_dir), timeout=5)
            except Exception as exc:
                context.metadata["public_output_export_error"] = type(exc).__name__
        if error is not None:
            raise error
