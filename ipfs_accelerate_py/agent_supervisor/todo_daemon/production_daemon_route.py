"""Join operator-pinned providers to the native checkout mutation owner."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from ..merge.workspace_quarantine import mutation
from ..merge.checkout_lock import checkout_mutation_lease_state
from ..proof.formal_verification_contracts import canonical_json_bytes
from .contract_packet_provider_router import (
    ProductionContractPacket, ProviderBounds, ProviderRole,
)
from .production_context_slice import (
    _assert_safe_effect_path, _read_repository_regular_nofollow,
    _repository_root, _run_git_bounded, assert_proposal_covered_by_context,
    build_production_context_slice, derive_production_context_read_paths,
    load_verified_production_provider_launch_authority,
    verify_production_context_slice,
)
from .production_provider_cli import (
    BoundProductionCLIProvider, _canonical_json_bytes as policy_json_bytes,
)
from .production_reviewed_effect import (
    capture_production_reviewed_effect, derive_production_reviewed_patch_effects,
    production_task_contract,
)
from .contract_packet_provider_router import build_production_contract_packet


class _CurrentPacket(ProductionContractPacket):
    def assert_current(self, current_snapshot_id: str) -> None:
        super().assert_current(current_snapshot_id)
        self._fence()


def _atomic_replace(root: Path, name: str, content: bytes, mode: int) -> None:
    """Retain no-follow directory descriptors through one exact replacement."""

    parts = name.split("/")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, mode=0o755, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        temporary = ".provider-write-" + os.urandom(16).hex()
        file_descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            mode, dir_fd=descriptor,
        )
        try:
            try:
                view = memoryview(content)
                while view:
                    written = os.write(file_descriptor, view)
                    if written <= 0:
                        raise OSError("provider write made no progress")
                    view = view[written:]
                os.fchmod(file_descriptor, mode)
                os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)
            os.replace(temporary, parts[-1], src_dir_fd=descriptor,
                       dst_dir_fd=descriptor)
            os.fsync(descriptor)
        finally:
            try:
                os.unlink(temporary, dir_fd=descriptor)
            except FileNotFoundError:
                pass
    finally:
        os.close(descriptor)


def run_production_model_assisted_route(
    daemon: Any, task: Any, *, attempt: int, workspace_path: Path,
    baseline_ref: str = "HEAD", apply: bool = False,
) -> dict[str, Any]:
    """Route one current task, with real lease authority required for writes."""

    policy = daemon.production_provider_policy
    if type(apply) is not bool:
        raise ValueError("production apply must be an explicit boolean")
    if policy is None:
        raise ValueError("production route requires an operator provider policy")
    if apply and not daemon.implement:
        raise ValueError("production writes require implementation execution")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError("production attempt must be a nonnegative integer")
    if daemon.manual_completion_authority_revalidation_only:
        raise ValueError("production routing is forbidden during revalidation")
    root = _repository_root(workspace_path)
    common_directories = []
    for checkout in (daemon.repo_root, root):
        code, output, _error = _run_git_bounded(
            Path(checkout), "rev-parse", "--path-format=absolute", "--git-common-dir",
            maximum_stdout_bytes=4096,
        )
        if code:
            raise ValueError("production workspace has no native Git common directory")
        common_directories.append(Path(output.decode("utf-8").strip()).resolve(strict=True))
    if not os.path.samefile(*common_directories):
        raise ValueError("production workspace is outside the leased repository")
    code, registered, _error = _run_git_bounded(
        daemon.repo_root, "worktree", "list", "--porcelain", "-z",
        maximum_stdout_bytes=2 * 1024 * 1024,
    )
    if code or os.fsencode(root) not in {
        item.removeprefix(b"worktree ") for item in registered.split(b"\0")
        if item.startswith(b"worktree ")
    }:
        raise ValueError("production workspace is not a registered native checkout")
    identity = daemon._identity_for_task(task)
    contract = production_task_contract(task, identity)
    contract_bytes = canonical_json_bytes(contract)
    policy_bytes = policy_json_bytes(policy.to_dict())
    launch_pins = (daemon.production_provider_launch_authority_receipt_path,
                   daemon.production_provider_launch_authority_receipt_content_id,
                   tuple(daemon.worktree_submodule_paths))
    providers = (daemon._production_grok_provider, daemon._production_codex_provider)
    for provider, role in zip(providers, (ProviderRole.GROK_IMPLEMENT,
                                         ProviderRole.CODEX_REVIEW)):
        if (type(provider) is not BoundProductionCLIProvider
                or provider.role is not role
                or policy_json_bytes(provider.policy.to_dict()) != policy_bytes):
            raise ValueError("production providers must have operator-bound roles and policy")
    if providers[0] is providers[1]:
        raise ValueError("production implementation and review must be independent")
    effects = tuple(contract["outputs"])
    if any(daemon._overlaps_implementation_protected_path(path) for path in effects):
        raise ValueError("production effect includes an implementation protected path")
    reads = derive_production_context_read_paths(
        repo_root=root, baseline_ref=baseline_ref, effect_paths=effects,
    )
    context = build_production_context_slice(
        repo_root=root, task_id=task.task_id, task_payload=contract,
        read_paths=reads, effect_paths=effects, baseline_ref=baseline_ref,
        max_provider_prompt_tokens=policy.context_budget_tokens,
    )
    baseline = context.baseline_commit
    snapshot = "git-commit:" + baseline

    def fence() -> None:
        if (daemon.production_provider_policy is not policy
                or policy_json_bytes(policy.to_dict()) != policy_bytes
                or daemon._production_grok_provider is not providers[0]
                or daemon._production_codex_provider is not providers[1]
                or (daemon.production_provider_launch_authority_receipt_path,
                    daemon.production_provider_launch_authority_receipt_content_id,
                    tuple(daemon.worktree_submodule_paths)) != launch_pins
                or canonical_json_bytes(production_task_contract(
                    task, daemon._identity_for_task(task))) != contract_bytes):
            raise ValueError("production task, provider or operator policy changed")
        if daemon.production_provider_launch_authority_receipt_path is not None:
            load_verified_production_provider_launch_authority(
                receipt_path=daemon.production_provider_launch_authority_receipt_path,
                expected_receipt_content_id=daemon.production_provider_launch_authority_receipt_content_id,
                repo_root=daemon.repo_root,
                governed_repository_roots=daemon.worktree_submodule_paths,
            )
        verify_production_context_slice(
            context, repo_root=root, current_task_id=task.task_id,
            current_task_payload=contract, expected_read_paths=reads,
            expected_effect_paths=effects, baseline_ref=baseline,
        )

    base_packet = build_production_contract_packet(
        task_id=task.task_id, snapshot_id=snapshot, write_paths=effects,
        read_paths=reads, validation_commands=contract["validation"],
        acceptance_criteria=contract["acceptance"],
        extra_goal={"title": contract["title"], "priority": contract["priority"],
                    "track": contract["track"]},
    )
    packet = _CurrentPacket(
        packet_id=base_packet.packet_id, snapshot_id=snapshot, task_id=task.task_id,
        payload={**dict(base_packet.payload), **context.provider_payload()},
    )
    object.__setattr__(packet, "_fence", fence)

    def execute() -> dict[str, Any]:
        lease = daemon._current_checkout_mutation_lease() if apply else None
        if apply and lease is None:
            raise ValueError("production write has no native checkout lease")
        lease_id = lease.lease_id if lease is not None else ""
        before: dict[str, tuple[bytes, int] | None] = {}
        written: dict[str, bytes | None] = {}
        written_modes: dict[str, int | None] = {}
        approved_implementation: bytes | None = None
        admitted_implementation: bytes | None = None
        for name in effects:
            target, exists = _assert_safe_effect_path(root, name)
            before[name] = (
                (_read_repository_regular_nofollow(root, name),
                 stat.S_IMODE(os.lstat(target).st_mode)) if exists else None
            )

        def admission(proposal: Any) -> dict[str, Any]:
            nonlocal approved_implementation, admitted_implementation
            fence()
            if proposal.role is ProviderRole.GROK_IMPLEMENT:
                assert_proposal_covered_by_context(
                    context, proposal.payload, repo_root=root,
                    current_task_id=task.task_id, current_task_payload=contract,
                    expected_read_paths=reads, expected_effect_paths=effects,
                    baseline_ref=baseline,
                )
                admitted_implementation = policy_json_bytes(dict(proposal.payload))
            elif proposal.role is ProviderRole.CODEX_REVIEW:
                if (proposal.payload.get("decision") == "approve"
                        and proposal.payload.get("findings") == []
                        and proposal.payload.get("proposal") in (None, {})):
                    approved_implementation = admitted_implementation
            return {"accepted": True, "reason_code": "operator_bound_current_context"}

        def writer(proposal: Any, observed_lease_id: str) -> None:
            def require_lease() -> None:
                if (daemon._current_checkout_mutation_lease() is not lease
                        or observed_lease_id != lease_id
                        or checkout_mutation_lease_state(lease) != "current"):
                    raise ValueError("production writer lease changed at mutation boundary")

            require_lease()
            if proposal.role is not ProviderRole.GROK_IMPLEMENT:
                raise ValueError("production writer lease or selected role changed")
            if (approved_implementation is None or policy_json_bytes(
                    dict(proposal.payload)) != approved_implementation):
                raise ValueError("production writer requires exact independent Codex approval")
            fence()
            body = proposal.payload.get("proposal", proposal.payload)
            declared = body.get("declared_paths")
            if (not isinstance(declared, (list, tuple)) or not declared
                    or len(set(declared)) != len(declared)
                    or not set(declared).issubset(effects)):
                raise ValueError("production proposal has invalid declared paths")
            try:
                with mutation(daemon.repo_root, root):
                    fence()
                    require_lease()
                    files = body.get("files")
                    if files:
                        if {item["path"] for item in files} != set(declared):
                            raise ValueError("replacement paths differ from declared effect")
                        for item in files:
                            require_lease()
                            name = item["path"]
                            target, exists = _assert_safe_effect_path(root, name)
                            observed = ((_read_repository_regular_nofollow(root, name),
                                         stat.S_IMODE(os.lstat(target).st_mode)) if exists else None)
                            if observed != before[name]:
                                raise ValueError("production target changed before replacement")
                            content = item.get("content", item.get("new_content"))
                            if not isinstance(content, str):
                                raise ValueError("production replacement must be text")
                            written[name] = content.encode("utf-8")
                            written_modes[name] = before[name][1] if before[name] else 0o644
                            _atomic_replace(root, name, written[name],
                                            before[name][1] if before[name] else 0o644)
                    else:
                        patch = body.get("patch")
                        intended = derive_production_reviewed_patch_effects(
                            repo_root=root, baseline_ref=baseline, patch=patch,
                        )
                        if set(intended) != set(declared):
                            raise ValueError("patch paths differ from declared effect")
                        fence()
                        require_lease()
                        for name, postimage in intended.items():
                            target, exists = _assert_safe_effect_path(root, name)
                            observed = ((_read_repository_regular_nofollow(root, name),
                                         stat.S_IMODE(os.lstat(target).st_mode)) if exists else None)
                            if observed != before[name]:
                                raise ValueError("production target changed before patch")
                            written[name] = postimage[0] if postimage else None
                            written_modes[name] = postimage[1] if postimage else None
                        require_lease()
                        returncode, _stdout, _stderr = _run_git_bounded(
                            root, "apply", "--whitespace=nowarn", "-",
                            maximum_stdout_bytes=4096,
                            input_bytes=patch.encode("utf-8"),
                            child_umask=0o022,
                        )
                        if returncode:
                            raise ValueError("production patch could not be applied")
            except BaseException:
                restore()
                raise

        def restore() -> None:
            if written and (daemon._current_checkout_mutation_lease() is not lease
                            or checkout_mutation_lease_state(lease) != "current"):
                daemon._checkout_mutation_context.retain_until_protected_clean = True
                raise RuntimeError("production rollback requires native lease recovery")
            for name, expected in tuple(written.items()):
                saved = before[name]
                target, exists = _assert_safe_effect_path(root, name)
                observed = _read_repository_regular_nofollow(root, name) if exists else None
                observed_mode = stat.S_IMODE(os.lstat(target).st_mode) if exists else None
                if observed == (saved[0] if saved else None) and observed_mode == (saved[1] if saved else None):
                    written.pop(name)
                    written_modes.pop(name)
                    continue
                if observed != expected:
                    daemon._checkout_mutation_context.retain_until_protected_clean = True
                    raise RuntimeError("production rollback has an ambiguous external effect")
                if observed_mode != written_modes[name]:
                    daemon._checkout_mutation_context.retain_until_protected_clean = True
                    raise RuntimeError("production rollback has an ambiguous external mode change")
                if saved is None:
                    if exists:
                        target.unlink()
                else:
                    _atomic_replace(root, name, saved[0], saved[1])
                written.pop(name)
                written_modes.pop(name)

        try:
            result, event, receipt_path = daemon.route_model_assisted_contract_packet(
                packet, current_snapshot_id=snapshot, task=task, attempt=attempt,
                grok_provider=providers[0], codex_provider=providers[1],
                admission_gate=admission, writer=writer if apply else None,
                apply=apply, writer_lease_id=lease_id,
                bounds=ProviderBounds(max_prompt_tokens=policy.context_budget_tokens,
                                      timeout_seconds=policy.provider_timeout_seconds),
            )
            if result.write_performed and (
                    daemon._current_checkout_mutation_lease() is not lease
                    or checkout_mutation_lease_state(lease) != "current"):
                raise ValueError("production native lease changed before effect capture")
            captured = capture_production_reviewed_effect(
                repo_root=root, task=task, task_identity=identity, packet=packet,
                route_result=result, baseline_ref=baseline,
            ) if result.write_performed else None
        except BaseException:
            if apply:
                restore()
            raise
        return {"route_result": result, "reviewed_effect_binding": captured,
                "packet": packet, "context_slice": context,
                "receipt_path": receipt_path, "event": event,
                "completion_authoritative": False}

    fence()
    if not apply:
        return execute()
    result = daemon._run_checkout_mutation_transaction(
        task_id=task.task_id, attempt=attempt,
        operation="production_reviewed_provider_effect", callback=execute,
        failure_fields={"completion_authoritative": False},
        extra={"canonical_task_cid": identity.canonical_task_cid,
               "provider_policy_id": policy.policy_id, "baseline_ref": baseline},
    )
    if "route_result" not in result or result.get("checkout_mutation_release_failed"):
        raise RuntimeError(str(result.get("reason") or "native production transaction failed"))
    return result
