"""Explicit local benchmark planning, separate from universal IR admission.

An independently authored, owner-signed manifest permits a bounded plan to
start while its exact acceptance checks remain pending. This module never
issues a CodeProof, generic hard-domain admission, or production activation.
The runner is trusted supervisor code; subprocess isolation is the caller's
container boundary, not a claim made by this contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from ..control.profile_authority import (
    load_local_profile,
    sign_profile_binding,
    verify_did_key_signature,
)
from ..core.multiformats_identity import cid_for_dag_json
from ..planning.formal_plan_compiler import FormalPlanCompiler
from ..planning.formal_plan_validator import validate_formal_plan
from ..proof.formal_verification_contracts import content_identity
from ..prompt.prompt_workflow import PromptEvidenceRecord, PromptGoalGraph

MANIFEST_SCHEMA = "supervisor-local-benchmark-manifest@1"
PLANNER_MANIFEST_SCHEMA = "supervisor-local-benchmark-manifest@2"
CREATE_MANIFEST_SCHEMA = "supervisor-local-benchmark-manifest@3"
INTENT_MANIFEST_SCHEMA = "supervisor-local-benchmark-manifest@4"
SUPPORTED_MANIFEST_SCHEMAS = frozenset({
    MANIFEST_SCHEMA, PLANNER_MANIFEST_SCHEMA, CREATE_MANIFEST_SCHEMA, INTENT_MANIFEST_SCHEMA,
})
CONTRACT_SCHEMA = "supervisor-local-pending-completion@1"
INTENT_CONTRACT_SCHEMA = "supervisor-local-pending-completion@2"
PLANNING_RECEIPT_SCHEMA = "supervisor-local-planning-receipt@1"
INTENT_PLANNING_RECEIPT_SCHEMA = "supervisor-local-planning-receipt@2"
INTENT_REQUIREMENT_ARTIFACT_SCHEMA = "supervisor-local-intent-requirement-artifact@1"
RESULT_SCHEMA = "supervisor-local-observed-validation@1"
CONTRACT_KEY = "local_planning_contract"
PLANNING_RECEIPT_REFERENCE_SCHEMA = "supervisor-local-planning-receipt-reference@1"
INTENT_PLANNING_RECEIPT_REFERENCE_SCHEMA = "supervisor-local-planning-receipt-reference@2"
MAX_PLANNING_RECEIPT_BYTES = 4 * 1024 * 1024
LOCAL_POLICY = {
    "schema": "supervisor-isolated-benchmark-planning-policy@1",
    "scope": "signed-local-benchmark-only",
    "pending_assurance": "candidate",
    "explicit_proof_obligations": [],
    "external_ir_roots": [],
    "production_activation": False,
}


class LocalPlanningError(ValueError):
    """Local manifest, planning or completion contract was not satisfied."""


def supports_created_outputs(manifest: Mapping) -> bool:
    """Share the declared-create capability across manifest readers."""
    return manifest.get("schema") in {CREATE_MANIFEST_SCHEMA, INTENT_MANIFEST_SCHEMA}


def _plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _path(name: str, *, dot=False) -> str:
    if dot and name == ".":
        return name
    if not isinstance(name, str):
        raise LocalPlanningError("path must be text")
    rel = PurePosixPath(name)
    if (
        not name
        or any(c in name for c in "\n\r\0")
        or rel.is_absolute()
        or ".." in rel.parts
        or str(rel) != name
        or ".git" in rel.parts
        or any(part.startswith(".runtime") for part in rel.parts)
    ):
        raise LocalPlanningError("path must be canonical and source-relative")
    return name


def _git(root: Path, *args: str) -> str:
    from .candidate_execution import GIT_OWNER_ENV
    return subprocess.check_output(
        ["/usr/bin/git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null",
         "-c", "core.fsmonitor=false", "-C", str(root), *args],
        env={"PATH": "/usr/bin:/bin", **GIT_OWNER_ENV}, text=True,
    ).strip()


def _repository(root: Path) -> tuple[Path, str, str]:
    root = Path(root).absolute()
    if root.resolve(strict=True) != root or _git(root, "rev-parse", "--show-toplevel") != str(root):
        raise LocalPlanningError("repository must be its exact non-symlink Git root")
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    identity = cid_for_dag_json(
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.observed-repository-root@1",
            "root": str(root),
            "head_tree": tree,
        }
    )
    return root, head, identity


def _sources(root: Path, names: Sequence[str], *, max_files=128) -> dict:
    result = {}
    total = 0
    for name in sorted(names):
        _path(name)
        path = root / name
        if path.resolve() != path or not path.is_file() or path.is_symlink():
            raise LocalPlanningError("source missing, symlinked, or outside repository")
        raw = path.read_bytes()
        total += len(raw)
        if len(raw) > 1_000_000 or total > 4_000_000:
            raise LocalPlanningError("local source bound exceeded")
        result[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "executable": bool(path.stat().st_mode & 0o111),
        }
    if not result or len(result) > max_files:
        raise LocalPlanningError("local source inventory bound exceeded")
    return result


def _untracked_sources(root: Path) -> None:
    # Include ignored files too: an ignored Python module or stale bytecode
    # can still affect a public check. Only supervisor-owned runtime artifacts
    # are outside the source tree used by this deliberately small policy.
    names = _git(root, "ls-files", "--others", "-z").split("\0")
    if any(name and not name.startswith(".runtime/") for name in names):
        raise LocalPlanningError("untracked source or bytecode outside runtime directory")


def _tree(sources: Mapping) -> str:
    return content_identity({"schema": "supervisor-local-source-tree@1", "sources": sources})


def observe_local_manifest_sources(root: Path, manifest: Mapping, *, initial=False) -> dict:
    """Observe the exact @3 baseline plus explicitly absent-to-created outputs.

    Existing manifest versions retain their original inventory implementation.
    This helper is also used by the signed Portal transition verifier so a
    newly created output cannot disappear from the resulting source binding.
    """
    if not supports_created_outputs(manifest):
        raise LocalPlanningError("declared-create observation requires manifest version 3 or 4")
    created = manifest.get("created_outputs")
    declared = [item["path"] for spec in manifest["tasks"] for item in spec["outputs"]
                if item.get("effect") == "create"]
    minimum = 0 if manifest.get("schema") == INTENT_MANIFEST_SCHEMA else 1
    if (not isinstance(created, list) or not minimum <= len(created) <= 32
            or created != sorted(set(declared)) or len(declared) != len(set(declared))):
        raise LocalPlanningError("exact unique declared-create output population required")
    baseline = set(manifest["sources"])
    baseline_git = set(filter(None, _git(root, "ls-tree", "-r", "--name-only", "-z", manifest["baseline_commit"]).split("\0")))
    if baseline_git != baseline:
        raise LocalPlanningError("baseline source inventory differs from its signed commit")
    for name in created:
        _path(name)
        path = root / name
        if name in baseline or path.resolve() != path or path.is_symlink():
            raise LocalPlanningError("created output must be absent at baseline and never symlinked")
        if initial and path.exists():
            raise LocalPlanningError("created output must be absent before admission")
    tracked = set(filter(None, _git(root, "ls-files", "-z").split("\0")))
    if not baseline <= tracked <= baseline | set(created) or (initial and tracked != baseline):
        raise LocalPlanningError("tracked source inventory escaped declared creations")
    untracked = set(filter(None, _git(root, "ls-files", "--others", "-z").split("\0")))
    if any(name not in created and not name.startswith(".runtime/") for name in untracked):
        raise LocalPlanningError("undeclared new source outside runtime directory")
    current_names = baseline | {name for name in created if (root / name).exists()}
    current = _sources(root, sorted(current_names), max_files=256)
    outputs = {item["path"] for spec in manifest["tasks"] for item in spec["outputs"]}
    if initial and current != manifest["sources"]:
        raise LocalPlanningError("source changed since independent manifest")
    if any(current[name] != original for name, original in manifest["sources"].items() if name not in outputs):
        raise LocalPlanningError("source changed outside permitted outputs")
    return current


def _signed(payload: dict, manifest: dict) -> dict:
    return {
        "payload": payload,
        "binding": sign_profile_binding(
            profile_dir=Path(manifest["profile_dir"]),
            lifecycle_dir=Path(manifest["lifecycle_dir"]),
            payload=payload,
        ),
    }


def _verify_signature(envelope: Mapping, profile) -> dict:
    if set(envelope) != {"payload", "binding"}:
        raise LocalPlanningError("exact signed envelope required")
    payload, binding = envelope["payload"], envelope["binding"]
    if (
        set(binding) != {"identity", "signature", "profile_id"}
        or binding["identity"] != profile.identity_did
        or binding["profile_id"] != profile.profile_id
    ):
        raise LocalPlanningError("binding belongs to a different owner")
    verify_did_key_signature(
        identity_did=profile.identity_did, payload=payload, signature=binding["signature"]
    )
    return _plain(payload)


def author_local_benchmark_manifest(
    *,
    repository: Path,
    profile_dir: Path,
    lifecycle_dir: Path,
    task_specs: Sequence[Mapping[str, Any]],
    planning_roots: Mapping[str, str],
    planning_inputs: Mapping | None = None,
    intent_requirements: Mapping | None = None,
) -> dict:
    """Sign independently declared benchmark inputs, before model planning.

    task_specs are administrator/benchmark data, never extracted from a model
    proposal. Unknown keys, proof claims and external policy roots fail closed.
    Versions 1/2 modify tracked files; version 3 also binds exact absent outputs.
    """
    root, head, repository_cid = _repository(repository)
    profile = load_local_profile(
        repository_cid=repository_cid, profile_dir=profile_dir, lifecycle_dir=lifecycle_dir
    )
    if head != profile.baseline_commit:
        raise LocalPlanningError("profile baseline does not match repository")
    names = _git(root, "ls-files", "-z").split("\0")
    names = [name for name in names if name]
    created = sorted({item["path"] for spec in task_specs for item in spec["outputs"]
                      if item.get("effect") == "create"})
    payload = {
        "schema": MANIFEST_SCHEMA,
        "repository": str(root),
        "repository_cid": repository_cid,
        "baseline_commit": head,
        "profile_content_id": profile.content_id,
        "profile_dir": str(Path(profile_dir).resolve(strict=True)),
        "lifecycle_dir": str(Path(lifecycle_dir).resolve(strict=True)),
        "sources": _sources(root, names, max_files=256 if created else 128),
        "policy": LOCAL_POLICY,
        "planning_roots": dict(planning_roots),
        "tasks": list(task_specs),
    }
    payload = _plain(payload)
    if planning_inputs is not None:
        payload["schema"] = PLANNER_MANIFEST_SCHEMA
        payload["planning_inputs"] = _plain(planning_inputs)
    if created:
        payload["schema"] = CREATE_MANIFEST_SCHEMA
        payload["created_outputs"] = created
    if intent_requirements is not None:
        from ..prompt.intent_plan_coverage import validate_intent_requirement_contract

        requirement_contract = validate_intent_requirement_contract(intent_requirements)
        payload["schema"] = INTENT_MANIFEST_SCHEMA
        payload["created_outputs"] = created
        # Candidate inference contains finite confidence values. Preserve those
        # bytes as inert JSON; universal proof identity contracts reject floats.
        payload["intent_requirements"] = {
            "schema": INTENT_REQUIREMENT_ARTIFACT_SCHEMA,
            "contract_json": json.dumps(requirement_contract, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False, allow_nan=False),
            "contract_cid": cid_for_dag_json(requirement_contract),
        }
    envelope = _signed(payload, payload)
    _manifest(envelope, initial=True)
    return envelope


def _validate_local_manifest_declarations(payload: Mapping) -> set[str]:
    """Check public signed declarations without observing files or owner state.

    Both owner admission and public worker replay use these declaration rules.
    This validation grants no dispatch permission; the owner still verifies its
    active profile, repository, source observations and native persisted tasks.
    """
    if not isinstance(payload, Mapping):
        raise LocalPlanningError("unsupported manifest contract")
    expected = {
        "schema",
        "repository",
        "repository_cid",
        "baseline_commit",
        "profile_content_id",
        "profile_dir",
        "lifecycle_dir",
        "sources",
        "policy",
        "planning_roots",
        "tasks",
    }
    if payload.get("schema") == PLANNER_MANIFEST_SCHEMA:
        expected.add("planning_inputs")
    if payload.get("schema") == CREATE_MANIFEST_SCHEMA:
        expected.add("created_outputs")
        if "planning_inputs" in payload:
            expected.add("planning_inputs")
    if payload.get("schema") == INTENT_MANIFEST_SCHEMA:
        expected.update({"created_outputs", "intent_requirements"})
        if "planning_inputs" in payload:
            expected.add("planning_inputs")
    if set(payload) != expected or payload.get("schema") not in SUPPORTED_MANIFEST_SCHEMAS:
        raise LocalPlanningError("unsupported manifest contract")
    sources = payload["sources"]
    maximum = 256 if supports_created_outputs(payload) else 128
    if not isinstance(sources, Mapping) or not 1 <= len(sources) <= maximum:
        raise LocalPlanningError("local source inventory bound exceeded")
    for name, source in sources.items():
        _path(name)
        if name == "." or not isinstance(source, Mapping) or set(source) != {"sha256", "executable"}:
            raise LocalPlanningError("exact signed source binding required")
        digest = source["sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or type(source["executable"]) is not bool):
            raise LocalPlanningError("exact signed source binding required")
    if payload["policy"] != LOCAL_POLICY:
        raise LocalPlanningError("external or explicit proof obligations are unsupported")
    roots = payload["planning_roots"]
    if not isinstance(roots, Mapping) or set(roots) != {"request_cid", "scan_cid", "program_root"} or not all(
        isinstance(value, str) and value for value in roots.values()
    ):
        raise LocalPlanningError("exact independent planning roots required")
    specs = payload["tasks"]
    if not isinstance(specs, list) or not 1 <= len(specs) <= 16:
        raise LocalPlanningError("bounded task population required")
    created = set()
    if supports_created_outputs(payload):
        names = payload["created_outputs"]
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise LocalPlanningError("exact unique declared-create output population required")
        created = set(names)
    seen, outputs = set(), set()
    for spec in specs:
        if not isinstance(spec, Mapping) or set(spec) != {
            "task_key",
            "scope_paths",
            "outputs",
            "validations",
            "acceptance",
            "dependencies",
        }:
            raise LocalPlanningError(
                "unsupported task declaration (phase/proofs cannot be supplied)"
            )
        if not isinstance(spec["task_key"], str) or not spec["task_key"] or spec["task_key"] in seen:
            raise LocalPlanningError("task keys must be unique")
        seen.add(spec["task_key"])
        if any(not isinstance(spec[name], list) for name in
               ("scope_paths", "outputs", "validations", "acceptance", "dependencies")):
            raise LocalPlanningError("unsupported task declaration (phase/proofs cannot be supplied)")
        if not spec["outputs"] or not spec["validations"] or not spec["acceptance"]:
            raise LocalPlanningError("output, validation and acceptance declarations required")
        for name in spec["scope_paths"]:
            if _path(name) not in set(payload["sources"]) | created:
                raise LocalPlanningError("scope outside source inventory")
        for output in spec["outputs"]:
            if not isinstance(output, Mapping):
                raise LocalPlanningError("version 1 only permits exact existing-file modifications")
            name = _path(output.get("path"))
            allowed_effect = "create" if name in created else "modify"
            if set(output) != {"path", "effect", "media_type"} or output["effect"] != allowed_effect:
                raise LocalPlanningError("version 1 only permits exact existing-file modifications")
            if _path(output["path"]) not in spec["scope_paths"]:
                raise LocalPlanningError("output outside declared scope")
            outputs.add(output["path"])
        keys = set()
        for check in spec["validations"]:
            if not isinstance(check, Mapping) or set(check) != {"validation_key", "argv", "cwd", "expected_exit_codes", "policy_cid"}:
                raise LocalPlanningError("exact validation contract required")
            if (
                not isinstance(check["validation_key"], str)
                or check["validation_key"] in keys
                or not check["validation_key"]
                or check["expected_exit_codes"] != [0]
                or check["policy_cid"] != content_identity(LOCAL_POLICY)
            ):
                raise LocalPlanningError("unsupported or duplicate validation")
            keys.add(check["validation_key"])
            _path(check["cwd"], dot=True)
            if (
                not isinstance(check["argv"], list)
                or not 1 <= len(check["argv"]) <= 64
                or any(
                    not isinstance(part, str)
                    or not part
                    or any(c in part for c in "\n\r\0")
                    or len(part) > 4096
                    for part in check["argv"]
                )
            ):
                raise LocalPlanningError("explicit bounded argv required")
        for criterion in spec["acceptance"]:
            descriptive_ids = {
                PromptEvidenceRecord.from_dict(item).evidence_cid
                for item in payload.get("planning_inputs", {}).get("selected_evidence", [])
            }
            if (
                not isinstance(criterion, Mapping)
                or set(criterion) != {"criterion_key", "criterion", "evidence_cids", "validation_keys"}
                or not set(criterion["evidence_cids"]) <= descriptive_ids
                or not criterion["validation_keys"]
                or not set(criterion["validation_keys"]) <= keys
            ):
                raise LocalPlanningError("acceptance must retain declared validation references")
    if any(any(not isinstance(name, str) for name in spec["dependencies"])
           or not set(spec["dependencies"]) <= seen for spec in specs):
        raise LocalPlanningError("dependency outside manifest")
    if supports_created_outputs(payload):
        declared = [output["path"] for spec in specs for output in spec["outputs"]
                    if output["effect"] == "create"]
        minimum = 0 if payload["schema"] == INTENT_MANIFEST_SCHEMA else 1
        if (not minimum <= len(payload["created_outputs"]) <= 32
                or payload["created_outputs"] != sorted(set(declared))
                or len(declared) != len(set(declared))):
            raise LocalPlanningError("exact unique declared-create output population required")
        for name in created:
            _path(name)
            if name == "." or name in sources:
                raise LocalPlanningError("created output must be absent at baseline and never symlinked")
    return outputs


def _manifest(envelope: Mapping, *, initial=False, source_transition=None) -> tuple[dict, Any, dict]:
    payload = envelope.get("payload", {})
    outputs = _validate_local_manifest_declarations(payload)
    profile = load_local_profile(
        repository_cid=payload["repository_cid"],
        profile_dir=Path(payload["profile_dir"]),
        lifecycle_dir=Path(payload["lifecycle_dir"]),
    )
    payload = _verify_signature(envelope, profile)
    if profile.content_id != payload["profile_content_id"] or any(
        not profile.allows(cap)
        for cap in ("read", "edit", "test", "isolated_worktree", "write_worktree")
    ):
        raise LocalPlanningError("active profile does not grant bounded local work")
    root, head, repository_cid = _repository(Path(payload["repository"]))
    if not supports_created_outputs(payload):
        _untracked_sources(root)
    if any(Path(payload[key]).is_relative_to(root) for key in ("profile_dir", "lifecycle_dir")):
        raise LocalPlanningError("owner keys and lifecycle must be outside worker repository")
    if source_transition is not None:
        if initial:
            raise LocalPlanningError("source transition cannot authorize initial planning")
        from .local_completion_bridge import verify_local_source_transition

        verify_local_source_transition(
            source_transition, manifest_envelope=envelope, profile=profile,
            current_head=head, current_repository_cid=repository_cid,
        )
    elif head != payload["baseline_commit"] or repository_cid != profile.repository_cid:
        raise LocalPlanningError("repository identity or baseline drift")
    if supports_created_outputs(payload):
        current = observe_local_manifest_sources(root, payload, initial=initial)
    else:
        inventory = set(_git(root, "ls-files", "-z").split("\0")) - {""}
        if inventory != set(payload["sources"]):
            raise LocalPlanningError("tracked source inventory changed")
        current = _sources(root, sorted(inventory))
        if initial and current != payload["sources"]:
            raise LocalPlanningError("source changed since independent manifest")
        if any(
            current[name] != original
            for name, original in payload["sources"].items()
            if name not in outputs
        ):
            raise LocalPlanningError("source changed outside permitted outputs")
    if "planning_inputs" in payload:
        _verify_planning_inputs(payload)
    if payload["schema"] == INTENT_MANIFEST_SCHEMA:
        _verify_intent_requirements(payload)
    return payload, profile, current


def decode_intent_requirement_contract(manifest: Mapping) -> dict:
    """Read an inert candidate artifact without changing proof identity rules."""
    from ..prompt.intent_plan_coverage import MAX_INTENT_PLAN_BYTES

    artifact = manifest.get("intent_requirements")
    if (not isinstance(artifact, Mapping)
            or set(artifact) != {"schema", "contract_json", "contract_cid"}
            or artifact["schema"] != INTENT_REQUIREMENT_ARTIFACT_SCHEMA
            or not isinstance(artifact["contract_json"], str)
            or len(artifact["contract_json"].encode("utf-8")) > MAX_INTENT_PLAN_BYTES):
        raise LocalPlanningError("exact bounded intent requirement artifact required")
    try:
        contract = json.loads(artifact["contract_json"])
        canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False)
        if canonical != artifact["contract_json"] or cid_for_dag_json(contract) != artifact["contract_cid"]:
            raise LocalPlanningError("intent requirement artifact identity differs")
    except (ValueError, TypeError, RecursionError) as exc:
        raise LocalPlanningError("invalid inert intent requirement JSON") from exc
    return contract


def _verify_intent_requirements(manifest: Mapping) -> dict:
    """Rebuild requirements against an immutable independently signed input."""
    from ..prompt.intent_plan_coverage import validate_intent_requirement_contract

    contract = decode_intent_requirement_contract(manifest)
    if not isinstance(contract, Mapping):
        raise LocalPlanningError("intent requirement contract must be an object")
    name = _path(contract.get("source_path"))
    outputs = {row["path"] for spec in manifest["tasks"] for row in spec["outputs"]}
    if name not in manifest["sources"] or name in outputs:
        raise LocalPlanningError("intent requirement source must be an immutable signed input")
    path = Path(manifest["repository"]) / name
    if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
        raise LocalPlanningError("intent requirement source must be a regular canonical input")
    raw = path.read_bytes()
    if len(raw) > 1_000_000 or hashlib.sha256(raw).hexdigest() != manifest["sources"][name]["sha256"]:
        raise LocalPlanningError("intent requirement source differs from signed bytes")
    return _verify_intent_requirement_text(manifest, raw.decode("utf-8"))


def _verify_intent_requirement_text(manifest: Mapping, source_text: str) -> dict:
    """Check explicitly observed source bytes without private profile access.

    Used by public worker replay after current/baseline source verification.
    Runtime admission continues to observe the canonical source independently.
    """
    from ..prompt.intent_plan_coverage import validate_intent_requirement_contract

    contract = decode_intent_requirement_contract(manifest)
    name = _path(contract.get("source_path"))
    outputs = {row["path"] for spec in manifest["tasks"] for row in spec["outputs"]}
    raw = source_text.encode("utf-8")
    if (name not in manifest["sources"] or name in outputs or len(raw) > 1_000_000
            or hashlib.sha256(raw).hexdigest() != manifest["sources"][name]["sha256"]):
        raise LocalPlanningError("intent requirement source differs from signed immutable input")
    try:
        return validate_intent_requirement_contract(contract, source_text=source_text)
    except (ValueError, TypeError, KeyError) as exc:
        raise LocalPlanningError("invalid source-bound intent requirement contract: " + str(exc)) from exc


def _domain_records(manifest: Mapping) -> dict:
    """Descriptive local scope records, expressly not external IR assurance."""
    constraints = _plain(manifest["tasks"])
    for task in constraints:
        for criterion in task["acceptance"]:
            # Scan evidence is observed after request construction. Its exact
            # references are signed in the enclosing manifest, while these
            # earlier domain records describe the independent task/checks.
            criterion.pop("evidence_cids")
    return {domain: {
        "schema": "supervisor-local-descriptive-domain@1", "domain": domain,
        "repository_cid": manifest["repository_cid"],
        "profile_content_id": manifest["profile_content_id"],
        "source_tree_id": _tree(manifest["sources"]),
        "task_constraints_cid": content_identity(constraints),
        "policy_cid": content_identity(LOCAL_POLICY),
        "external_ir_assurance": "unavailable", "proof_obligations": [],
        "completion_authority": False, "declared_authority": "descriptive_input",
    } for domain in ("intent", "legal", "security")}


def local_planning_domain_declarations(*, repository: Path, profile_dir: Path,
    lifecycle_dir: Path, task_specs: Sequence[Mapping]) -> dict:
    root, _, repository_cid = _repository(repository)
    profile = load_local_profile(repository_cid=repository_cid,
        profile_dir=profile_dir, lifecycle_dir=lifecycle_dir)
    created = any(item.get("effect") == "create" for spec in task_specs for item in spec["outputs"])
    return _domain_records({"repository_cid": repository_cid,
        "profile_content_id": profile.content_id, "tasks": _plain(task_specs),
        "sources": _sources(root, [name for name in _git(root, "ls-files", "-z").split("\0") if name], max_files=256 if created else 128)})


def _verify_planning_inputs(manifest: Mapping) -> tuple[tuple[str, ...], tuple]:
    from ..prompt.prompt_workflow import PromptWorkflowRequest, DirectoryScanReceipt
    from ..prompt.prompt_directory_scanner import repository_root_cid
    from ..prompt.prompt_goal_planner import _select_evidence, _validate_request_scan_pair, PromptGoalPlannerConfig, PromptGoalProviderRequestError

    inputs = manifest["planning_inputs"]
    if set(inputs) != {"request", "scan", "domain_declarations", "selected_evidence"}:
        raise LocalPlanningError("exact local native planning inputs required")
    domains = _domain_records(manifest)
    if inputs["domain_declarations"] != domains:
        raise LocalPlanningError("local domain declarations changed or claim external assurance")
    request = PromptWorkflowRequest.from_dict(inputs["request"])
    scan = DirectoryScanReceipt.from_dict(inputs["scan"])
    try:
        _validate_request_scan_pair(request, scan)
    except PromptGoalProviderRequestError as exc:
        raise LocalPlanningError("native request/scan binding differs") from exc
    if (request.repository_root != manifest["repository"]
            or request.repository_root_cid != repository_root_cid(manifest["repository"])
            or {"request_cid": request.request_cid, "scan_cid": scan.scan_cid,
                "program_root": request.program_root} != manifest["planning_roots"]
            or request.policy_root != content_identity(LOCAL_POLICY)):
        raise LocalPlanningError("native planner source/root bindings differ from local manifest")
    for domain, declaration in domains.items():
        if getattr(request, domain + "_ir_root") != content_identity(declaration):
            raise LocalPlanningError("native planner has a foreign domain root")
    selected = _select_evidence(request, scan, PromptGoalPlannerConfig())
    if inputs["selected_evidence"] != [item.to_dict() for item in selected]:
        raise LocalPlanningError("native selected evidence changed")
    allowed_sources = {"prompt", "directory_scan", "ast", "program_behavior", "semantic_index"}
    if any((item.source_kind not in allowed_sources and not item.source_kind.startswith("directory_scan_"))
           or item.authority.value not in {"prompt", "scan_advisory"}
           for item in selected):
        raise LocalPlanningError("native evidence authority is unsupported by local planning")
    return tuple(sorted({request.policy_root, *(content_identity(row) for row in domains.values())})), selected


def _declared_closure(graph: PromptGoalGraph, manifest: dict, tree_id: str) -> dict:
    """Native closure of verified *declared inputs*, with no code-proof claim."""
    from ..analysis.semantic_dependency_graph import (
        SemanticDependencyGraph,
        SemanticNode,
        SemanticEdge,
        compute_mandatory_closure,
    )

    decision = content_identity({"local_planning_graph": graph.content_id, "tree": tree_id})
    common = {
        "root_id": tree_id,
        "provenance": "source",
        "trust": "verified",
        "authority": "descriptive_input",
        "version": "local-declared-inputs@1",
    }
    nodes = [
        SemanticNode(
            node_id=decision, kind="decision", record={"graph_cid": graph.content_id}, **common
        )
    ]
    edges = []
    declarations = [
        ("file", {"path": path, **source}) for path, source in manifest["sources"].items()
    ]
    declarations += [("action", spec) for spec in manifest["tasks"]]
    declarations += [
        (
            "authorization",
            {"profile_content_id": manifest["profile_content_id"], "policy": manifest["policy"]},
        )
    ]
    if "planning_inputs" in manifest:
        inputs = manifest["planning_inputs"]
        declarations += [
            ("premise", {"input_kind": key, "content_cid": value})
            for key, value in manifest["planning_roots"].items()
        ]
        declarations += [
            ("premise", {"input_kind": "local-descriptive-domain", "domain": domain,
                         "content_cid": content_identity(value)})
            for domain, value in inputs["domain_declarations"].items()
        ]
        declarations += [
            ("premise", {"input_kind": "descriptive-evidence",
                         "content_cid": PromptEvidenceRecord.from_dict(value).evidence_cid,
                         "artifact_cid": value["artifact_cid"]})
            for value in inputs["selected_evidence"]
        ]
    for kind, record in declarations:
        node_id = content_identity({"kind": kind, "declaration": record})
        nodes.append(SemanticNode(node_id=node_id, kind=kind, record=record, **common))
        edges.append(
            SemanticEdge(
                source=decision,
                target=node_id,
                kind="requires",
                provenance_id=content_identity(record),
                **common,
            )
        )
    native = SemanticDependencyGraph(root_id=tree_id, nodes=tuple(nodes), edges=tuple(edges))
    closure = compute_mandatory_closure(native, decision)
    if set(closure.node_ids) != {node.node_id for node in native.nodes}:
        raise LocalPlanningError("declared mandatory inputs were omitted")
    return {
        "scope": "signed-declarations-and-complete-tracked-file-inventory",
        "semantic_dependency_graph": native.to_dict(),
        "closure": closure.to_dict(),
        "code_semantic_closure_claimed": False,
        "proof_authority": False,
    }


def _graph_contract(graph: PromptGoalGraph, manifest: dict, tree_id: str) -> tuple[Any, dict, list]:
    policy_roots = (content_identity(LOCAL_POLICY),)
    allowed_evidence = ()
    if "planning_inputs" in manifest:
        policy_roots, allowed_evidence = _verify_planning_inputs(manifest)
    if (graph.unresolved_questions or graph.uncertainty_debt
            or [item.to_dict() for item in graph.evidence] != [item.to_dict() for item in allowed_evidence]):
        raise LocalPlanningError("unresolved or externally evidenced graph unsupported")
    if any(getattr(graph, key) != value for key, value in manifest["planning_roots"].items()):
        raise LocalPlanningError("planning roots differ from signed inputs")
    if graph.policy_roots != policy_roots:
        raise LocalPlanningError("external policy or proof obligations unsupported")
    specs = {item["task_key"]: item for item in manifest["tasks"]}
    tasks = {task.task_key: task for task in graph.tasks}
    if set(specs) != set(tasks):
        raise LocalPlanningError("proposal changed required task population")
    all_acceptance = []
    for key, task in tasks.items():
        spec = specs[key]
        observed = {
            "task_key": key,
            "scope_paths": list(task.scope_paths),
            "outputs": [
                {name: getattr(row, name) for name in ("path", "effect", "media_type")}
                for row in task.outputs
            ],
            "validations": [
                {
                    name: _plain(getattr(row, name))
                    for name in (
                        "validation_key",
                        "argv",
                        "cwd",
                        "expected_exit_codes",
                        "policy_cid",
                    )
                }
                for row in task.validations
            ],
            "acceptance": [
                {
                    name: _plain(getattr(row, name))
                    for name in ("criterion_key", "criterion", "evidence_cids", "validation_keys")
                }
                for row in task.acceptance
            ],
            "dependencies": sorted(
                next(k for k, t in tasks.items() if t.task_cid == dep)
                for dep in task.dependency_task_cids
            ),
        }

        def ordered(value):
            return {
                name: sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))
                if isinstance(rows, list)
                else rows
                for name, rows in value.items()
            }

        if (
            ordered(observed) != ordered(spec)
            or task.assumptions
            or not set(task.evidence_cids) <= {item.evidence_cid for item in allowed_evidence}
            or task.policy_roots != policy_roots
        ):
            raise LocalPlanningError(
                "proposal changed signed scope, acceptance, dependency or command"
            )
        if not set(task.predicted_files) <= set(spec["scope_paths"]):
            raise LocalPlanningError("predicted change escaped permitted scope")
        all_acceptance.extend(task.acceptance)
    # Goal reviews are only aggregation of the exact task tests in this policy;
    # free-form review/proof goals cannot be silently deferred.
    for goal in graph.goals:
        if (
            goal.assumptions
            or not set(goal.evidence_cids) <= {item.evidence_cid for item in allowed_evidence}
            or not set(goal.scope_paths) <= set(manifest["sources"]) | set(manifest.get("created_outputs", ()))
            or any(item not in all_acceptance for item in goal.acceptance)
        ):
            raise LocalPlanningError("goal acceptance is not an exact aggregation of task tests")
    compiled = FormalPlanCompiler().compile_prompt_graph(
        graph, repository_tree_id=tree_id, actor_id=manifest["profile_content_id"]
    )
    if compiled.plan is None or compiled.status.value != "compiled" or compiled.proof_results:
        raise LocalPlanningError("native formal compilation did not produce an unproved plan")
    validation = validate_formal_plan(compiled.plan, compiled.formulas)
    if validation.status.value != "consistent":
        raise LocalPlanningError("native bounded plan is not consistent")
    pending = []
    for requirement in compiled.plan.evidence_requirements:
        if (
            requirement.kind.value not in {"test", "review"}
            or requirement.minimum_code_assurance.value != "candidate"
            or not requirement.fallback_check_ids
        ):
            raise LocalPlanningError("non-test or proof-assurance obligation cannot be deferred")
        pending.append({"phase": "post_execution", "required": True, **requirement.to_dict()})
    return compiled, validation.to_dict(), pending


def admit_local_benchmark_plan(
    *, graph: PromptGoalGraph, manifest: Mapping, requirement_bindings: Sequence[Mapping] | None = None,
    applicability_timeout_seconds: float = 45., source_applicability_nomination=None,
) -> dict:
    """Verify bounded planning permission; future task acceptance remains pending."""
    declared, profile, current = _manifest(manifest, initial=True)
    graph = PromptGoalGraph.from_dict(graph.to_dict())
    payload = _planning_payload(graph, manifest, declared, profile, current, requirement_bindings,
        applicability_timeout_seconds=applicability_timeout_seconds,
        source_applicability_nomination=source_applicability_nomination)
    _post_header_manifest(manifest, declared, current, initial=True)
    admission = {"manifest": _plain(manifest), "graph": graph.to_dict(), "receipt": _signed(payload, declared)}
    if declared["schema"] == INTENT_MANIFEST_SCHEMA:
        admission["requirement_bindings"] = _plain(requirement_bindings)
    return admission


def _planning_payload(graph, manifest, declared, profile, sources, requirement_bindings=None,
                      *, requirement_source_text=None, applicability_timeout_seconds=45.,
                      source_applicability_nomination=None) -> dict:
    if source_applicability_nomination is not None and not _has_header_contract(declared):
        raise LocalPlanningError("runtime nomination requires explicit header intent contract")
    if requirement_source_text is not None:
        _validate_local_manifest_declarations(declared)
    compiled, validation, pending = _graph_contract(graph, declared, _tree(sources))
    payload = {
        "schema": PLANNING_RECEIPT_SCHEMA,
        "manifest_cid": content_identity(manifest),
        "graph_cid": graph.content_id,
        "plan_id": compiled.plan.plan_id,
        "source_tree_id": _tree(sources),
        "pending_requirements": pending,
        "pending_cid": content_identity(pending),
        "declared_input_closure": _declared_closure(graph, declared, _tree(sources)),
        "plan_evidence": validation,
        "owner_profile_id": profile.profile_id,
        "planning_permitted": True,
        "completion_authority": False,
        "code_proof_authority": False,
        "production_activation": False,
    }
    if declared["schema"] == INTENT_MANIFEST_SCHEMA:
        from ..prompt.intent_plan_coverage import check_intent_plan_coverage

        if not isinstance(requirement_bindings, list):
            raise LocalPlanningError("intent admission requires explicit requirement bindings")
        try:
            requirements = (_verify_intent_requirements(declared) if requirement_source_text is None
                else _verify_intent_requirement_text(declared, requirement_source_text))
            coverage = check_intent_plan_coverage(
                requirements, graph=graph, bindings=requirement_bindings,
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise LocalPlanningError("invalid intent plan coverage: " + str(exc)) from exc
        if coverage.get("accepted") is not True:
            error = LocalPlanningError("intent plan does not cover the signed requirements: " + str(coverage.get("errors", [])))
            error.requirement_coverage = coverage
            raise error
        payload.update(schema=INTENT_PLANNING_RECEIPT_SCHEMA, requirement_coverage=coverage)
        if requirements["schema"] in {"intent-plan-requirement-contract@2", "intent-plan-requirement-contract@3"}:
            planned = _replay_intent_symbolic_plan(requirements, manifest,
                applicability_timeout_seconds=applicability_timeout_seconds,
                source_applicability_nomination=source_applicability_nomination)
            if graph.to_dict() != planned["graph"].to_dict() or coverage != planned["coverage"]:
                raise LocalPlanningError("intent plan differs from deterministic symbolic selection")
            payload["intent_symbolic_planning"] = planned["receipt"]
    elif requirement_bindings is not None:
        raise LocalPlanningError("requirement bindings require the intent manifest version")
    return payload


def _replay_intent_symbolic_plan(requirements, manifest, *, applicability_timeout_seconds=45.,
                                 source_applicability_nomination=None):
    """Replay proposals against verified signed baseline inputs, without a provider."""
    from ..planning.intent_symbolic_planning import build_intent_symbolic_plan

    try:
        return build_intent_symbolic_plan(requirements, manifest=manifest,
            applicability_timeout_seconds=applicability_timeout_seconds,
            source_applicability_nomination=source_applicability_nomination)
    except (ValueError, TypeError, KeyError) as exc:
        error = LocalPlanningError("invalid symbolic intent planning: " + str(exc))
        for field in ("symbolic_issues", "requirement_coverage"):
            if hasattr(exc, field):
                setattr(error, field, getattr(exc, field))
        raise error from exc


def _has_header_contract(declared):
    return (declared.get("schema") == INTENT_MANIFEST_SCHEMA
        and decode_intent_requirement_contract(declared)["schema"] == "intent-plan-requirement-contract@3")


def _header_nomination(payload):
    return payload.get("intent_symbolic_planning", {}).get("source_applicability_nomination")


def _post_header_manifest(envelope, declared, before, **mode):
    if _has_header_contract(declared):
        after, _, current = _manifest(envelope, **mode)
        if after != declared or _tree(current) != _tree(before):
            raise LocalPlanningError("source changed during external header applicability replay")
        from .header_intent_applicability import require_applicability_budget
        require_applicability_budget()


def _require_admission_fields(admission: Mapping) -> None:
    expected_keys = {"manifest", "graph", "receipt"}
    if admission.get("manifest", {}).get("payload", {}).get("schema") == INTENT_MANIFEST_SCHEMA:
        expected_keys.add("requirement_bindings")
    if set(admission) != expected_keys:
        raise LocalPlanningError("exact local admission bundle required")


def verify_local_benchmark_admission(admission: Mapping, *, initial: bool = True) -> dict:
    """Revalidate an existing local admission without signing or materializing."""
    _require_admission_fields(admission)
    manifest, profile, current = _manifest(admission["manifest"], initial=initial)
    graph = PromptGoalGraph.from_dict(admission["graph"])
    receipt = _verify_signature(admission["receipt"], profile)
    expected = _planning_payload(
        graph, admission["manifest"], manifest, profile, manifest["sources"],
        admission.get("requirement_bindings"),
        source_applicability_nomination=_header_nomination(receipt),
    )
    _post_header_manifest(admission["manifest"], manifest, current, initial=initial)
    if receipt != expected:
        raise LocalPlanningError("planning receipt does not match recomputed contract")
    return {"manifest": manifest, "profile": profile, "receipt": receipt,
            "graph": graph, "current_source_tree_id": _tree(current)}


def _receipt_bytes(envelope: Mapping) -> bytes:
    return (json.dumps(envelope, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _receipt_reference(envelope: Mapping, raw: bytes) -> dict:
    payload = envelope["payload"]
    reference = {
        "schema": PLANNING_RECEIPT_REFERENCE_SCHEMA,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "receipt_cid": content_identity(envelope),
        **{key: payload[key] for key in (
            "manifest_cid", "graph_cid", "plan_id", "source_tree_id",
            "pending_cid", "pending_requirements",
        )},
        "completion_authority": False,
        "code_proof_authority": False,
        "production_activation": False,
    }
    if payload["schema"] == INTENT_PLANNING_RECEIPT_SCHEMA:
        coverage = payload["requirement_coverage"]
        reference.update(
            schema=INTENT_PLANNING_RECEIPT_REFERENCE_SCHEMA,
            requirement_contract_cid=coverage["contract_cid"],
            requirement_coverage_cid=content_identity(coverage),
        )
    return reference


def _receipt_artifact_path(manifest: Mapping, digest: str, *, create=False) -> Path:
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise LocalPlanningError("planning receipt digest must be canonical SHA256")
    lifecycle = Path(manifest["lifecycle_dir"])
    directory = lifecycle / "local-planning-receipts"
    if lifecycle.resolve(strict=True) != lifecycle:
        raise LocalPlanningError("planning receipt lifecycle path is not canonical")
    if create:
        directory.mkdir(mode=0o700, exist_ok=True)
    if directory.resolve(strict=True) != directory:
        raise LocalPlanningError("planning receipt directory must not be symlinked")
    observed = directory.stat()
    if observed.st_uid != os.geteuid() or stat.S_IMODE(observed.st_mode) != 0o700:
        raise LocalPlanningError("planning receipt directory must be private to its owner")
    return directory / (digest + ".json")


def load_local_planning_receipt(reference: Mapping, *, manifest: Mapping) -> dict:
    """Resolve owner storage without treating a reference as admission authority.

    The full signed receipt and its complete declared-input closure remain
    available outside the worker repository. Native plan projections retain
    only its content identity and exact pending checks, not a truncated proof.
    """
    declared, profile, current = _manifest(manifest)
    expected_keys = {
        "schema", "sha256", "bytes", "receipt_cid", "manifest_cid", "graph_cid",
        "plan_id", "source_tree_id", "pending_cid", "pending_requirements",
        "completion_authority", "code_proof_authority", "production_activation",
    }
    schema = PLANNING_RECEIPT_REFERENCE_SCHEMA
    receipt_schema = PLANNING_RECEIPT_SCHEMA
    if declared["schema"] == INTENT_MANIFEST_SCHEMA:
        expected_keys.update({"requirement_contract_cid", "requirement_coverage_cid"})
        schema = INTENT_PLANNING_RECEIPT_REFERENCE_SCHEMA
        receipt_schema = INTENT_PLANNING_RECEIPT_SCHEMA
    if (not isinstance(reference, Mapping) or set(reference) != expected_keys
            or reference.get("schema") != schema
            or type(reference.get("bytes")) is not int
            or not 0 < reference["bytes"] <= MAX_PLANNING_RECEIPT_BYTES
            or len(_receipt_bytes(reference)) > 32768):
        raise LocalPlanningError("planning receipt reference is not closed and bounded")
    path = _receipt_artifact_path(declared, reference["sha256"])
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        observed = os.fstat(stream.fileno())
        if (not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1
                or observed.st_uid != os.geteuid()
                or stat.S_IMODE(observed.st_mode) != 0o600
                or observed.st_size != reference["bytes"]):
            raise LocalPlanningError("planning receipt artifact is not exact private owner storage")
        raw = stream.read(MAX_PLANNING_RECEIPT_BYTES + 1)
    if (len(raw) != reference["bytes"]
            or hashlib.sha256(raw).hexdigest() != reference["sha256"]):
        raise LocalPlanningError("planning receipt artifact bytes changed")
    envelope = json.loads(raw)
    payload = _verify_signature(envelope, profile)
    if (_receipt_bytes(envelope) != raw
            or _receipt_bytes(_receipt_reference(envelope, raw)) != _receipt_bytes(reference)
            or payload.get("schema") != receipt_schema
            or payload.get("manifest_cid") != content_identity(manifest)
            or payload.get("owner_profile_id") != profile.profile_id
            or payload.get("planning_permitted") is not True
            or any(payload.get(key) is not False for key in (
                "completion_authority", "code_proof_authority", "production_activation"))):
        raise LocalPlanningError("planning receipt artifact signature or reference bindings differ")
    if declared["schema"] == INTENT_MANIFEST_SCHEMA and (
        payload["requirement_coverage"].get("accepted") is not True
        or payload["requirement_coverage"].get("contract_cid") != declared["intent_requirements"]["contract_cid"]
    ):
        raise LocalPlanningError("planning receipt requirement contract differs")
    if declared["schema"] == INTENT_MANIFEST_SCHEMA:
        requirements = _verify_intent_requirements(declared)
        if requirements["schema"] in {"intent-plan-requirement-contract@2", "intent-plan-requirement-contract@3"}:
            planned = _replay_intent_symbolic_plan(requirements, manifest,
                source_applicability_nomination=_header_nomination(payload))
            if (payload.get("intent_symbolic_planning") != planned["receipt"]
                    or payload["requirement_coverage"] != planned["coverage"]
                    or payload["graph_cid"] != planned["graph"].content_id):
                raise LocalPlanningError("planning receipt differs from replayed symbolic selection")
        elif "intent_symbolic_planning" in payload:
            raise LocalPlanningError("symbolic receipt requires the symbolic requirement contract")
    _post_header_manifest(manifest, declared, current)
    return envelope


def _store_local_planning_receipt(envelope: Mapping, *, manifest: Mapping) -> dict:
    raw = _receipt_bytes(envelope)
    reference = _receipt_reference(envelope, raw)
    if len(raw) > MAX_PLANNING_RECEIPT_BYTES or len(_receipt_bytes(reference)) > 32768:
        raise LocalPlanningError("planning receipt artifact or pending summary exceeds bound")
    if (manifest["payload"]["schema"] == INTENT_MANIFEST_SCHEMA
            and decode_intent_requirement_contract(manifest["payload"])["schema"] == "intent-plan-requirement-contract@3"):
        from .header_intent_applicability import require_applicability_budget
        require_applicability_budget()
    path = _receipt_artifact_path(manifest["payload"], reference["sha256"], create=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        pass  # Never replace existing bytes; the mandatory reload verifies them.
    else:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    if load_local_planning_receipt(reference, manifest=manifest) != envelope:
        raise LocalPlanningError("planning receipt storage changed the signed envelope")
    return reference


def _contract(task_body: Mapping, task_cid: str, *, source_transition=None) -> tuple[dict, dict, Any, dict]:
    envelope = task_body[CONTRACT_KEY]
    payload = envelope.get("payload", {})
    manifest, profile, current = _manifest(
        payload.get("manifest", {}), source_transition=source_transition,
    )
    payload = _verify_signature(envelope, profile)
    expected = {
        "schema",
        "task_cid",
        "task_key",
        "manifest",
        "manifest_cid",
        "plan_id",
        "graph_cid",
        "pending_requirements",
        "pending_cid",
        "dependencies",
        "task_spec",
        "planning_receipt_cid",
        "intent_owner_id",
    }
    schema = CONTRACT_SCHEMA
    if manifest["schema"] == INTENT_MANIFEST_SCHEMA:
        expected.add("intent_plan")
        schema = INTENT_CONTRACT_SCHEMA
    if (
        set(payload) != expected
        or payload.get("schema") != schema
        or payload.get("task_cid") != task_cid
    ):
        raise LocalPlanningError("local completion contract task/schema mismatch")
    if not payload["pending_requirements"] or any(
        row.get("phase") != "post_execution"
        or row.get("required") is not True
        or row.get("kind") not in {"test", "review"}
        or row.get("minimum_code_assurance") != "candidate"
        for row in payload["pending_requirements"]
    ):
        raise LocalPlanningError("pending contract cannot defer proof or omit acceptance")
    if (
        payload["task_spec"] not in manifest["tasks"]
        or payload["task_spec"]["task_key"] != payload["task_key"]
    ):
        raise LocalPlanningError("pending task spec differs from independent manifest")
    if content_identity(payload["pending_requirements"]) != payload["pending_cid"]:
        raise LocalPlanningError("pending completion identity mismatch")
    if payload["manifest_cid"] != content_identity(payload["manifest"]):
        raise LocalPlanningError("manifest binding changed")
    if manifest["schema"] == INTENT_MANIFEST_SCHEMA:
        intent_plan = payload["intent_plan"]
        symbolic = decode_intent_requirement_contract(manifest)["schema"] in {"intent-plan-requirement-contract@2", "intent-plan-requirement-contract@3"}
        plan_fields = {"schema", "graph", "requirement_bindings", "coverage"}
        if symbolic:
            plan_fields.add("symbolic_planning")
        if (not isinstance(intent_plan, Mapping)
                or set(intent_plan) != plan_fields
                or intent_plan["schema"] != ("supervisor-local-intent-plan@2" if symbolic else "supervisor-local-intent-plan@1")):
            raise LocalPlanningError("intent pending contract must retain exact plan coverage")
        graph = PromptGoalGraph.from_dict(intent_plan["graph"])
        receipt = _planning_payload(
            graph, payload["manifest"], manifest, profile, manifest["sources"],
            intent_plan["requirement_bindings"],
            source_applicability_nomination=intent_plan.get("symbolic_planning", {}).get("source_applicability_nomination"),
        )
        if (intent_plan["coverage"] != receipt["requirement_coverage"]
                or (symbolic and intent_plan["symbolic_planning"] != receipt["intent_symbolic_planning"])
                or any(payload[key] != receipt[key] for key in (
                    "graph_cid", "plan_id", "pending_cid", "pending_requirements"))
                or not any(task.task_cid == task_cid and task.task_key == payload["task_key"]
                           and list(task.dependency_task_cids) == payload["dependencies"]
                           for task in graph.tasks)):
            raise LocalPlanningError("intent pending contract differs from recomputed plan coverage")
    if source_transition is not None and (
        source_transition["payload"].get("task_cid") != task_cid
        or source_transition["payload"].get("contract_cid") != content_identity(envelope)
    ):
        raise LocalPlanningError("source transition belongs to another task contract")
    _post_header_manifest(payload["manifest"], manifest, current, source_transition=source_transition)
    return payload, manifest, profile, current


def _pending_contract_payload(*, admission, verified, task, intent_owner_id) -> dict:
    """Reconstruct the complete pending contract from verified admission."""
    manifest, graph, receipt = verified["manifest"], verified["graph"], verified["receipt"]
    spec = next(row for row in manifest["tasks"] if row["task_key"] == task.task_key)
    contract = {
        "schema": CONTRACT_SCHEMA,
        "task_cid": task.task_cid,
        "task_key": task.task_key,
        "manifest": admission["manifest"],
        "manifest_cid": receipt["manifest_cid"],
        "plan_id": receipt["plan_id"],
        "graph_cid": graph.content_id,
        "pending_requirements": receipt["pending_requirements"],
        "pending_cid": receipt["pending_cid"],
        "dependencies": list(task.dependency_task_cids),
        "task_spec": spec,
        "planning_receipt_cid": content_identity(admission["receipt"]),
        "intent_owner_id": intent_owner_id,
    }
    if manifest["schema"] == INTENT_MANIFEST_SCHEMA:
        contract.update(schema=INTENT_CONTRACT_SCHEMA, intent_plan={
            "schema": "supervisor-local-intent-plan@1",
            "graph": graph.to_dict(),
            "requirement_bindings": _plain(admission["requirement_bindings"]),
            "coverage": receipt["requirement_coverage"],
        })
        if "intent_symbolic_planning" in receipt:
            contract["intent_plan"].update(
                schema="supervisor-local-intent-plan@2",
                symbolic_planning=receipt["intent_symbolic_planning"],
            )
    return contract


def materialize_local_benchmark_plan(*, admission: Mapping, intent) -> dict:
    declared = admission.get("manifest", {}).get("payload", {})
    if (declared.get("schema") == INTENT_MANIFEST_SCHEMA
            and decode_intent_requirement_contract(declared)["schema"] == "intent-plan-requirement-contract@3"):
        from .header_intent_applicability import applicability_budget, require_applicability_budget
        # One bound covers verification and stored-receipt replay. When called
        # by the planner this can only tighten its inherited remaining budget.
        with applicability_budget(45.):
            require_applicability_budget()
            return _materialize_local_transaction(admission=admission, intent=intent)
    return _materialize_local_transaction(admission=admission, intent=intent)


def _materialize_local_transaction(*, admission: Mapping, intent) -> dict:
    """Recheck the signed source and materialize native tasks with pending gates."""
    from ..task_sources.intent_repository import IntentRepository

    if not isinstance(intent, IntentRepository) or intent.uses_bound_connection:
        raise LocalPlanningError(
            "materialization requires an independently owned native transaction"
        )
    # Nothing becomes schedulable until the complete graph and final source
    # check commit through the same owner transaction.
    with intent._connection(write=True) as connection:
        with IntentRepository(
            bound_connection=connection, owner_id=intent.owner_id, session_id=intent.session_id
        ) as owner:
            result = _materialize_local_benchmark_plan(admission=admission, intent=owner)
            _manifest(admission["manifest"], initial=True)
            if (admission["manifest"]["payload"]["schema"] == INTENT_MANIFEST_SCHEMA
                    and decode_intent_requirement_contract(admission["manifest"]["payload"])["schema"]
                        == "intent-plan-requirement-contract@3"):
                from .header_intent_applicability import require_applicability_budget
                require_applicability_budget()
            return result


def _materialize_local_benchmark_plan(*, admission: Mapping, intent) -> dict:
    verified = verify_local_benchmark_admission(admission)
    manifest, graph, receipt = verified["manifest"], verified["graph"], verified["receipt"]
    if any(intent.get_task(task.task_cid) is not None for task in graph.tasks):
        raise LocalPlanningError("local plan requires new native tasks")
    receipt_reference = _store_local_planning_receipt(
        admission["receipt"], manifest=admission["manifest"],
    )
    objective = content_identity(
        {"manifest": receipt["manifest_cid"], "objective": graph.root_goal.objective}
    )
    intent.upsert_objective(
        objective_id=objective,
        objective_alias="LOCAL-OBJECTIVE-" + objective[-12:],
        title=graph.root_goal.objective,
    )
    remaining = {goal.goal_cid: goal for goal in graph.goals}
    installed = set()
    while remaining:
        for cid, goal in tuple(remaining.items()):
            if goal.parent_goal_cid and goal.parent_goal_cid not in installed:
                continue
            intent.upsert_goal(
                goal_cid=cid,
                goal_alias=goal.goal_key,
                title=goal.title,
                objective_id=objective,
                parent_goal_cid=goal.parent_goal_cid,
                body={"local_manifest_cid": receipt["manifest_cid"]},
            )
            installed.add(cid)
            del remaining[cid]
    intent.upsert_plan(
        plan_cid=receipt["plan_id"],
        plan_alias="LOCAL-PLAN-" + receipt["plan_id"][-12:],
        goal_cid=graph.root_goal.goal_cid,
        status="active",
        body={"local_planning_receipt_ref": receipt_reference},
    )
    tasks = {task.task_cid: task for task in graph.tasks}
    installed = set()
    while tasks:
        for cid, task in tuple(tasks.items()):
            if not set(task.dependency_task_cids) <= installed:
                continue
            spec = next(row for row in manifest["tasks"] if row["task_key"] == task.task_key)
            contract = _pending_contract_payload(
                admission=admission, verified=verified, task=task, intent_owner_id=intent.owner_id,
            )
            body = {"title": task.objective, CONTRACT_KEY: _signed(contract, manifest)}
            intent.upsert_task(
                task_cid=cid,
                task_alias=task.task_key,
                goal_cid=task.goal_cid,
                objective_id=objective,
                plan_cid=receipt["plan_id"],
                body=body,
                identity={"local_contract_cid": content_identity(body[CONTRACT_KEY]),
                          "repository_tree_id": receipt["source_tree_id"]},
                dependencies=contract["dependencies"],
                outputs=spec["outputs"],
                acceptance=spec["acceptance"],
                validations=spec["validations"],
            )
            installed.add(cid)
            del tasks[cid]
    return {
        "schema": "supervisor-local-materialization@1",
        "plan_id": receipt["plan_id"],
        "task_cids": sorted(installed),
        "pending_cid": receipt["pending_cid"],
        "manifest_cid": receipt["manifest_cid"],
        "completion_authority": False,
    }


def guard_local_task_update(
    *,
    previous: Mapping,
    body: Mapping,
    identity: Mapping,
    task_cid: str,
    status: str,
    dependencies=None,
    outputs=None,
    acceptance=None,
    validations=None,
) -> None:
    """Canonical-owner seam: an opt-in task cannot drop or rewrite its gate."""
    if CONTRACT_KEY not in previous and CONTRACT_KEY not in body:
        return
    if CONTRACT_KEY in previous and body.get(CONTRACT_KEY) != previous[CONTRACT_KEY]:
        raise LocalPlanningError("local pending contract cannot be removed or replaced")
    contract, _, _, _ = _contract(body, task_cid)
    if identity.get("local_contract_cid") != content_identity(body[CONTRACT_KEY]):
        raise LocalPlanningError("native task identity must retain exact local contract")
    if status in {"done", "completed", "complete", "skipped"}:
        raise LocalPlanningError("local completion requires canonical evidence-gated transition")
    spec = contract["task_spec"]
    for supplied, expected in (
        (dependencies, contract["dependencies"]),
        (outputs, spec["outputs"]),
        (acceptance, spec["acceptance"]),
        (validations, spec["validations"]),
    ):
        if supplied is not None and _plain(supplied) != expected:
            raise LocalPlanningError("native relation differs from exact pending contract")


def run_local_task_validations(
    *, intent, task_cid: str, attempt_id: str, timeout: float = 60, source_transition=None,
    candidate_runner=None,
) -> dict:
    """Execute the signed checks and record observed, owner-signed evidence.

    This performs no status transition. Failed results remain failed. A caller
    must still use the native owner/fencing/claim completion boundary.
    """
    task = intent.get_task(task_cid)
    if task is None or task["status"] != "in_progress" or not attempt_id:
        raise LocalPlanningError("an in-progress native task and attempt are required")
    contract, manifest, _, before = _contract(
        task["body"], task_cid, source_transition=source_transition,
    )
    if not 0 < timeout <= 300:
        raise LocalPlanningError("validation timeout outside local bound")
    if candidate_runner is not None:
        from .candidate_execution import verify_candidate_runner
        verify_candidate_runner(candidate_runner)
    results = []
    for check in contract["task_spec"]["validations"]:
        root = Path(manifest["repository"])
        cwd = root / check["cwd"]
        if cwd.resolve(strict=True) != cwd or not cwd.is_dir():
            raise LocalPlanningError("validation cwd is not exact repository directory")
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                run_options = dict(
                    cwd=cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout,
                    check=False,
                    env={
                        "PATH": os.defpath,
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONHASHSEED": "0",
                        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                        "LANG": "C.UTF-8",
                    },
                )
                if candidate_runner is None:
                    command = check["argv"]
                    if command[0] in {"python", "python3"}:
                        # Match the native pre-merge launcher contract. Slim
                        # containers keep Python outside os.defpath; do not
                        # restore ambient PATH or consult task executables.
                        from ..validation.validation_runtime import validation_argv_command
                        command = validation_argv_command(command, environment=run_options["env"])
                    process = subprocess.run(command, **run_options)
                else:
                    from .candidate_execution import run_candidate
                    process = run_candidate(
                        candidate_runner, check["argv"], cwd=cwd, timeout=timeout,
                        stdout=stdout, stderr=stderr,
                    )
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                exit_code = None

            def digest(stream):
                stream.seek(0)
                hasher = hashlib.sha256()
                for block in iter(lambda: stream.read(65536), b""):
                    hasher.update(block)
                return hasher.hexdigest()

            stdout_hash, stderr_hash = digest(stdout), digest(stderr)
        _, _, after = _manifest(contract["manifest"], source_transition=source_transition)
        current_task = intent.get_task(task_cid)
        if (
            before != after
            or current_task["revision"] != task["revision"]
            or current_task["body"].get(CONTRACT_KEY) != task["body"][CONTRACT_KEY]
        ):
            raise LocalPlanningError("source or native task changed during validation")
        payload = {
            "schema": RESULT_SCHEMA,
            "task_cid": task_cid,
            "task_revision": task["revision"],
            "attempt_id": attempt_id,
            "contract_cid": content_identity(task["body"][CONTRACT_KEY]),
            "intent_owner_id": intent.owner_id,
            "manifest_cid": contract["manifest_cid"],
            "pending_cid": contract["pending_cid"],
            "source_tree_id": _tree(after),
            "validation": check,
            "exit_code": exit_code,
            "stdout_sha256": stdout_hash,
            "stderr_sha256": stderr_hash,
            "outcome": "passed" if exit_code == 0 else "failed",
        }
        if source_transition is not None:
            payload["source_transition"] = _plain(source_transition)
        if candidate_runner is not None:
            payload["candidate_runner"] = _plain(candidate_runner)
        signed = _signed(payload, manifest)
        evidence_digest = content_identity(signed)
        intent.record_validation_result(
            task_cid=task_cid,
            outcome=payload["outcome"],
            evidence_digest=evidence_digest,
            argv=check["argv"],
            attempt_id=attempt_id,
            body={"local_observed_validation": signed},
        )
        results.append(
            {
                "validation_key": check["validation_key"],
                "outcome": payload["outcome"],
                "evidence_digest": evidence_digest,
            }
        )
    return {
        "task_cid": task_cid,
        "source_tree_id": _tree(before),
        "results": results,
        "passed": all(row["outcome"] == "passed" for row in results),
    }


def local_completion_missing(
    connection, task_cid: str, body: Mapping, revision: int
) -> tuple[str, ...]:
    """Native transaction gate: accept only observed checks for current bytes."""
    if CONTRACT_KEY not in body:
        return ()
    try:
        rows = connection.execute(
            "SELECT r.outcome, r.evidence_digest, r.body_json FROM validation_results r "
            "JOIN domain_events e ON e.task_cid = r.task_cid "
            "AND e.event_type = 'intent.validation_recorded' "
            "AND json_extract_string(e.body_json, '$.subject_id') = r.result_id "
            "WHERE r.task_cid = ? ORDER BY e.global_sequence DESC",
            [task_cid],
        ).fetchall()
        # Source-transition authority is accepted only from a signed observed
        # validation result, never from a generic completion receipt. Every
        # candidate is independently checked against the immutable contract.
        contract = None
        wanted, passed, seen, missing_outputs = {}, set(), set(), set()
        for row in rows:
            result = json.loads(row[2]).get("local_observed_validation")
            if not result or content_identity(result) != row[1]:
                continue
            transition = result.get("payload", {}).get("source_transition")
            try:
                candidate, _, profile, current = _contract(
                    body, task_cid, source_transition=transition,
                )
                observed = _verify_signature(result, profile)
            except (ValueError, KeyError, TypeError, OSError):
                continue
            contract = candidate
            wanted = {item["validation_key"]: item for item in contract["task_spec"]["validations"]}
            missing_outputs = {
                item["path"] for item in contract["task_spec"]["outputs"]
                if item["effect"] == "create" and item["path"] not in current
            }
            check = observed.get("validation", {})
            key = check.get("validation_key")
            claim = body.get("completion_receipt", {})
            current_attempt = claim.get("attempt_id")
            if (
                observed.get("schema") == RESULT_SCHEMA
                and observed.get("task_cid") == task_cid
                and observed.get("task_revision") == revision
                and observed.get("contract_cid") == content_identity(body[CONTRACT_KEY])
                and observed.get("manifest_cid") == contract["manifest_cid"]
                and observed.get("pending_cid") == contract["pending_cid"]
                and observed.get("source_tree_id") == _tree(current)
                and observed.get("intent_owner_id") == contract["intent_owner_id"]
                and observed.get("attempt_id")
                and (not current_attempt or observed["attempt_id"] == current_attempt)
                and (transition is None or (
                    transition["payload"].get("task_revision") == revision
                    and transition["payload"].get("attempt_id") == observed["attempt_id"]
                ))
                and key in wanted
                and check == wanted[key]
                and key not in seen
            ):
                seen.add(key)
                if observed.get("outcome") == row[0] == "passed" and observed.get("exit_code") == 0:
                    passed.add(key)
        if contract is None:
            contract, _, _, current = _contract(body, task_cid)
            wanted = {item["validation_key"]: item for item in contract["task_spec"]["validations"]}
            missing_outputs = {
                item["path"] for item in contract["task_spec"]["outputs"]
                if item["effect"] == "create" and item["path"] not in current
            }
        return tuple(
            ["local-validation:" + key for key in sorted(set(wanted) - passed)]
            + ["local-output:" + path for path in sorted(missing_outputs)]
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return ("local-contract:" + type(exc).__name__,)
