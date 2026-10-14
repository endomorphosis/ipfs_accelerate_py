"""Deliver an explicitly admitted public task requirement to a coding worker.

The owner verifies admission and pins this artifact in the implementation
command. The worker replays the manifest signature and exact source bindings;
it does not acquire owner credentials or treat requirements as execution policy.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

from ..proof.formal_verification_contracts import content_identity
from . import local_planning_admission as local
from .doctor_candidate_runner import _directory, _git, _read, _unique

SCHEMA = "supervisor-public-instruction@1"
INTENT_SCHEMA = "supervisor-public-instruction@2"
INVENTORY_SCHEMA = "supervisor-public-instruction@3"
SUCCESSOR_SCHEMA = "supervisor-public-instruction@4"
SCOPED_SCHEMA = "supervisor-public-instruction@5"
SCOPED_INCLUSION_SCHEMA = "supervisor-public-instruction-inclusion@5"
SCOPE_INCLUSION_SCHEMA = "terminal-signed-semantic-worker-inclusion@1"
DIRECTORY = ".runtime/router-public-instruction"
MAX_BYTES = 1_000_000
MAX_INVENTORY_BYTES = 8 * 1024 * 1024
MAX_SUCCESSOR_BYTES = 16 * 1024 * 1024
MAX_INSTRUCTION_BYTES = 32_768
FIELDS = frozenset({"schema", "repository", "task_cid", "task_id", "manifest", "manifest_cid",
    "owner_identity", "owner_profile_id", "source_path", "source_sha256", "source_bytes",
    "completion_authority", "publication_authority", "scope_expansion_authority", "context_cid"})
INTENT_FIELDS = FIELDS | {"intent_plan_admission"}
INVENTORY_FIELDS = FIELDS | {"inventory_plan_admission", "codebase_inventory_context"}
SUCCESSOR_FIELDS = INVENTORY_FIELDS | {"codebase_successor_context"}
SCOPED_FIELDS = FIELDS | {"terminal_task_profile"}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read_public_artifact(parent, name, *, maximum=MAX_INVENTORY_BYTES):
    """The explicit inventory artifact can retain the full signed input closure."""
    if type(maximum) is not int or maximum not in {MAX_BYTES, MAX_INVENTORY_BYTES, MAX_SUCCESSOR_BYTES}:
        raise ValueError("explicit bounded public instruction read profile required")
    descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
            raise ValueError("bounded single-link public instruction required")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(maximum + 1)
        after = os.fstat(descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_nlink")
        if len(raw) > maximum or any(getattr(before, key) != getattr(after, key) for key in fields):
            raise ValueError("public instruction changed during read")
        return raw, before
    finally:
        os.close(descriptor)


def _source(root, name):
    local._path(name)
    path = root / name
    parent = _directory(path.parent)
    try:
        return _read(parent, path.name)
    finally:
        os.close(parent)


def _current_sources(root, sources):
    total = 0
    for name, expected in sources.items():
        if set(expected) != {"sha256", "executable"} or type(expected["executable"]) is not bool:
            raise ValueError("exact signed source binding required")
        raw, info = _source(root, name)
        total += len(raw)
        if len(raw) > 1_000_000 or total > 4_000_000:
            raise ValueError("instruction source inventory exceeds bound")
        if {"sha256": _sha(raw), "executable": bool(info.st_mode & 0o111)} != expected:
            raise ValueError("public instruction source inventory is stale")


def _scoped_declaration(payload, manifest):
    """Replay the public declaration without opening private owner state."""
    if payload["schema"] != SCOPED_SCHEMA:
        return None
    from .terminal_task_profile import (
        INSTRUCTION, PROFILE, SMOKE, SCOPED_SCHEMA as PROFILE_SCHEMA,
        task_profile_bytes, task_profile_smoke, task_profile_spec,
        task_profile_worker_inputs, validate_task_profile,
    )
    declaration = payload["terminal_task_profile"]
    checked = validate_task_profile(declaration)
    local._validate_local_manifest_declarations(manifest)
    if (checked["schema"] != PROFILE_SCHEMA or declaration != checked
            or manifest["schema"] not in {local.MANIFEST_SCHEMA, local.PLANNER_MANIFEST_SCHEMA,
                local.CREATE_MANIFEST_SCHEMA} or len(manifest["tasks"]) != 1
            or payload["source_path"] != INSTRUCTION
            or set(manifest["sources"]) != set(task_profile_worker_inputs(checked))
            or manifest["sources"][PROFILE]["sha256"] != _sha(task_profile_bytes(checked))
            or manifest["sources"][INSTRUCTION]["sha256"] != checked["instruction_sha256"]
            or manifest["sources"][SMOKE]["sha256"] != _sha(task_profile_smoke(checked).encode())):
        raise ValueError("scoped public instruction differs from its complete signed task declaration")
    expected = task_profile_spec(checked, policy_cid=content_identity(local.LOCAL_POLICY))
    actual = manifest["tasks"][0]
    # Evidence is already signed with the complete task specification. Retain
    # its exact declaration while independently replaying all effects/support.
    expected["acceptance"][0]["evidence_cids"] = actual["acceptance"][0]["evidence_cids"]
    if actual != expected or payload["task_id"] != expected["task_key"]:
        raise ValueError("scoped public instruction differs from complete signed task effects")
    return checked


def _signed_terminal_profile(root, manifest, *, current):
    """Detect scoped profiles from signed originals, including on downgrade."""
    from .terminal_task_profile import PROFILE, SCOPED_SCHEMA as PROFILE_SCHEMA, task_profile_bytes, validate_task_profile
    if PROFILE not in manifest["sources"]:
        return None
    raw = (_source(root, PROFILE)[0] if current else
           _git(root, "show", manifest["baseline_commit"] + ":" + PROFILE))
    if len(raw) > 65536 or _sha(raw) != manifest["sources"][PROFILE]["sha256"]:
        raise ValueError("signed public task profile bytes differ")
    declared = json.loads(raw, object_pairs_hook=_unique)
    if type(declared) is not dict:
        raise ValueError("signed public task profile requires a declaration")
    if declared.get("schema") != PROFILE_SCHEMA:
        return None
    checked = validate_task_profile(declared)
    if raw != task_profile_bytes(checked):
        raise ValueError("canonical scoped public task profile required")
    return checked


def _scoped_semantic_inclusion(payload, manifest, *, root, workspace, prompt, current, declaration):
    """Join the actual compiled worker evidence to signed bounded selection."""
    from .terminal_task_profile import (PROFILE, task_profile_bytes, task_profile_index_paths,
        task_profile_semantic_inputs, task_profile_semantic_required_paths)
    from .terminal_semantic_scope import terminal_semantic_scope_observation
    from .semantic_router_translation import _prompt_parts, _verified_semantic
    from ...mcp_server.mcplusplus.kubo_cid import cid_for_bytes

    wire, _, _, _, text, artifact = _prompt_parts(prompt)
    semantic, _ = _verified_semantic(root, artifact=artifact, text=text,
        task_id=payload["task_id"], current=current)
    paths = sorted(task_profile_semantic_inputs(declaration))
    required = sorted(task_profile_semantic_required_paths(declaration))
    program = sorted(task_profile_index_paths(declaration))
    if (wire["objective_id"] != payload["task_id"]
            or semantic["schema"] != "supervisor-semantic-worker-context@2"
            or set(semantic["manifest"]) != set(paths)
            or semantic.get("required_raw_paths") != required
            or semantic.get("program_paths") != program
            or semantic.get("raw_sources", {}).get(PROFILE) != task_profile_bytes(declaration).decode()):
        raise ValueError("compiled semantic worker scope differs from the signed declaration")
    # Reproduce the complete baseline tree, including executable modes, from
    # existing Git objects. Historical replay needs no surviving worker or
    # private owner state; live replay also verifies both working checkouts.
    tree = _git(root, "ls-tree", "-rz", "--full-tree", manifest["baseline_commit"])
    entries = tree.split(b"\0")
    if len(tree) > MAX_BYTES or len(entries) - 1 > local.source_inventory_limit(manifest):
        raise ValueError("scoped source tree exceeds its complete inventory bound")
    modes = {}
    for entry in entries:
        if not entry:
            continue
        header, name = entry.split(b"\t", 1)
        mode, kind, _ = header.split(b" ")
        name = name.decode("utf-8")
        if mode not in {b"100644", b"100755"} or kind != b"blob" or name in modes:
            raise ValueError("scoped inputs must be exact regular Git originals")
        modes[name] = mode == b"100755"
    if set(modes) != set(manifest["sources"]) or any(
            modes[name] != row["executable"] for name, row in manifest["sources"].items()):
        raise ValueError("scoped tree differs from complete signed input modes")
    selected, total = {}, 0
    for name in paths:
        if current:
            original, info = _source(workspace, name)
            executable = bool(info.st_mode & 0o111)
        else:
            original = _git(root, "show", manifest["baseline_commit"] + ":" + name)
            executable = modes[name]
        total += len(original)
        if total > 1_000_000:
            raise ValueError("selected worker original bytes exceed signed scope bound")
        original.decode("utf-8")
        row = manifest["sources"][name]
        source_cid = cid_for_bytes(original)
        if (_sha(original) != row["sha256"] or executable != row["executable"]
                or semantic["manifest"][name] != {"sha256": row["sha256"], "source_cid": source_cid}
                or name in semantic["raw_sources"] and semantic["raw_sources"][name] != original.decode("utf-8")):
            raise ValueError("compiled selected source differs from signed original bytes")
        selected[name] = dict(sha256=row["sha256"], executable=executable,
            source_cid=source_cid, source_bytes=len(original),
            role="program" if name in program else "support", required=name in required)
    observation = terminal_semantic_scope_observation(declaration, payload["manifest"], selected_sources=selected)
    if current:
        # Semantic producer verification runs before this closing full-source
        # replay, so a changed omitted input cannot escape through selection.
        for checkout in (root, workspace):
            if _git(checkout, "rev-parse", "HEAD").decode().strip() != manifest["baseline_commit"]:
                raise ValueError("scoped public instruction baseline changed during worker replay")
            _current_sources(checkout, manifest["sources"])
    return dict(schema=SCOPE_INCLUSION_SCHEMA,
        **{key: observation[key] for key in (
            "manifest_cid", "profile_sha256", "declaration_cid", "frontier_cid",
            "selected_count", "full_count", "frontier_count", "selected_source_bytes",
            "max_files", "max_source_bytes")},
        scope_observation_cid=content_identity(observation), scope_cid=semantic["scope_cid"],
        semantic_root_cid=semantic["semantic_root_cid"], semantic_payload_sha256=_sha(text.encode()),
        required_count=len(required), program_count=len(program),
        native_claim_verified=False, native_allocation_verified=False, process_identity_verified=False,
        proof_authority=False, execution_authority=False, completion_authority=False, omission_authority=False)


def _selection(payload):
    """Replay public signature and selection; the launch digest pins the owner."""
    verify_signature = local._verify_signature
    if payload.get("schema") == SUCCESSOR_SCHEMA:
        from .codebase_successor_dispatch_context import _verify_signature
        verify_signature = _verify_signature
    manifest = verify_signature(payload["manifest"], SimpleNamespace(
        identity_did=payload["owner_identity"], profile_id=payload["owner_profile_id"]))
    if (content_identity(payload["manifest"]) != payload["manifest_cid"]
            or manifest["repository"] != payload["repository"]
            or manifest.get("schema") not in local.SUPPORTED_MANIFEST_SCHEMAS):
        raise ValueError("public instruction signed manifest differs")
    specs = [row for row in manifest["tasks"] if row["task_key"] == payload["task_id"]]
    path = payload["source_path"]
    local._path(path)
    intent = payload.get("schema") == INTENT_SCHEMA
    inventory = payload.get("schema") in {INVENTORY_SCHEMA, SUCCESSOR_SCHEMA}
    successor = payload.get("schema") == SUCCESSOR_SCHEMA
    if manifest["schema"] == local.INTENT_MANIFEST_SCHEMA and not intent:
        raise ValueError("signed IntentIR admission requires the task-specific instruction version")
    if intent and (manifest["schema"] != local.INTENT_MANIFEST_SCHEMA
            or local.decode_intent_requirement_contract(manifest)["source_path"] != path):
        raise ValueError("worker requirement context must select the signed IntentIR source")
    if inventory != (manifest["schema"] in local.INVENTORY_MANIFEST_SCHEMAS):
        raise ValueError("inventory context requires its explicit public instruction profile")
    if successor != (manifest["schema"] == local.SUCCESSOR_MANIFEST_SCHEMA):
        raise ValueError("successor context requires its explicit signed public instruction profile")
    if (len(specs) != 1 or (not intent and path not in specs[0]["scope_paths"])
            or any(path == output["path"] for spec in manifest["tasks"] for output in spec["outputs"])
            or manifest["sources"].get(path, {}).get("sha256") != payload["source_sha256"]
            or type(payload["source_bytes"]) is not int or not 0 < payload["source_bytes"] <= MAX_INSTRUCTION_BYTES):
        raise ValueError("public instruction must be the exact declared read-only task input")
    _scoped_declaration(payload, manifest)
    return manifest


def _intent_context(payload, manifest, source_text):
    """Replay public signed admission without opening profile keys or state."""
    if payload["schema"] != INTENT_SCHEMA:
        return None
    local._validate_local_manifest_declarations(manifest)
    plan = payload["intent_plan_admission"]
    if (not isinstance(plan, dict) or set(plan) != {"graph", "receipt", "requirement_bindings"}
            or manifest["policy"] != local.LOCAL_POLICY):
        raise ValueError("exact public intent admission and local policy required")
    profile = SimpleNamespace(identity_did=payload["owner_identity"],
                              profile_id=payload["owner_profile_id"])
    graph = local.PromptGoalGraph.from_dict(plan["graph"])
    signed = local._verify_signature(plan["receipt"], profile)
    expected = local._planning_payload(graph, payload["manifest"], manifest, profile,
        manifest["sources"], plan["requirement_bindings"], requirement_source_text=source_text,
        source_applicability_nomination=local._header_nomination(signed))
    if signed != expected:
        raise ValueError("worker intent admission differs from exact replayed planning receipt")
    tasks = [task for task in graph.tasks if task.task_cid == payload["task_cid"]]
    if len(tasks) != 1 or tasks[0].task_key != payload["task_id"]:
        raise ValueError("worker requirement context task differs from signed graph identity")
    from .intent_requirement_context import build_task_intent_context

    return build_task_intent_context(contract=local._verify_intent_requirement_text(manifest, source_text),
        graph=graph, coverage=expected["requirement_coverage"], task_cid=payload["task_cid"])


def _inventory_context(payload):
    if payload["schema"] not in {INVENTORY_SCHEMA, SUCCESSOR_SCHEMA}:
        return None
    from .codebase_inventory_evidence_worker_context import build_inventory_worker_context

    plan = payload["inventory_plan_admission"]
    if type(plan) is not dict or set(plan) != {"graph", "receipt"}:
        raise ValueError("exact public inventory full-plan admission required")
    if payload["schema"] == SUCCESSOR_SCHEMA:
        from .codebase_successor_dispatch_context import build_successor_worker_context
        return build_successor_worker_context(manifest_envelope=payload["manifest"],
            graph=plan["graph"], receipt=plan["receipt"], task_cid=payload["task_cid"],
            owner_identity=payload["owner_identity"], owner_profile_id=payload["owner_profile_id"],
            inventory_context=payload["codebase_inventory_context"],
            successor_context=payload["codebase_successor_context"])
    return build_inventory_worker_context(manifest_envelope=payload["manifest"],
        graph=plan["graph"], receipt=plan["receipt"], task_cid=payload["task_cid"],
        owner_identity=payload["owner_identity"], owner_profile_id=payload["owner_profile_id"],
        inventory_context=payload["codebase_inventory_context"])


def prepare_public_instruction_context(*, repository: Path, admission: dict, task_cid: str,
                                       source_path: str, expected_source_sha256: str,
                                       codebase_inventory_context: dict | None = None,
                                       codebase_successor_context: dict | None = None) -> dict:
    """Owner selects a specific signed input; ambient filenames are never used."""
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    manifest, profile = verified["manifest"], verified["profile"]
    root = Path(repository).absolute()
    if root.resolve(strict=True) != root or str(root) != manifest["repository"]:
        raise ValueError("public instruction requires the independently admitted repository")
    tasks = [row for row in verified["graph"].tasks if row.task_cid == task_cid]
    if len(tasks) != 1:
        raise ValueError("public instruction requires the exact admitted task")
    raw, _ = _source(root, source_path)
    if _sha(raw) != expected_source_sha256 or not raw.decode("utf-8").strip() or b"\0" in raw:
        raise ValueError("public instruction differs from the explicitly selected source bytes")
    payload = {"schema": SCHEMA, "repository": str(root), "task_cid": task_cid,
        "task_id": tasks[0].task_key, "manifest": admission["manifest"],
        "manifest_cid": verified["receipt"]["manifest_cid"],
        "owner_identity": profile.identity_did, "owner_profile_id": profile.profile_id,
        "source_path": source_path, "source_sha256": expected_source_sha256, "source_bytes": len(raw),
        "completion_authority": False, "publication_authority": False, "scope_expansion_authority": False}
    terminal_profile = _signed_terminal_profile(root, manifest, current=True)
    if terminal_profile is not None:
        if manifest["schema"] not in {local.MANIFEST_SCHEMA, local.PLANNER_MANIFEST_SCHEMA,
                local.CREATE_MANIFEST_SCHEMA}:
            raise ValueError("selected Terminal worker context requires an ordinary singleton manifest")
        payload.update(schema=SCOPED_SCHEMA, terminal_task_profile=terminal_profile)
    if manifest["schema"] == local.INTENT_MANIFEST_SCHEMA:
        payload.update(schema=INTENT_SCHEMA, intent_plan_admission={
            "graph": admission["graph"], "receipt": admission["receipt"],
            "requirement_bindings": admission["requirement_bindings"]})
    if manifest["schema"] in local.INVENTORY_MANIFEST_SCHEMAS:
        payload.update(schema=INVENTORY_SCHEMA, inventory_plan_admission={
            "graph": admission["graph"], "receipt": admission["receipt"]},
            codebase_inventory_context=codebase_inventory_context)
        if manifest["schema"] == local.SUCCESSOR_MANIFEST_SCHEMA:
            payload.update(schema=SUCCESSOR_SCHEMA, codebase_successor_context=codebase_successor_context)
        elif codebase_successor_context is not None:
            raise ValueError("successor context requires its explicit signed manifest profile")
    elif codebase_inventory_context is not None:
        raise ValueError("full inventory context requires its explicit signed profile")
    elif codebase_successor_context is not None:
        raise ValueError("successor context requires its explicit signed profile")
    selected = _selection(payload)
    requirements = _intent_context(payload, selected, raw.decode("utf-8"))
    inventory_context = _inventory_context(payload)
    payload["context_cid"] = content_identity(payload)
    encoded = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    maximum = (MAX_SUCCESSOR_BYTES if payload["schema"] == SUCCESSOR_SCHEMA else
               MAX_INVENTORY_BYTES if inventory_context is not None else MAX_BYTES)
    if len(encoded) > maximum:
        raise ValueError("public instruction artifact exceeds bound")
    parent = root
    for name in Path(DIRECTORY).parts:
        parent = parent / name
        try:
            parent.mkdir(mode=0o755)
        except FileExistsError:
            pass
        info = parent.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o022 or stat.S_IMODE(info.st_mode) & 0o005 != 0o005):
            raise ValueError("instruction artifact requires owner-controlled worker-readable storage")
    artifact = parent / (_sha(encoded) + ".json")
    try:
        with artifact.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
    except FileExistsError:
        if inventory_context is None:
            raise
        descriptor = _directory(parent)
        try:
            retained, info = _read_public_artifact(descriptor, artifact.name, maximum=maximum)
        finally:
            os.close(descriptor)
        if retained != encoded or stat.S_IMODE(info.st_mode) != 0o444:
            raise ValueError("immutable inventory instruction artifact differs")
    result = {"artifact": str(artifact), "sha256": _sha(encoded), "task_cid": task_cid,
        "context_cid": payload["context_cid"], "manifest_cid": payload["manifest_cid"],
        "source_path": source_path, "source_sha256": expected_source_sha256, "source_bytes": len(raw),
        "completion_authority": False, "scope_expansion_authority": False}
    if requirements is not None:
        result.update(requirements_context_cid=requirements["requirements_context_cid"],
                      requirement_ids=requirements["requirement_ids"])
    if inventory_context is not None:
        result.update(inventory_context_cid=inventory_context["context_cid"],
            codebase_inventory_context_cid=inventory_context["codebase_inventory_context_cid"])
        if payload["schema"] == SUCCESSOR_SCHEMA:
            result.update(codebase_successor_context_cid=inventory_context["codebase_successor_context_cid"],
                successor_selection_cid=inventory_context["codebase_successor"]["selection_cid"],
                source_delta_cid=inventory_context["codebase_successor"]["source_delta_cid"])
    return result


def load_public_instruction(*, artifact: Path, expected_sha256: str, task_cid: str, prompt: str,
                            workspace: Path | None = None, repository: Path | None = None,
                            require_current_source: bool = True) -> tuple[str, dict]:
    """Load exact UTF-8 requirements; historical replay grants no dispatch authority."""
    artifact = Path(artifact).absolute()
    parent = _directory(artifact.parent)
    try:
        raw, info = _read_public_artifact(parent, artifact.name, maximum=MAX_SUCCESSOR_BYTES)
    finally:
        os.close(parent)
    if len(raw) > MAX_SUCCESSOR_BYTES or _sha(raw) != expected_sha256 or stat.S_IMODE(info.st_mode) & 0o222:
        raise ValueError("public instruction artifact digest, size or immutability differs")
    payload = json.loads(raw, object_pairs_hook=_unique)
    fields = (SCOPED_FIELDS if isinstance(payload, dict) and payload.get("schema") == SCOPED_SCHEMA else
        SUCCESSOR_FIELDS if isinstance(payload, dict) and payload.get("schema") == SUCCESSOR_SCHEMA else
        INVENTORY_FIELDS if isinstance(payload, dict) and payload.get("schema") == INVENTORY_SCHEMA
        else INTENT_FIELDS if isinstance(payload, dict) and payload.get("schema") == INTENT_SCHEMA else FIELDS)
    if (not isinstance(payload, dict) or set(payload) != fields or payload["schema"] not in {SCHEMA, INTENT_SCHEMA, INVENTORY_SCHEMA, SUCCESSOR_SCHEMA, SCOPED_SCHEMA}
            or len(raw) > (MAX_SUCCESSOR_BYTES if payload["schema"] == SUCCESSOR_SCHEMA else
                          MAX_INVENTORY_BYTES if payload["schema"] == INVENTORY_SCHEMA else MAX_BYTES)
            or content_identity({k: v for k, v in payload.items() if k != "context_cid"}) != payload["context_cid"]
            or payload["task_cid"] != task_cid
            or any(payload[k] is not False for k in
                   ("completion_authority", "publication_authority", "scope_expansion_authority"))):
        raise ValueError("public instruction identity or authority differs")
    root = Path(payload["repository"])
    if (not root.is_absolute() or root.resolve(strict=True) != root or artifact.parent != root / DIRECTORY
            or (repository is not None and Path(repository).absolute() != root)):
        raise ValueError("public instruction artifact belongs to another canonical repository")
    manifest = _selection(payload)
    if payload["schema"] == SCOPED_SCHEMA and type(require_current_source) is not bool:
        raise ValueError("selected Terminal replay requires an explicit current/historical mode")
    if len(prompt.encode()) > 256_000:
        raise ValueError("native prompt exceeds bound")
    wire, _ = json.JSONDecoder(object_pairs_hook=_unique).raw_decode(prompt.lstrip())
    if not isinstance(wire, dict) or wire.get("objective_id") != payload["task_id"]:
        raise ValueError("public instruction belongs to another native task")
    sources, baseline = manifest["sources"], manifest["baseline_commit"]
    maximum = local.source_inventory_limit(manifest)
    if not isinstance(sources, dict) or not 1 <= len(sources) <= maximum:
        raise ValueError("bounded signed source inventory required")
    if _git(root, "rev-parse", baseline + "^{commit}").decode().strip() != baseline:
        raise ValueError("public instruction baseline is unavailable")
    terminal_profile = _signed_terminal_profile(root, manifest, current=require_current_source)
    if (terminal_profile is not None) != (payload["schema"] == SCOPED_SCHEMA):
        raise ValueError("signed selected Terminal profile requires its explicit public instruction version")
    if terminal_profile is not None and terminal_profile != payload["terminal_task_profile"]:
        raise ValueError("retained selected Terminal declaration differs from signed originals")
    if require_current_source:
        if workspace is None:
            raise ValueError("public instruction dispatch requires an allocated worktree")
        if _git(root, "rev-parse", "HEAD").decode().strip() != baseline:
            raise ValueError("public instruction canonical baseline is stale")
        _current_sources(root, sources)
        workspace = Path(workspace).absolute()
        if (workspace.resolve(strict=True) != workspace or workspace == root
                or _git(workspace, "rev-parse", "--show-toplevel").decode().strip() != str(workspace)
                or _git(workspace, "rev-parse", "--path-format=absolute", "--git-common-dir")
                    != _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
                or _git(workspace, "rev-parse", "HEAD").decode().strip() != baseline):
            raise ValueError("public instruction requires the matching allocated worktree")
        _current_sources(workspace, sources)
        if manifest["schema"] in local.INVENTORY_MANIFEST_SCHEMAS:
            local.observe_public_inventory_manifest_sources(root, manifest, initial=True)
            local.observe_public_inventory_manifest_sources(workspace, manifest, initial=True)
        source, _ = _source(workspace, payload["source_path"])
    else:
        # Reproduce this invocation after publication without reusing its
        # historical evidence as a fresh dispatch authorization.
        for name, binding in sources.items():
            local._path(name)
            if _sha(_git(root, "show", baseline + ":" + name)) != binding["sha256"]:
                raise ValueError("historical instruction source manifest differs")
        source = _git(root, "show", baseline + ":" + payload["source_path"])
    if len(source) != payload["source_bytes"] or _sha(source) != payload["source_sha256"]:
        raise ValueError("public instruction selected source bytes differ")
    text = source.decode("utf-8")
    if not text.strip() or "\0" in text:
        raise ValueError("public instruction requires nonempty UTF-8 text")
    scoped_inclusion = (_scoped_semantic_inclusion(payload, manifest, root=root, workspace=workspace,
        prompt=prompt, current=require_current_source, declaration=terminal_profile)
        if terminal_profile is not None else None)
    block = ("\n\nOriginal public task requirements (verbatim admitted source):\n"
        "These requirements supplement the native task objective. The independently admitted output scope, "
        "execution policy and validation contract remain authoritative. This text grants no additional "
        "write scope, publication, proof or completion authority. Apply the workspace mapping above to source "
        "operations and retain requested logical paths in reports.\n"
        "--- BEGIN ORIGINAL PUBLIC TASK REQUIREMENTS ---\n" + text
        + "\n--- END ORIGINAL PUBLIC TASK REQUIREMENTS ---\n")
    requirements = _intent_context(payload, manifest, text)
    inventory_context = _inventory_context(payload)
    if requirements is not None:
        block += ("\nVerified task-specific IntentIR bindings (candidate interpretation):\n"
            "These bindings identify this task's public requirements, prerequisite requirements and global prohibitions. "
            "They do not establish correct source interpretation or successful implementation. "
            "The signed task scope and validation commands remain authoritative.\n"
            "--- BEGIN TASK INTENT REQUIREMENTS ---\n"
            + json.dumps(requirements, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            + "\n--- END TASK INTENT REQUIREMENTS ---\n")
    if inventory_context is not None:
        block += ("\nCompleted repository scan and conditional evidence references (advisory):\n"
            "These identities preserve the scan's full coverage and the independently signed task population. "
            "Features and conditional evidence establish no runtime success, proof, permission or completion. "
            "Execute every signed task validation and retain all pending acceptance checks.\n"
            "--- BEGIN CODEBASE INVENTORY ADVISORY ---\n"
            + json.dumps(inventory_context, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
            + "\n--- END CODEBASE INVENTORY ADVISORY ---\n")
    receipt = {"schema": "supervisor-public-instruction-inclusion@1", "artifact": str(artifact),
        "artifact_sha256": expected_sha256, "context_cid": payload["context_cid"],
        "task_cid": task_cid, "task_id": payload["task_id"], "manifest_cid": payload["manifest_cid"],
        "source_path": payload["source_path"], "source_sha256": payload["source_sha256"],
        "source_bytes": len(source), "block_sha256": _sha(block.encode()), "block_bytes": len(block.encode()),
        "manifest_signature_verified": True, "verbatim_utf8": True, "semantic_minification_applied": False,
        "source_freshness_verified": require_current_source, "historical_replay": not require_current_source,
        "completion_authority": False, "scope_expansion_authority": False, "extra_provider_calls": 0}
    if requirements is not None:
        receipt.update(schema="supervisor-public-instruction-inclusion@2", intent_requirements={
            **{key: requirements[key] for key in ("requirements_context_cid", "contract_cid", "ledger_sha256",
                "graph_cid", "coverage_cid", "requirement_ids", "source_semantics_verified", "semantic_alignment_verified")},
            "dependency_requirement_ids": [row["requirement_id"] for row in requirements["dependency_requirements"]],
            "prohibition_requirement_ids": [row["requirement_id"] for row in requirements["global_prohibitions"]],
            "planning_receipt_cid": content_identity(payload["intent_plan_admission"]["receipt"]),
            "admission_graph_verified": True, "native_persistence_verified_here": False,
            "proof_authority": False, "execution_authority": False, "completion_authority": False})
    if inventory_context is not None:
        receipt.update(schema="supervisor-public-instruction-inclusion@3", codebase_inventory={
            "context_cid": inventory_context["context_cid"],
            "codebase_inventory_context_cid": inventory_context["codebase_inventory_context_cid"],
            "root_cid": inventory_context["scan"]["root_cid"],
            "completion_cid": inventory_context["scan"]["completion_cid"],
            "membership_cid": inventory_context["scan"]["membership_cid"],
            "planning_receipt_cid": inventory_context["planning_receipt_cid"],
            "administrator_task_cids": inventory_context["administrator_task_cids"],
            "pending_cid": inventory_context["pending_cid"], "current_facts": [],
            "removed_task_cids": [], "runtime_requirements_preserved": True,
            "native_inventory_current_verified_here": False, "native_persistence_verified_here": False,
            "authority": inventory_context["authority"]})
        if payload["schema"] == SUCCESSOR_SCHEMA:
            receipt.update(schema="supervisor-public-instruction-inclusion@4", codebase_successor={
                "context_cid": inventory_context["codebase_successor_context_cid"],
                "selection_cid": inventory_context["codebase_successor"]["selection_cid"],
                "source_delta_cid": inventory_context["codebase_successor"]["source_delta_cid"],
                "previous_head": inventory_context["codebase_successor"]["previous_head"],
                "current_head": inventory_context["codebase_successor"]["current_head"],
                "root_cid": inventory_context["scan"]["root_cid"],
                "completion_cid": inventory_context["scan"]["completion_cid"],
                "native_inventory_current_verified_here": False,
                "native_successor_current_verified_here": False,
                "authority": inventory_context["authority"]})
    if scoped_inclusion is not None:
        receipt.update(schema=SCOPED_INCLUSION_SCHEMA, terminal_semantic_scope=scoped_inclusion)
    return block, receipt
