"""Indexed, advisory planning evidence before any native task is admitted.

The independently signed declaration supplies an alias, not a canonical task.
An empty owner is captured through the native world builder and re-captured
before planning. Task-bound world persistence is used only after admission.
Source artifacts are reused verbatim; changed input fails instead of rebuilding
silently or switching embedding policies.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import DirectoryScanReceipt, PromptWorkflowRequest
from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import (
    load_code_retrieval_context, prepare_code_retrieval_context,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    load_semantic_worker_context, prepare_semantic_context,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
from ipfs_accelerate_py.agent_supervisor.semantic_state.intent_world_snapshot import (
    capture_intent_world_snapshot, load_intent_world_context, persist_intent_world_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


SCHEMA = "terminal-initial-indexed-planning@1"
MAX_BYTES = 1_000_000


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_bytes(value))
    return {"artifact": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _read(root, reference):
    name = reference["artifact"]
    path = root / name
    if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
            or Path(name).as_posix() != name or path.resolve() != path
            or not path.is_relative_to(root) or path.is_symlink()):
        raise ValueError("initial planning artifact path is not canonical")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("initial planning artifact digest or bound differs")
    return json.loads(raw)


def _reference(root, path):
    return {"artifact": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _semantic_view(root, metadata, alias):
    payload = json.loads(load_semantic_worker_context(repository=root,
        artifact=metadata["Semantic context artifact"],
        expected_sha256=metadata["Semantic context sha256"], task_id=alias))
    blocks = (root / metadata["Semantic context artifact"]).parent / "blocks"

    def get_block(cid):
        if not isinstance(cid, str) or not cid or Path(cid).name != cid or cid in {".", ".."}:
            raise ValueError("invalid initial semantic block identity")
        path = blocks / cid
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("noncanonical initial semantic block path")
        return path.read_bytes()

    view = IpfsDatasetsSemanticStateProvider().open_verified_view(payload["semantic_root_cid"], get_block)
    for capsule in payload["capsules"]:
        if view.capsule(capsule["stable_symbol_id"]).to_dict() != capsule:
            raise ValueError("initial capsule differs from verified native semantic root")
    return payload, view, get_block


def _summaries(descriptor, semantic, retrieval, world, *, intent_freshness_checked):
    """Bounded projections retain native identities and explicit omission counts."""
    capsules = []
    for capsule in semantic["capsules"][:2]:
        item = {key: capsule[key] for key in (
            "capsule_cid", "stable_symbol_id", "symbol_fact_cid", "version_cid",
            "source_cid", "source_slice_path", "confidence",
        ) if key in capsule}
        signature = _bytes(capsule.get("signature", {})).decode()
        item["signature_json"] = signature if len(signature.encode()) <= 2048 else None
        item["signature_omitted"] = len(signature.encode()) > 2048
        item["effects"] = capsule.get("effects", [])[:8]
        item["effects_omitted"] = max(0, len(capsule.get("effects", [])) - 8)
        capsules.append(item)
    hits = [{key: hit[key] for key in (
        "path", "symbol", "line_start", "line_end", "rank", "score", "row_id", "ast_record_id",
    )} for hit in retrieval["hits"][:5]]
    shared = {"execution_authority": False, "completion_authority": False,
              "semantic_equivalence_claimed": False}
    summaries = [
        {"schema": "terminal-initial-world-planning-summary@1",
         "planning_request_cid": descriptor["request_cid"],
         "capture_cid": descriptor["world_capture_cid"],
         "artifact_kind": "empty-intent-planning-capture",
         **{key: world[key] for key in ("world_snapshot_cid", "repository_id",
             "plan_projection_cid", "event_watermark", "semantic_root_cid",
             "unavailable_components", "schedulable", "tasks", "goals", "objectives")},
         "intent_freshness_checked": intent_freshness_checked, **shared},
        {"schema": "terminal-initial-semantic-planning-summary@1",
         "semantic_root_cid": semantic["semantic_root_cid"], "scope_cid": semantic["scope_cid"],
         "source_bindings": semantic["manifest"], "capsules": capsules,
         "full_capsules": descriptor["semantic"]["capsules"],
         "omitted_capsules": descriptor["semantic"]["capsules"] - len(capsules),
         "doctor_snapshot_id": semantic["doctor_snapshot_id"],
         "doctor_manifest_cid": semantic["doctor_manifest_cid"],
         "doctor_findings": descriptor["semantic"]["doctor_findings"],
         "repair_admission_evaluated": False,
         "task_alias": descriptor["task_alias"], "task_alias_is_declaration_only": True,
         "raw_source_required_before_edit": True, **shared},
        {"schema": "terminal-initial-retrieval-planning-summary@1",
         **{key: retrieval[key] for key in ("index_id", "query_id", "result_id", "source_sha256")},
         "public_query_sha256": descriptor["public_query_sha256"],
         "hits": hits, "omitted_hits": len(retrieval["hits"]) - len(hits),
         "learned_embeddings": descriptor["learned_embeddings"], "nomination_only": True, **shared},
    ]
    learner = descriptor.get("codebase_autoencoder")
    if learner is not None:
        nominations = learner.get("security_candidate_nominations")
        selected_rows = {row["row_id"] for row in learner["ranks"][:5]}
        advisory = ({**{key: value for key, value in nominations.items() if key != "rows"},
            "omitted_rows": sum(row["row_id"] not in selected_rows for row in nominations["rows"])}
            if nominations else None)
        summaries.append({"schema": "terminal-code-autoencoder-planning-summary@1",
            "checkpoint_sha256": learner["checkpoint_sha256"], "receipt_sha256": learner["receipt_sha256"],
            "sample_count": learner["sample_count"], "epochs_completed": learner["epochs_completed"],
            "source_hashes": learner["source_hashes"], "nomination_only": True,
            "formal_translation_authority": False,
            **({"legal_ir_weights_forked": True, "legal_ir_mutable_state_shared": False}
                if learner.get("weight_transfer") else {"legal_ir_state_reused": False}),
            "ranked_candidates": learner["ranks"][:5],
            "catalog_receipt_sha256": descriptor["codebase_autoencoder_catalog"]["receipt_sha256"],
            "world_record_cid": descriptor["codebase_autoencoder_catalog"]["hydration"]["world_record_cid"],
            "weight_transfer": ({key: learner["weight_transfer"][key] for key in (
                "source_checkpoint_sha256", "initializer_sha256", "transferred_row_count")}
                if learner.get("weight_transfer") else None),
            "canonical_cve_manifest_sha256": (learner["canonical_cve_training"]["manifest_sha256"]
                if learner.get("canonical_cve_training") else None),
            "security_candidate_projection": advisory,
            "security_candidate_nominations": ([row for row in nominations["rows"] if row["row_id"] in selected_rows]
                if nominations else None),
            **shared})
    frozen = descriptor.get("security_autoencoder_advice")
    if frozen is not None:
        summaries.append({**frozen["summary"],
            "catalog_receipt_sha256": frozen["receipt_sha256"],
            "world_record_cid": frozen["hydration"]["world_record_cid"]})
    source384 = descriptor.get("source384_context")
    if source384 is not None:
        summaries.append(source384["summary"])
    if any(len(_bytes(summary)) > 8192 for summary in summaries):
        raise ValueError("initial planning summary exceeds its fixed 8192-byte bound")
    return summaries


def prepare_initial_context(*, state: Path, prepared: dict, model_snapshot: Path | None,
                            model_revision: str, required_raw_paths: list[str],
                            train_autoencoder: bool = False,
                            weight_transfer: dict | None = None,
                            canonical_cve_training: dict | None = None,
                            security_checkpoint: dict | None = None,
                            security_checkpoint_hub: dict | None = None,
                            formula_decoder: dict | None = None,
                            header_protocol: dict | None = None,
                            source384_config: Path | None = None) -> dict:
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
        CodeVectorIndexSnapshot, CodeVectorSearchResult,
    )

    started = time.monotonic()
    root = Path(prepared["repository"]).resolve(strict=True)
    if source384_config is not None and (train_autoencoder or any(value is not None for value in (
            weight_transfer, canonical_cve_training, security_checkpoint, security_checkpoint_hub,
            formula_decoder, header_protocol))):
        raise ValueError("Source384 pinned-parent and legacy security profiles are mutually exclusive")
    if security_checkpoint_hub is not None and security_checkpoint is None:
        raise ValueError("security Hub provenance requires a frozen checkpoint")
    if formula_decoder is not None and security_checkpoint is None:
        raise ValueError("formula decoder requires the frozen security context profile")
    if header_protocol is not None and formula_decoder is None:
        raise ValueError("reviewed header protocol requires a selected formula decoder")
    if security_checkpoint is not None and (train_autoencoder or weight_transfer is not None or canonical_cve_training is not None):
        raise ValueError("frozen security checkpoint and local training are mutually exclusive")
    if (weight_transfer is not None or canonical_cve_training is not None) and not train_autoencoder:
        raise ValueError("security training assets require the explicit autoencoder profile")
    learning_assets = None
    if weight_transfer is not None or canonical_cve_training is not None:
        # Freeze the controller-selected assets before any training. Subsequent
        # planner/admission replay must retain this exact selection.
        learning_assets = json.loads(_bytes({"weight_transfer": weight_transfer,
            "canonical_cve_training": canonical_cve_training}))
    if bool(model_snapshot) != bool(model_revision):
        raise ValueError("learned index requires both the pinned snapshot and revision")
    output = root / ".runtime/terminal-initial-context"
    if output.exists() or (state / "admission.json").exists() or (state / "planner-invoked.json").exists():
        raise ValueError("initial context must precede planning and admission")
    if learning_assets is not None:
        _write(state / "code-learning-assets.json", learning_assets)
    if security_checkpoint is not None:
        selected = {"checkpoint": security_checkpoint, "hub": security_checkpoint_hub}
        if formula_decoder is not None:
            selected.update(formula_decoder=formula_decoder, header_protocol=header_protocol)
        _write(state / "security-checkpoint-selection.json", selected)
    if source384_config is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_advisor import _read as read_selected
        _write(state / "source384-selection.json", {
            "config_path": str(source384_config),
            "config_sha256": hashlib.sha256(read_selected(source384_config, 32768)).hexdigest()})
    alias = prepared["spec"]["task_key"]
    timings, stage = {}, time.monotonic()
    vectors = root / ".runtime/terminal-vectors"
    if model_snapshot:
        from .learned_vector_preflight import qualify
        indexed = qualify(root, vectors, ["bottle.py"], prepared["query"], model_snapshot, model_revision)
    else:
        from .vector_index_preflight import qualify
        indexed = qualify(root, vectors, ["bottle.py"], prepared["query"])
    timings["vector_qualification"] = time.monotonic() - stage
    stage = time.monotonic()
    learner = learner_catalog = frozen_advice = source384 = None
    if source384_config is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.source384_repository_context import prepare_source384_context
        from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
        manifest, _, _ = local._manifest(prepared["manifest"], initial=True)
        source384 = prepare_source384_context(repository=root,
            source_hashes={name: source["sha256"] for name, source in manifest["sources"].items()},
            output=state / "source384-context", config_path=source384_config)
        timings["source384_capture_and_inference"] = time.monotonic() - stage
        stage = time.monotonic()
    if security_checkpoint is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_advisor import prepare_security_advice
        from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
        manifest, _, _ = local._manifest(prepared["manifest"], initial=True)
        paths = [name for name in prepared["worker_inputs"] if name.endswith(".py")]
        frozen_advice = prepare_security_advice(repository=root, paths=paths,
            source_hashes={name: manifest["sources"][name]["sha256"] for name in paths},
            checkpoint=security_checkpoint, output=state / "security-advice", hub_descriptor=security_checkpoint_hub,
            formula_decoder=formula_decoder, header_protocol=header_protocol)
        timings["frozen_security_inference_and_hydration"] = time.monotonic() - stage
        stage = time.monotonic()
    if train_autoencoder:
        from ipfs_datasets_py.logic.formalization.autoencoder.security.codebase_autoencoder import train_codebase_autoencoder
        from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
        manifest, _, _ = local._manifest(prepared["manifest"], initial=True)
        paths = [name for name in prepared["worker_inputs"] if name.endswith(".py")]
        learner = train_codebase_autoencoder(repository=root, paths=paths,
            source_hashes={name: manifest["sources"][name]["sha256"] for name in paths},
            output=state / "code-autoencoder", **({"weight_transfer": weight_transfer}
                if weight_transfer is not None else {}), **({"canonical_cve_training": canonical_cve_training}
                if canonical_cve_training is not None else {}))
        timings["codebase_autoencoder_training"] = time.monotonic() - stage
        stage = time.monotonic()
        from ipfs_accelerate_py.agent_supervisor.runtime.codebase_autoencoder_index import build_codebase_autoencoder_index
        learner_catalog = build_codebase_autoencoder_index(repository=root, learner=learner,
            output=state / "code-autoencoder-catalog")
        timings["codebase_autoencoder_catalog"] = time.monotonic() - stage
        stage = time.monotonic()
    with duckdb.connect(str(vectors / "vectors.duckdb"), read_only=True, config={"threads": 1}) as connection:
        row = connection.execute("SELECT payload FROM snapshots WHERE id=?", [indexed["index_id"]]).fetchone()
        snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(row[0]))
    timings["persisted_snapshot_reopen"] = time.monotonic() - stage
    stage = time.monotonic()
    semantic = prepare_semantic_context(repository=root, paths=prepared["worker_inputs"],
        # Native semantic objectives are trimmed, printable descriptive text.
        # Normalize whitespace only here; the public query remains byte-exact
        # in retrieval, required raw input, planner constraints and hashes.
        required_raw_paths=required_raw_paths, objective=" ".join(prepared["query"].split()), task_id=alias,
        output=output / "semantic", max_symbols=1024, worker_query=prepared["query"], worker_max_bytes=32768)
    metadata = {"Semantic context artifact": (output / "semantic/worker-context.json").relative_to(root).as_posix(),
                "Semantic context sha256": semantic["worker_payload_sha256"], "Semantic context refresh": "true"}
    _, view, get_block = _semantic_view(root, metadata, alias)
    timings["semantic_build_reconstruction_and_hydration"] = time.monotonic() - stage
    stage = time.monotonic()
    retrieval = prepare_code_retrieval_context(repository=root, task_id=alias,
        query_text=prepared["query"], snapshot=snapshot,
        result=CodeVectorSearchResult.from_dict(indexed["hits"]), output=output / "code-retrieval.json")
    metadata.update(retrieval["metadata"])
    timings["retrieval_persistence"] = time.monotonic() - stage
    stage = time.monotonic()
    with IntentRepository(state / "intent.duckdb") as intent:
        capture = capture_intent_world_snapshot(intent, repository_id=view.root.repository_id,
            semantic_root_cid=semantic["semantic_root_cid"], get_semantic_block=get_block)
        if capture["plan_projection"]["plans"] or any(capture["planning_context"][key] for key in (
                "tasks", "goals", "objectives", "goal_edges", "active_plan_heads")):
            raise ValueError("initial planning requires a genuinely empty native intent owner")
    _write(output / "initial-world.json", capture)
    timings["empty_native_world_capture"] = time.monotonic() - stage
    descriptor = {"schema": SCHEMA, "request_cid": PromptWorkflowRequest.from_dict(prepared["request"]).request_cid,
        "scan_cid": DirectoryScanReceipt.from_dict(prepared["scan"]).scan_cid, "task_alias": alias,
        "task_alias_is_declaration_only": True, "manifest_cid": content_identity(prepared["manifest"]),
        "public_query_sha256": hashlib.sha256(prepared["query"].encode()).hexdigest(),
        "repository_id": view.root.repository_id, "metadata": metadata, "semantic": semantic,
        "retrieval": retrieval, "world": _reference(root, output / "initial-world.json"),
        "world_capture_cid": capture["capture_cid"], "world_artifact_kind": "empty-intent-planning-capture",
        "index": _reference(root, vectors / "result.json"),
        "learned_embeddings": bool(model_snapshot), "model_snapshot": str(model_snapshot.resolve(strict=True)) if model_snapshot else None,
        "model_revision": model_revision, "execution_authority": False, "completion_authority": False}
    if learner is not None:
        descriptor["codebase_autoencoder"] = learner
        descriptor["codebase_autoencoder_catalog"] = learner_catalog
        if learning_assets is not None:
            descriptor["code_learning_assets_sha256"] = hashlib.sha256(_bytes(learning_assets)).hexdigest()
    if frozen_advice is not None:
        descriptor["security_autoencoder_advice"] = frozen_advice
    if source384 is not None:
        descriptor["source384_context"] = source384
    if model_snapshot:
        descriptor["model_manifest"] = _reference(root, vectors / "model-manifest.json")
    _write(output / "descriptor.json", descriptor)
    result = {"schema": SCHEMA, "descriptor": _reference(root, output / "descriptor.json"),
        "semantic_root_cid": semantic["semantic_root_cid"], "world_snapshot_cid": capture["snapshot"]["snapshot_cid"],
        "index_id": indexed["index_id"], "indexed_symbols": indexed["symbols"],
        "full_capsules": semantic["capsules"], "learned_embeddings": bool(model_snapshot),
        "world_task_count": 0, "canonical_tasks_created": False, "provider_calls": 0,
        "nonoverlapping_seconds": timings, "seconds": time.monotonic() - started,
        "final_result_persistence_included_in_seconds": False,
        "execution_authority": False, "completion_authority": False}
    if learner is not None:
        result["codebase_autoencoder"] = learner
        result["codebase_autoencoder_catalog"] = learner_catalog
    if frozen_advice is not None:
        result["security_autoencoder_advice"] = frozen_advice
    if source384 is not None:
        result["source384_context"] = source384
    # Strict source and live-owner replay before this result may reach a router.
    stage = time.monotonic()
    _load_initial_context(state=state, prepared=prepared, require_empty_owner=True, result=result)
    timings["final_evidence_verification"] = time.monotonic() - stage
    result["seconds"] = time.monotonic() - started
    _write(state / "initial-context-result.json", result)
    return result


def load_initial_context(*, state: Path, prepared: dict, require_empty_owner: bool) -> dict:
    marker = state / "initial-context-result.json"
    if marker.is_symlink() or marker.stat().st_size > MAX_BYTES:
        raise ValueError("initial context receipt is outside bounds")
    result = json.loads(marker.read_bytes())
    return _load_initial_context(state=state, prepared=prepared, require_empty_owner=require_empty_owner, result=result)


def _load_initial_context(*, state: Path, prepared: dict, require_empty_owner: bool, result: dict) -> dict:
    root = Path(prepared["repository"]).resolve(strict=True)
    descriptor = _read(root, result["descriptor"])
    if (descriptor.get("schema") != SCHEMA or descriptor["request_cid"] != PromptWorkflowRequest.from_dict(prepared["request"]).request_cid
            or descriptor["scan_cid"] != DirectoryScanReceipt.from_dict(prepared["scan"]).scan_cid
            or descriptor["manifest_cid"] != content_identity(prepared["manifest"])
            or descriptor["task_alias"] != prepared["spec"]["task_key"]
            or descriptor["public_query_sha256"] != hashlib.sha256(prepared["query"].encode()).hexdigest()
            or descriptor.get("execution_authority") is not False or descriptor.get("completion_authority") is not False):
        raise ValueError("initial indexed context declaration binding differs")
    alias, metadata = descriptor["task_alias"], descriptor["metadata"]
    semantic, view, get_block = _semantic_view(root, metadata, alias)
    retrieval = json.loads(load_code_retrieval_context(repository=root,
        artifact=metadata["Code retrieval artifact"], expected_sha256=metadata["Code retrieval sha256"], task_id=alias))
    indexed = _read(root, descriptor["index"])
    learner = descriptor.get("codebase_autoencoder")
    frozen = descriptor.get("security_autoencoder_advice")
    source384 = descriptor.get("source384_context")
    source384_selection = state / "source384-selection.json"
    if source384 is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.source384_repository_context import validate_source384_context
        from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_advisor import _read as read_selected
        from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
        manifest, _, _ = local._manifest(prepared["manifest"], initial=True)
        selected = json.loads(read_selected(source384_selection, 131072))
        if (learner is not None or frozen is not None
                or source384["output"] != str(state / "source384-context")
                or result.get("source384_context") != source384
                or source384["source_hashes"] != {name: value["sha256"] for name, value in manifest["sources"].items()}
                or selected != {"config_path": source384["config_path"],
                    "config_sha256": source384["config_sha256"]}):
            raise ValueError("Source384 context differs from selected model/source declaration")
        validate_source384_context(repository=root, expected_receipt=source384)
    elif source384_selection.exists() or result.get("source384_context") is not None:
        raise ValueError("selected Source384 context is missing")
    selection = state / "security-checkpoint-selection.json"
    if frozen is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_advisor import validate_security_advice, _read as read_security
        from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
        manifest, _, _ = local._manifest(prepared["manifest"], initial=True)
        sources = {name: manifest["sources"][name]["sha256"] for name in prepared["worker_inputs"] if name.endswith(".py")}
        selected = {"checkpoint": frozen["checkpoint"], "hub": frozen["model_registration"]["hub"]}
        if frozen.get("formula_decoder") is not None:
            selected.update(formula_decoder=frozen["formula_decoder"], header_protocol=frozen["header_protocol"])
        if (learner is not None or frozen["output"] != str(state / "security-advice")
                or result.get("security_autoencoder_advice") != frozen or frozen["source_hashes"] != sources
                or json.loads(read_security(selection, 32_000)) != selected):
            raise ValueError("frozen security advice differs from selected model/source declaration")
        validate_security_advice(repository=root, expected_receipt=frozen)
    elif selection.exists() or result.get("security_autoencoder_advice") is not None:
        raise ValueError("selected frozen security checkpoint advice is missing")
    if learner is not None:
        from ipfs_datasets_py.logic.formalization.autoencoder.security.codebase_autoencoder import validate_codebase_autoencoder
        from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
        manifest, _, _ = local._manifest(prepared["manifest"], initial=True)
        expected_sources = {name: manifest["sources"][name]["sha256"]
                            for name in prepared["worker_inputs"] if name.endswith(".py")}
        if (learner["output"] != str(state / "code-autoencoder")
                or learner["source_hashes"] != expected_sources
                or result.get("codebase_autoencoder") != learner):
            raise ValueError("code autoencoder differs from independent scope or isolated namespace")
        validate_codebase_autoencoder(repository=root, expected_receipt=learner)
        selected_assets = state / "code-learning-assets.json"
        if selected_assets.exists():
            if selected_assets.is_symlink() or selected_assets.stat().st_size > MAX_BYTES:
                raise ValueError("code learning asset selection is outside bounds")
            raw_assets = selected_assets.read_bytes()
            assets = json.loads(raw_assets)
            if (hashlib.sha256(raw_assets).hexdigest() != descriptor.get("code_learning_assets_sha256")
                    or set(assets) != {"weight_transfer", "canonical_cve_training"}
                    or any(learner.get(key) != assets[key] for key in assets)):
                raise ValueError("code learner differs from controller-selected training assets")
        elif (descriptor.get("code_learning_assets_sha256") is not None
                or learner.get("weight_transfer") is not None
                or learner.get("canonical_cve_training") is not None):
            raise ValueError("code learner training asset selection is missing")
        from ipfs_accelerate_py.agent_supervisor.runtime.codebase_autoencoder_index import validate_codebase_autoencoder_index
        catalog = descriptor.get("codebase_autoencoder_catalog")
        if (not isinstance(catalog, dict) or catalog.get("output") != str(state / "code-autoencoder-catalog")
                or result.get("codebase_autoencoder_catalog") != catalog):
            raise ValueError("code autoencoder catalog differs from isolated index namespace")
        validate_codebase_autoencoder_index(repository=root, learner=learner, expected_receipt=catalog)
    elif descriptor.get("codebase_autoencoder_catalog") is not None:
        raise ValueError("code autoencoder catalog requires bound learner")
    if descriptor["learned_embeddings"]:
        _read(root, descriptor["model_manifest"])
    if (set(semantic["manifest"]) != set(prepared["worker_inputs"])
            or descriptor["semantic"]["semantic_root_cid"] != semantic["semantic_root_cid"]
            or descriptor["repository_id"] != view.root.repository_id
            or retrieval["status"] != "current" or retrieval["query_text"] != prepared["query"]
            or retrieval["index_id"] != indexed["index_id"] or indexed["source_sha256"] != retrieval["source_sha256"]):
        raise ValueError("initial indexes differ from current permitted source/query")
    capture = _read(root, descriptor["world"])
    claimed = capture.get("capture_cid")
    if (claimed != descriptor["world_capture_cid"]
            or claimed != content_identity({key: value for key, value in capture.items() if key != "capture_cid"})
            or capture["planning_context"]["semantic_root_cid"] != semantic["semantic_root_cid"]
            or capture["plan_projection"]["plans"]
            or any(capture["planning_context"][key] for key in ("tasks", "goals", "objectives", "goal_edges", "active_plan_heads"))):
        raise ValueError("initial world is not the bound empty native capture")
    if require_empty_owner:
        with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
            current = capture_intent_world_snapshot(intent, repository_id=view.root.repository_id,
                semantic_root_cid=semantic["semantic_root_cid"], get_semantic_block=get_block)
        if current != capture:
            raise ValueError("initial world differs from current native owner")
    return {"descriptor": descriptor, "receipt": result, "semantic": semantic,
            "retrieval": retrieval, "indexed": indexed, "capture": capture,
            "summaries": _summaries(descriptor, semantic, retrieval, capture["planning_context"],
                intent_freshness_checked=require_empty_owner)}


def bind_admitted_context(*, state: Path, prepared: dict, admission: dict, verified: dict,
                         output: Path, model_snapshot: Path | None, model_revision: str) -> dict:
    loaded = load_initial_context(state=state, prepared=prepared, require_empty_owner=False)
    descriptor = loaded["descriptor"]
    actual_model = str(model_snapshot.resolve(strict=True)) if model_snapshot else None
    if descriptor["model_snapshot"] != actual_model or descriptor["model_revision"] != model_revision:
        raise ValueError("admitted context may not switch the initial embedding policy")
    if len(verified["graph"].tasks) != 1 or verified["graph"].tasks[0].task_key != descriptor["task_alias"]:
        raise ValueError("admitted task differs from independently declared initial alias")
    root, task = Path(prepared["repository"]), verified["graph"].tasks[0]
    semantic, view, get_block = _semantic_view(root, descriptor["metadata"], task.task_key)
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        capture = capture_intent_world_snapshot(intent, repository_id=view.root.repository_id,
            task_cids=[task.task_cid], semantic_root_cid=semantic["semantic_root_cid"], get_semantic_block=get_block)
        records = capture["plan_projection"]["tasks"]
        if len(records) != 1 or records[0]["status"] != "ready":
            raise ValueError("post-planning context requires the exact ready native task")
        from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
        contract, _, _, _ = local._contract(records[0]["body"], task.task_cid)
        if contract["manifest"] != admission["manifest"] or contract["graph_cid"] != verified["graph"].content_id:
            raise ValueError("native task differs from admitted indexed plan")
        world = persist_intent_world_snapshot(capture, output=output / "world", task_id=task.task_key)
        load_intent_world_context(artifact=Path(world["artifact"]), expected_sha256=world["artifact_sha256"],
            task_id=task.task_key, repository_id=view.root.repository_id, intent=intent)
    # Source artifacts must still verify after the independently owned capture.
    load_initial_context(state=state, prepared=prepared, require_empty_owner=False)
    metadata = {**descriptor["metadata"],
        "World context artifact": Path(world["artifact"]).relative_to(root).as_posix(),
        "World context sha256": world["artifact_sha256"], "World context repository": view.root.repository_id}
    result = {"schema": "supervisor-task-context-preparation@1", "task_cid": task.task_cid,
        "task_id": task.task_key, "task_title": task.objective, "task_revision": records[0]["revision"],
        "plan_projection_cid": capture["plan_projection"]["projection_cid"],
        "event_watermark": capture["planning_context"]["event_watermark"], "repository_id": view.root.repository_id,
        "semantic_root_cid": semantic["semantic_root_cid"], "world_snapshot_cid": capture["snapshot"]["snapshot_cid"],
        "metadata": metadata, "semantic": descriptor["semantic"], "retrieval": descriptor["retrieval"], "world": world,
        "initial_context_descriptor": loaded["receipt"]["descriptor"], "new_embedding_calls": 0,
        "preparation_mode": "initial-index-reuse-with-new-admitted-world",
        "execution_authority": False, "completion_authority": False, "canonical_task_mutated": False}
    if descriptor.get("codebase_autoencoder") is not None:
        result["codebase_autoencoder"] = descriptor["codebase_autoencoder"]
        result["codebase_autoencoder_catalog"] = descriptor["codebase_autoencoder_catalog"]
        result["new_autoencoder_training_steps"] = 0
    if descriptor.get("security_autoencoder_advice") is not None:
        result["security_autoencoder_advice"] = descriptor["security_autoencoder_advice"]
        result["new_autoencoder_training_steps"] = 0
    if descriptor.get("source384_context") is not None:
        result["source384_context"] = descriptor["source384_context"]
        result["new_autoencoder_training_steps"] = 0
    _write(output / "result.json", result)
    return {"prepared_context": result, "indexed": loaded["indexed"],
            "diagnostic_artifact": (root / metadata["Semantic context artifact"]).parent / "doctor.json"}
