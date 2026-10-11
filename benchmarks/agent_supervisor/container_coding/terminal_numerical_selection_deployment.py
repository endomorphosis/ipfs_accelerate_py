"""Offline deployment of explicitly selected experimental numerical advice."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

from ipfs_accelerate_py.agent_supervisor.runtime.numerical_selection_config import (
    validate_numerical_selection_config, validate_numerical_selection_assets,
)

ROOT = "/opt/ipfs-supervisor"
PREFIX = "models/numerical-selection-384/"
CHECKPOINT_PATH = PREFIX + "checkpoint.json"
CONFIG_PATH = PREFIX + "config.json"
EMBEDDING_PATH = PREFIX + "models--thenlper--gte-small/snapshots/"
KEY = "numerical_selection_384"
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/terminal_numerical_selection_deployment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_numerical_refresh_accounting.py",
    "benchmarks/agent_supervisor/container_coding/terminal_deployment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_context_audit.py",
    "benchmarks/agent_supervisor/container_coding/terminal_context_rebind.py",
    "benchmarks/agent_supervisor/container_coding/benchmark_comparison.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/numerical_selection_config.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/numerical_selection_context.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/published_numerical_selection.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/source384_config.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/source384_program_scope.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/terminal_source_partition.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/terminal_task_profile.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/security_autoencoder_advisor.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/supervisor_meta_index.py",
    "ipfs_accelerate_py/agent_supervisor/semantic_state/program_world_database.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/task_context_bundle.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/source_formalization_context.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
)
DATASET_PATHS = (
    "ipfs_datasets_py/logic/software_contracts/codebase_numerical_selection_384.py",
    "ipfs_datasets_py/logic/software_contracts/codebase_numerical_selection_384_worker.py",
    "ipfs_datasets_py/logic/software_contracts/codebase_source_384.py",
    "ipfs_datasets_py/logic/software_contracts/codebase_source_384_worker.py",
    "ipfs_datasets_py/logic/software_contracts/codebase_ir.py",
    "ipfs_datasets_py/logic/software_contracts/codebase_resources.py",
    "ipfs_datasets_py/logic/software_contracts/duckdb_ast_store.py",
    "ipfs_datasets_py/logic/software_contracts/duckdb_ingest.py",
    "ipfs_datasets_py/logic/software_contracts/content.py",
    "ipfs_datasets_py/logic/software_contracts/python_frontend.py",
    "ipfs_datasets_py/logic/software_contracts/semantic_index/scanner.py",
    "ipfs_datasets_py/logic/software_contracts/semantic_index/snapshot.py",
    "ipfs_datasets_py/logic/software_contracts/semantic_index/python_analysis.py",
    "ipfs_datasets_py/logic/backends/codebase_process.py",
    "ipfs_datasets_py/duckdb_control/autoencoder_registry.py",
    "ipfs_datasets_py/duckdb_control/codebase_catalog.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/numerical_selection_384.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/structured_source_384.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/source_training_v2.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/grouped_source_training_384.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/complete_training.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/ui_source_contract_384.py",
    "ipfs_datasets_py/logic/ui_ux_ir/model/components.py",
    "ipfs_datasets_py/logic/ui_ux_ir/formalize/flogic.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/source_embeddings_384.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/source_program_runtime_384_v2.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/security/source_program_binding_384_v2.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/distributed_384/numerics.py",
    "ipfs_datasets_py/logic/formalization/autoencoder/distributed_384/contracts.py",
    "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_embedding_runtime.py",
    "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_embedding_production.py",
    "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_corpus_manifest.py",
    "ipfs_datasets_py/optimizers/logic_theorem_optimizer/domain_384_autoencoder.py",
    "ipfs_datasets_py/optimizers/logic_theorem_optimizer/modal_latent_formula.py",
    "ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_formula_codec.py",
    "ipfs_datasets_py/optimizers/logic_theorem_optimizer/legal_ir_grammar_decoder.py",
    "ipfs_datasets_py/optimizers/logic_theorem_optimizer/resource_scheduler.py",
    "ipfs_datasets_py/logic/legal_ir/canonical_contracts.py",
    "ipfs_datasets_py/logic/autoformal/tree_pin.py",
    "ipfs_datasets_py/logic/intent_ir/decoder.py",
    "ipfs_datasets_py/logic/intent_ir/formalize/rich_grammar.py",
    "ipfs_datasets_py/logic/intent_ir/schema.py",
    "ipfs_datasets_py/logic/security_ir/model.py",
    "ipfs_datasets_py/logic/ui_ux_ir/decoder.py",
    "ipfs_datasets_py/logic/ui_ux_ir/schema.py",
)


def _raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def numerical_selection_assets(config):
    """Copy only validated local bytes into a separate, closed model directory."""
    if config is None:
        return [], None
    from .terminal_deployment import _portable_asset_bytes
    from ipfs_datasets_py.logic.formalization.autoencoder.numerical_selection_384 import MAX_BYTES
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding

    config = validate_numerical_selection_assets(config)
    checkpoint = Path(config["checkpoint_path"])
    payload = _portable_asset_bytes(checkpoint, expected_sha256=config["checkpoint_sha256"], maximum=MAX_BYTES)
    snapshot, rows = embedding._snapshot_assets(config["embedding_snapshot"])
    embedding_path = EMBEDDING_PATH + embedding.PINNED_REVISION
    assets = [(CHECKPOINT_PATH, payload)]
    for row in rows:
        assets.append((embedding_path + "/" + row["name"], _portable_asset_bytes(
            (snapshot / row["name"]).resolve(strict=True), expected_sha256=row["sha256"], maximum=row["bytes"])))
    if (embedding._snapshot_assets(snapshot)[1] != rows
            or _portable_asset_bytes(checkpoint, expected_sha256=config["checkpoint_sha256"], maximum=MAX_BYTES) != payload):
        raise ValueError("numerical selection assets changed during packaging")
    relocated = dict(config, checkpoint_path=ROOT + "/" + CHECKPOINT_PATH,
        embedding_snapshot=ROOT + "/" + embedding_path)
    raw = _raw(relocated)
    assets.append((CONFIG_PATH, raw))
    return assets, dict(schema="terminal-numerical-selection-assets@1", config=relocated,
        config_path=CONFIG_PATH, config_sha256=hashlib.sha256(raw).hexdigest(),
        source_training_data_included=False)


def _producer_paths():
    from .terminal_numerical_policy import _producer_paths as numerical_producers
    from .terminal_worker_capability import SOURCE_PATHS as worker_paths
    from ipfs_datasets_py.logic.software_contracts import codebase_numerical_selection_384 as native

    source = Path(__file__).resolve().parents[3]
    datasets = Path(native.__file__).resolve().parents[3]
    return {**numerical_producers(),
        **{"source/" + name: source / name for name in (*SOURCE_PATHS, *worker_paths)},
        **{"datasets/" + name: datasets / name for name in DATASET_PATHS}}


def require_numerical_selection_archive(manifest):
    """Bind the adapter, shared grammar, owner, worker and guard to this archive."""
    binding = validate_numerical_selection_binding(manifest)
    if binding is None:
        return None
    rows = {row["path"]: row for row in manifest["files"]}
    for name, path in _producer_paths().items():
        row = rows.get(name, {})
        if (not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= 4 * 1024**2
                or type(row.get("bytes")) is not int or row["bytes"] != path.stat().st_size
                or row.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError("numerical selection requires a newly qualified producer archive")
    return binding


def validate_numerical_selection_binding(manifest):
    """Close both selected and absent populations without opening model assets."""
    if type(manifest) is not dict:
        raise ValueError("numerical selection archive manifest required")
    files = manifest.get("files", [])
    if (type(files) is not list or len(files) > 250_000
            or any(type(row) is not dict or type(row.get("path")) is not str for row in files)):
        raise ValueError("numerical selection archive inventory required")
    by_path = {row["path"]: row for row in files}
    if len(by_path) != len(files):
        raise ValueError("duplicate runtime archive member")
    selected_paths = {name for name in by_path if name.startswith(PREFIX)}
    binding = manifest.get(KEY)
    if binding is None:
        if selected_paths or manifest.get("numerical_selection_requirements"):
            raise ValueError("numerical assets require explicit experimental selection")
        return None
    if (type(binding) is not dict or set(binding) != {"schema", "config", "config_path", "config_sha256", "source_training_data_included"}
            or binding["schema"] != "terminal-numerical-selection-assets@1"
            or binding["config_path"] != CONFIG_PATH or binding["source_training_data_included"] is not False):
        raise ValueError("closed numerical selection deployment binding required")
    config = validate_numerical_selection_config(binding["config"])
    from ipfs_accelerate_py.agent_supervisor.runtime.numerical_selection_config import PUBLISHED_CHECKPOINT_SHA256
    if config["checkpoint_sha256"] != PUBLISHED_CHECKPOINT_SHA256:
        raise ValueError("explicit published numerical checkpoint digest required")
    embedding_path = EMBEDDING_PATH + config["embedding_revision"]
    raw = _raw(config)
    if (config["checkpoint_path"] != ROOT + "/" + CHECKPOINT_PATH
            or config["embedding_snapshot"] != ROOT + "/" + embedding_path
            or binding["config_sha256"] != hashlib.sha256(raw).hexdigest()):
        raise ValueError("numerical relocated identity differs")
    from .terminal_deployment import LEARNED_REQUIREMENTS, TORCH_CPU_REQUIREMENT
    if (manifest.get("numerical_selection_requirements") != list(LEARNED_REQUIREMENTS)
            or manifest.get("torch_cpu_requirement") != TORCH_CPU_REQUIREMENT
            or manifest.get("setup_cache") is not None):
        raise ValueError("numerical selection dependencies or unsupported setup cache differ")
    wanted = [(CHECKPOINT_PATH, config["checkpoint_sha256"], None), (CONFIG_PATH, binding["config_sha256"], len(raw))]
    wanted += [(embedding_path + "/" + row["name"], row["sha256"], row["bytes"]) for row in config["embedding_assets"]]
    if selected_paths != {row[0] for row in wanted}:
        raise ValueError("unexpected numerical selection archive member")
    from ipfs_datasets_py.logic.formalization.autoencoder.numerical_selection_384 import MAX_BYTES
    for name, digest, size in wanted:
        row = by_path[name]
        if (row.get("sha256") != digest or type(row.get("bytes")) is not int or row["bytes"] <= 0
                or (size is not None and row["bytes"] != size)
                or (name == CHECKPOINT_PATH and row["bytes"] > MAX_BYTES)):
            raise ValueError("numerical selection archive member missing or changed")
    return binding


def verify_numerical_selection_archive(archive, manifest):
    binding = require_numerical_selection_archive(manifest)
    producers = set(_producer_paths()) if binding is not None else set()
    wanted = {row["path"]: row for row in manifest.get("files", [])
        if row["path"].startswith(PREFIX) or row["path"] in producers}
    observed = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not member.name.startswith(PREFIX) and member.name not in producers:
                continue
            row = wanted.get(member.name)
            if (row is None or member.name in observed or not member.isfile()
                    or member.size != row["bytes"]):
                raise ValueError("unexpected numerical model or producer archive member")
            observed.add(member.name)
            digest, size = hashlib.sha256(), 0
            with bundle.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    size += len(block)
                    digest.update(block)
            if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("numerical model or producer archive member digest differs")
    if observed != set(wanted):
        raise ValueError("numerical model or producer archive member missing")
    return binding


def numerical_selection_asset_arguments(manifest, arm, resource_profile=None):
    from .benchmark_resource_profile import PROFILES

    if arm not in {"full", "no-index"}:
        raise ValueError("explicit numerical selection ablation required")
    binding = require_numerical_selection_archive(manifest)
    enabled = binding is not None and arm == "full"
    if enabled and resource_profile not in PROFILES:
        raise ValueError("numerical selection requires an explicit common resource profile")
    return (["--numerical-selection-config", ROOT + "/" + CONFIG_PATH] if enabled else []), dict(
        selected=binding is not None, enabled=enabled,
        disabled_reason="no_index_ablation" if binding is not None and not enabled else None,
        config_sha256=binding["config_sha256"] if binding else None,
        checkpoint_sha256=binding["config"]["checkpoint_sha256"] if binding else None,
        training_steps=0, download_calls=0, proof_authority=False, execution_authority=False)
