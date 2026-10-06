"""Deploy the native supervisor inside an existing original Harbor task container.

No task source is copied back for host execution. Installation lives under /opt;
original /app bytes and Git status are measured before/after. Only the supported
Codex auth.json file may be injected separately, never into the runtime archive.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import re
import shlex
import stat
import tarfile
import time
import zipfile
from email.parser import BytesParser
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.runtime.source384_config import (
    load_source384_config, validate_source384_config, validate_source384_assets,
)
from .benchmark_provider_profile import CLI_VERSION as CODEX_VERSION, require_runtime_cli_version

ROOT = "/opt/ipfs-supervisor"
PYTHON = ROOT + "/venv/bin/python"
# UV 0.9.24 migrates a sibling `toolchains` directory when its selected
# Python root is absent. Keep that legacy lookup away from shipped Lean.
PYTHON_INSTALL_DIR = ROOT + "/python-runtime/python"
RUNTIME_PYTHON_VERSION = "3.12.12"
TORCH_CPU_REQUIREMENT = "torch==2.13.0+cpu"
TORCH_CPU_WHEEL_PREFIX = "runtime-wheels/torch-cpu/"
TORCH_CPU_WHEEL_MAX_BYTES = 256 * 1024 * 1024
TORCH_CPU_WHEEL_NAME = r"torch-2\.13\.0\+cpu-cp312-cp312-manylinux_2_28_(aarch64|x86_64)\.whl"
BASE_REQUIREMENTS = (
    "duckdb==1.5.5",
    # DuckDB's scalar binder retries failed optional pandas imports per value.
    "pandas==3.0.2",
    "numpy==1.26.4",
    "multiformats==0.3.1.post4",
    "requests==2.34.2",
    "anyio==4.14.2",
    "cryptography==49.0.0",
    "pytest==9.1.1",
    # Grok structured output is validated before planner admission.
    "jsonschema==4.26.0",
    "referencing==0.37.0",
)
LEARNED_REQUIREMENTS = (
    "sentence-transformers==5.4.1",
    "transformers==4.52.1",
    "safetensors==0.7.0",
    "numpy==1.26.4",
)
SECURITY_TRAINING_REQUIREMENTS = ("numpy==1.26.4",)
NATIVE_IMPORT_PROBE = """import json,sys,duckdb,numpy,pandas
from importlib.metadata import version
from jsonschema import Draft202012Validator
from referencing import Registry
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.todo_daemon.native_owner_bootstrap import NativeOwnerHeartbeat
print(json.dumps({'python':sys.version.split()[0],'duckdb':duckdb.__version__,
 'numpy':numpy.__version__,'pandas':pandas.__version__,
 'jsonschema':version('jsonschema'),'referencing':version('referencing'),
 'native_imports':True}))
"""
SECURITY_INITIALIZER_PATH = "models/security-code-initializer"
SECURITY_INITIALIZER_DESCRIPTOR = "models/security-code-initializer.descriptor.json"
SECURITY_CHECKPOINT_PATH = "models/security-autoencoder"
SECURITY_CHECKPOINT_HUB = "models/security-autoencoder-hub.json"
SECURITY_FORMULA_PATH = "models/security-formula-decoder"
SECURITY_FORMULA_DESCRIPTOR = "models/security-formula-decoder.descriptor.json"
SECURITY_HEADER_PROTOCOL = "models/security-header-protocol.json"
INTENT_CHECKPOINT_PATH = "models/intent-autoencoder/candidate.json"
INTENT_ROUNDTRIP_PATH = "models/intent-autoencoder/manifest.json"
INTENT_CHECKPOINT_DESCRIPTOR = "models/intent-autoencoder.descriptor.json"
INTENT_PROJECTION_REQUEST_PATH = "models/intent-projection-request.json"
INTENT_ACTION_384_PATH = "models/intent-action-384/checkpoint.json"
INTENT_ACTION_384_CONFIG = "models/intent-action-384/config.json"
INTENT_ACTION_384_EMBEDDING = "models/intent-action-384/models--thenlper--gte-small/snapshots/"
SOURCE384_PATH = "models/source384/checkpoint.json"
SOURCE384_CONFIG = "models/source384/config.json"
SOURCE384_EMBEDDING = "models/source384/models--thenlper--gte-small/snapshots/"
CANONICAL_CVE_PATH = "training/canonical-cve"

INPUT_SNAPSHOT = r"""
import hashlib,json,pathlib,subprocess
root=pathlib.Path('/app')
files={}
for raw in subprocess.check_output(['git','-C',str(root),'ls-files','-z']).split(b'\0'):
 if not raw:continue
 name=raw.decode();p=root/name
 if p.is_symlink():raise ValueError('tracked task symlink requires explicit handling')
 if not p.is_file():raise ValueError('tracked task input is missing')
 files[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mode':p.stat().st_mode&0o777}
print(json.dumps({'head':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
 'status':subprocess.check_output(['git','-C',str(root),'status','--porcelain=v1','--untracked-files=all'],text=True),
 'files':files,'report_exists':(root/'report.jsonl').exists()},sort_keys=True))
"""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_asset_bytes(path: Path, *, expected_sha256: str, maximum: int) -> bytes:
    """Capture the exact validated regular file; never follow an asset symlink."""
    path = Path(path).absolute()
    if path.resolve(strict=True) != path or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("canonical independently pinned portable asset required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
            raise ValueError("bounded independent regular portable asset required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    if (len(raw) > maximum or identity(before) != identity(after)
            or identity(after) != identity(path.stat())
            or hashlib.sha256(raw).hexdigest() != expected_sha256):
        raise ValueError("portable asset changed after validation")
    return raw


def _torch_cpu_wheel_metadata(raw: bytes, name: str) -> None:
    """Validate the selected wheel's package/version and exact supported tag."""
    match = re.fullmatch(TORCH_CPU_WHEEL_NAME, name)
    if match is None:
        raise ValueError("exact CPython 3.12 CPU Torch wheel filename required")
    prefix = "torch-2.13.0+cpu.dist-info/"
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as wheel:
            members = wheel.infolist()
            metadata = []
            for leaf in ("METADATA", "WHEEL"):
                found = [row for row in members if row.filename == prefix + leaf]
                if len(found) != 1 or not 0 < found[0].file_size <= 65536:
                    raise ValueError("bounded unique Torch wheel metadata required")
                metadata.append(BytesParser().parsebytes(wheel.read(found[0])))
    except (zipfile.BadZipFile, RuntimeError, KeyError) as exc:
        raise ValueError("valid pinned Torch wheel metadata required") from exc
    package, tags = metadata
    if (package.get_all("Name") != ["torch"] or package.get_all("Version") != ["2.13.0+cpu"]
            or tags.get_all("Wheel-Version") != ["1.0"]
            or tags.get_all("Root-Is-Purelib") != ["false"]
            or tags.get_all("Tag") != ["cp312-cp312-manylinux_2_28_" + match.group(1)]):
        raise ValueError("Torch wheel package, version or platform metadata differs")


def _torch_cpu_wheel_assets(path, expected_sha256):
    if (path is None) != (expected_sha256 is None):
        raise ValueError("Torch CPU wheel path and independent SHA-256 must be paired")
    if path is None:
        return [], None
    if type(expected_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("independent Torch wheel SHA-256 required")
    path = Path(path)
    if re.fullmatch(TORCH_CPU_WHEEL_NAME, path.name) is None:
        raise ValueError("exact CPython 3.12 CPU Torch wheel filename required")
    raw = _portable_asset_bytes(path, expected_sha256=expected_sha256, maximum=TORCH_CPU_WHEEL_MAX_BYTES)
    _torch_cpu_wheel_metadata(raw, path.name)
    name = TORCH_CPU_WHEEL_PREFIX + path.name
    binding = dict(schema="terminal-torch-cpu-wheel@1", path=name, bytes=len(raw),
        sha256=expected_sha256, requirement=TORCH_CPU_REQUIREMENT)
    return [(name, raw)], binding


def validate_torch_cpu_wheel_binding(manifest):
    """Bind explicit wheel selection to exactly one inventory member."""
    binding = manifest.get("torch_cpu_wheel")
    files = manifest.get("files", [])
    if type(files) is not list or any(type(row) is not dict or type(row.get("path")) is not str for row in files):
        raise ValueError("runtime archive inventory required")
    selected = [row for row in files if row["path"].startswith(TORCH_CPU_WHEEL_PREFIX)]
    if binding is None:
        if selected:
            raise ValueError("Torch wheel inventory lacks its explicit selection")
        return None
    if (type(binding) is not dict or set(binding) != {"schema", "path", "bytes", "sha256", "requirement"}
            or binding["schema"] != "terminal-torch-cpu-wheel@1"
            or type(binding["path"]) is not str
            or re.fullmatch(re.escape(TORCH_CPU_WHEEL_PREFIX) + TORCH_CPU_WHEEL_NAME, binding["path"]) is None
            or type(binding["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", binding["sha256"]) is None
            or type(binding["bytes"]) is not int or not 0 < binding["bytes"] <= TORCH_CPU_WHEEL_MAX_BYTES
            or binding["requirement"] != TORCH_CPU_REQUIREMENT
            or manifest.get("torch_cpu_requirement") != TORCH_CPU_REQUIREMENT):
        raise ValueError("closed pinned Torch CPU wheel selection required")
    if len({row["path"] for row in files}) != len(files):
        raise ValueError("duplicate runtime archive member")
    if (len(selected) != 1 or any(selected[0].get(key) != binding[key] for key in ("path", "bytes", "sha256"))
            or type(selected[0].get("bytes")) is not int):
        raise ValueError("Torch wheel inventory missing or changed")
    return binding


def verify_torch_cpu_wheel_archive(archive, manifest):
    binding = validate_torch_cpu_wheel_binding(manifest)
    # The empty selection is closed too: a tar member cannot hide from inventory.
    _verify_model_archive(archive, manifest, TORCH_CPU_WHEEL_PREFIX)
    return binding


def verify_extracted_torch_cpu_wheel(path, expected_sha256, expected_bytes):
    """Recheck actual extracted bytes and interpreter compatibility before pip."""
    import platform
    import sys

    _, binding = _torch_cpu_wheel_assets(path, expected_sha256)
    if (binding is None or binding["bytes"] != expected_bytes or type(expected_bytes) is not int
            or sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12)
            or re.fullmatch(TORCH_CPU_WHEEL_NAME, Path(path).name).group(1) != platform.machine()):
        raise ValueError("extracted Torch wheel or interpreter platform differs")
    return binding


def _torch_cpu_wheel_probe(binding):
    return ("import json; from benchmarks.agent_supervisor.container_coding.terminal_deployment "
        "import verify_extracted_torch_cpu_wheel; print(json.dumps(verify_extracted_torch_cpu_wheel("
        + repr(ROOT + "/" + binding["path"]) + "," + repr(binding["sha256"]) + ","
        + repr(binding["bytes"]) + "),sort_keys=True))")


def _load_initializer_descriptor(path: Path) -> dict:
    from ipfs_datasets_py.logic.formalization.autoencoder.security.codebase_autoencoder_transfer import _read

    return json.loads(_read(Path(path).absolute(), 32_000))


def _security_checkpoint_assets(*, package, manifest_sha256, hub_descriptor=None, cache_root=None):
    """Resolve at controller build time; runtime receives only pinned inert files."""
    if hub_descriptor is not None:
        if package is not None or manifest_sha256 is not None or cache_root is None:
            raise ValueError("pinned Hub checkpoint requires its cache and excludes a local checkpoint")
        from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_hub import download_security_checkpoint
        downloaded = download_security_checkpoint(descriptor=hub_descriptor, cache_root=Path(cache_root))
        package, manifest_sha256 = Path(downloaded["package"]), downloaded["manifest_sha256"]
        hub_descriptor = downloaded["hub"]
    elif cache_root is not None:
        raise ValueError("checkpoint cache requires an explicit pinned Hub descriptor")
    if bool(package) != bool(manifest_sha256):
        raise ValueError("frozen checkpoint package and manifest SHA256 are required together")
    if package is None:
        return [], None
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_autoencoder_checkpoint import PACKAGE_FILES, load_security_checkpoint
    root = Path(package).absolute()
    loaded = load_security_checkpoint(root, expected_manifest_sha256=manifest_sha256)
    assets = []
    for name in sorted(PACKAGE_FILES):
        digest = manifest_sha256 if name == "release-manifest.json" else loaded["manifest"]["files"][name]["sha256"]
        assets.append((SECURITY_CHECKPOINT_PATH + "/" + name,
            _portable_asset_bytes(root / name, expected_sha256=digest, maximum=32_000_000)))
    # Recheck after capturing every file; never package a partly changed model.
    if load_security_checkpoint(root, expected_manifest_sha256=manifest_sha256)["descriptor"] != loaded["descriptor"]:
        raise ValueError("frozen checkpoint changed during runtime packaging")
    if hub_descriptor is not None:
        assets.append((SECURITY_CHECKPOINT_HUB,
            json.dumps(hub_descriptor, sort_keys=True, separators=(",", ":")).encode()))
    return assets, {"path": SECURITY_CHECKPOINT_PATH, "manifest_sha256": manifest_sha256,
        "descriptor": {**loaded["descriptor"], "output": ROOT + "/" + SECURITY_CHECKPOINT_PATH},
        "hub": hub_descriptor, "hub_descriptor_path": SECURITY_CHECKPOINT_HUB if hub_descriptor else None,
        "mode": "frozen_inference", "runtime_download_calls": 0,
        "runtime_training_steps": 0, "source_training_data_included": False}


def _security_formula_assets(*, formula_decoder, header_protocol):
    """Capture a separate frozen production head and explicit reviewed protocol."""
    if formula_decoder is None:
        if header_protocol is not None:
            raise ValueError("reviewed header protocol requires a frozen formula decoder")
        return [], None, None
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_decoder import (
        FILES, MAX_BYTES, load_security_formula_decoder,
    )
    loaded = load_security_formula_decoder(formula_decoder)
    root = Path(formula_decoder["output"])
    assets = []
    for name in sorted(FILES):
        digest = (formula_decoder["manifest_sha256"] if name == "manifest.json"
                  else loaded["manifest"]["files"][name]["sha256"])
        assets.append((SECURITY_FORMULA_PATH + "/" + name,
            _portable_asset_bytes(root / name, expected_sha256=digest, maximum=MAX_BYTES)))
    if load_security_formula_decoder(formula_decoder) != loaded:
        raise ValueError("formula checkpoint changed during runtime packaging")
    descriptor = {**formula_decoder, "output": ROOT + "/" + SECURITY_FORMULA_PATH}
    assets.append((SECURITY_FORMULA_DESCRIPTOR,
        json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()))
    binding = {"path": SECURITY_FORMULA_PATH, "descriptor_path": SECURITY_FORMULA_DESCRIPTOR,
        "descriptor": descriptor, "manifest_sha256": descriptor["manifest_sha256"],
        "mode": "frozen_production_inference", "runtime_training_steps": 0,
        "runtime_download_calls": 0, "source_training_data_included": False}
    protocol_binding = None
    if header_protocol is not None:
        from ipfs_datasets_py.logic.security_ir.doctor_header_contracts import WsgiHeaderProtocolContract
        if type(header_protocol) is not dict or set(header_protocol) != {"review_ref", "callback_parameter"}:
            raise ValueError("explicit closed reviewed header protocol required")
        WsgiHeaderProtocolContract(**header_protocol)
        raw = json.dumps(header_protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        assets.append((SECURITY_HEADER_PROTOCOL, raw))
        protocol_binding = {"path": SECURITY_HEADER_PROTOCOL,
            "sha256": hashlib.sha256(raw).hexdigest(), "protocol": header_protocol}
    return assets, binding, protocol_binding


def _intent_checkpoint_assets(descriptor):
    """Package a selected inert Intent model, without a training corpus."""
    if descriptor is None:
        return [], None
    from ipfs_datasets_py.logic.intent_ir.formalize.preplanning import (
        MAX_CHECKPOINT_BYTES, load_intent_feature_checkpoint,
    )
    if type(descriptor) is dict and descriptor.get("schema") in {
            "intent-roundtrip-checkpoint/v1", "intent-copy-roundtrip-checkpoint/v1"}:
        is_copy = descriptor["schema"] == "intent-copy-roundtrip-checkpoint/v1"
        if is_copy:
            from ipfs_datasets_py.logic.intent_ir.formalize.copy_roundtrip import load_intent_copy_checkpoint as load_checkpoint
            from ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_paired_copy import MAX_BYTES as PAIRED_MAX_BYTES
        else:
            from ipfs_datasets_py.logic.intent_ir.formalize.roundtrip import load_intent_roundtrip_checkpoint as load_checkpoint
            from ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_paired_text import MAX_BYTES as PAIRED_MAX_BYTES

        loaded = load_checkpoint(descriptor)
        backend = loaded["manifest"]["backend"]
        if (type(backend) is not dict or set(backend) != {"schema", "file", "sha256"}
                or type(backend["file"]) is not str
                or not re.fullmatch(r"model/[A-Za-z0-9][A-Za-z0-9._-]*\.json", backend["file"])):
            raise ValueError("roundtrip backend must be one colocated inert model JSON")
        checkpoint_path = INTENT_ROUNDTRIP_PATH
        mode = "frozen_copy_roundtrip_inference" if is_copy else "frozen_semantic_roundtrip_inference"
        model_path = Path(descriptor["path"]).parent / backend["file"]
        payload = _portable_asset_bytes(Path(descriptor["path"]), expected_sha256=descriptor["sha256"],
            maximum=65536)
        model_payload = _portable_asset_bytes(model_path, expected_sha256=backend["sha256"],
            maximum=PAIRED_MAX_BYTES)
        load_checkpoint(descriptor)
        assets = [(checkpoint_path, payload),
                  (str(Path(checkpoint_path).parent / backend["file"]), model_payload)]
    else:
        load_intent_feature_checkpoint(descriptor)
        payload = _portable_asset_bytes(Path(descriptor["path"]), expected_sha256=descriptor["sha256"],
            maximum=MAX_CHECKPOINT_BYTES)
        load_intent_feature_checkpoint(descriptor)
        checkpoint_path = INTENT_CHECKPOINT_PATH
        mode = "frozen_structural_feature_inference"
        assets = [(checkpoint_path, payload)]
    relocated = {**descriptor, "path": ROOT + "/" + checkpoint_path}
    assets.append((INTENT_CHECKPOINT_DESCRIPTOR,
        json.dumps(relocated, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()))
    return assets, {"path": checkpoint_path, "descriptor_path": INTENT_CHECKPOINT_DESCRIPTOR,
        "descriptor": relocated, "sha256": descriptor["sha256"],
        "mode": mode, "runtime_training_steps": 0,
        "runtime_download_calls": 0, "source_training_data_included": False}



def load_intent_action_384_config(path: Path) -> dict:
    """Read an explicitly selected bounded JSON file with no duplicate keys."""
    from ipfs_accelerate_py.agent_supervisor.runtime.intent_384_advisor import _config
    path = Path(path).absolute()
    if path.resolve(strict=True) != path or not path.is_file() or path.stat().st_size > 32768:
        raise ValueError("bounded canonical local Intent384 config required")
    raw = _portable_asset_bytes(path, expected_sha256=_sha(path), maximum=32768)
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate Intent384 config field")
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError("nonfinite Intent384 config value")
    return _config(json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite))


def _intent_action_384_assets(config):
    """Capture shared Intent weights and the exact offline GTE inference assets."""
    if config is None:
        return [], None
    from ipfs_accelerate_py.agent_supervisor.runtime.intent_384_advisor import _config
    from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384 as structured
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding
    config = _config(config)
    if config["embedding_snapshot_path"] is None:
        raise ValueError("Intent384 packaging requires an explicit local embedding snapshot")
    checkpoint = Path(config["checkpoint_path"])
    payload = _portable_asset_bytes(checkpoint, expected_sha256=config["checkpoint_sha256"],
        maximum=structured.MAX_BYTES)
    structured.load_checkpoint(checkpoint, expected_sha256=config["checkpoint_sha256"], expected_domain="intent_ir")
    snapshot, manifest = embedding._snapshot_assets(config["embedding_snapshot_path"])
    embedding_path = INTENT_ACTION_384_EMBEDDING + embedding.PINNED_REVISION
    assets = [(INTENT_ACTION_384_PATH, payload)]
    for item in manifest:
        # Native validation permits only the exact cached model/blob targets.
        # Capture resolved regular bytes; the container needs no host symlinks.
        assets.append((embedding_path + "/" + item["name"],
            _portable_asset_bytes((snapshot / item["name"]).resolve(strict=True),
                expected_sha256=item["sha256"], maximum=item["bytes"])))
    if embedding._snapshot_assets(snapshot)[1] != manifest or _sha(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Intent384 assets changed during packaging")
    relocated = {**config, "checkpoint_path": ROOT + "/" + INTENT_ACTION_384_PATH,
        "embedding_snapshot_path": ROOT + "/" + embedding_path}
    raw = json.dumps(relocated, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assets.append((INTENT_ACTION_384_CONFIG, raw))
    binding = {"schema": "terminal-intent-action-384-assets@1", "path": INTENT_ACTION_384_PATH,
        "config_path": INTENT_ACTION_384_CONFIG, "config": relocated,
        "config_sha256": hashlib.sha256(raw).hexdigest(), "checkpoint_sha256": config["checkpoint_sha256"],
        "embedding_path": embedding_path, "embedding_revision": embedding.PINNED_REVISION,
        "embedding_assets": manifest, "mode": "frozen_source_audited_action_384_inference",
        "runtime_training_steps": 0, "runtime_download_calls": 0, "source_training_data_included": False}
    return assets, binding


def validate_intent_action_384_binding(manifest):
    """Check fixed relocated paths and every model member against native pins."""
    binding = manifest.get("intent_action_384")
    if binding is None:
        return None
    from ipfs_accelerate_py.agent_supervisor.runtime.intent_384_advisor import _config
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding
    fields = {"schema", "path", "config_path", "config", "config_sha256", "checkpoint_sha256",
        "embedding_path", "embedding_revision", "embedding_assets", "mode", "runtime_training_steps",
        "runtime_download_calls", "source_training_data_included"}
    if type(binding) is not dict or set(binding) != fields:
        raise ValueError("closed Intent384 deployment binding required")
    config = _config(binding["config"])
    embedding_path = INTENT_ACTION_384_EMBEDDING + embedding.PINNED_REVISION
    expected = [{"name": name, "sha256": sha, "bytes": size}
        for name, (size, sha) in sorted(embedding._PINNED_ASSETS.items())]
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if (binding["schema"] != "terminal-intent-action-384-assets@1"
            or binding["path"] != INTENT_ACTION_384_PATH or binding["config_path"] != INTENT_ACTION_384_CONFIG
            or config["checkpoint_path"] != ROOT + "/" + INTENT_ACTION_384_PATH
            or binding["checkpoint_sha256"] != config["checkpoint_sha256"]
            or config["embedding_snapshot_path"] != ROOT + "/" + embedding_path
            or binding["embedding_path"] != embedding_path or binding["embedding_revision"] != embedding.PINNED_REVISION
            or binding["embedding_assets"] != expected
            or binding["config_sha256"] != hashlib.sha256(raw).hexdigest()
            or binding["mode"] != "frozen_source_audited_action_384_inference"
            or type(binding["runtime_training_steps"]) is not int or binding["runtime_training_steps"] != 0
            or type(binding["runtime_download_calls"]) is not int or binding["runtime_download_calls"] != 0
            or binding["source_training_data_included"] is not False):
        raise ValueError("Intent384 deployment identity or asset pins differ")
    files = manifest.get("files")
    if type(files) is not list or any(type(row) is not dict or type(row.get("path")) is not str for row in files):
        raise ValueError("Intent384 archive inventory required")
    by_path = {row["path"]: row for row in files}
    if len(by_path) != len(files):
        raise ValueError("duplicate runtime archive member")
    wanted = [(INTENT_ACTION_384_PATH, config["checkpoint_sha256"], None),
        (INTENT_ACTION_384_CONFIG, binding["config_sha256"], len(raw))]
    wanted += [(embedding_path + "/" + row["name"], row["sha256"], row["bytes"]) for row in expected]
    if {name for name in by_path if name.startswith("models/intent-action-384/")} != {row[0] for row in wanted}:
        raise ValueError("unexpected Intent384 archive member")
    for name, digest, size in wanted:
        row = by_path.get(name, {})
        if (row.get("sha256") != digest or type(row.get("bytes")) is not int
                or row["bytes"] <= 0 or (size is not None and row["bytes"] != size)):
            raise ValueError("Intent384 archive member missing or changed")
    return binding


def _source384_assets(config):
    """Transport exact shared parent bytes; never train, download or add task sources."""
    if config is None:
        return [], None
    from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384 as structured
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_embedding_runtime as embedding
    config = validate_source384_assets(config)
    checkpoint = Path(config["checkpoint_path"])
    payload = _portable_asset_bytes(checkpoint, expected_sha256=config["checkpoint_sha256"], maximum=structured.MAX_BYTES)
    snapshot, manifest = embedding._snapshot_assets(config["embedding_snapshot"])
    embedding_path = SOURCE384_EMBEDDING + embedding.PINNED_REVISION
    assets = [(SOURCE384_PATH, payload)]
    for item in manifest:
        assets.append((embedding_path + "/" + item["name"],
            _portable_asset_bytes((snapshot / item["name"]).resolve(strict=True),
                expected_sha256=item["sha256"], maximum=item["bytes"])))
    if (embedding._snapshot_assets(snapshot)[1] != manifest
            or _portable_asset_bytes(checkpoint, expected_sha256=config["checkpoint_sha256"],
                                     maximum=structured.MAX_BYTES) != payload):
        raise ValueError("Source384 assets changed during packaging")
    relocated = {**config, "checkpoint_path": ROOT + "/" + SOURCE384_PATH,
                 "embedding_snapshot": ROOT + "/" + embedding_path}
    raw = json.dumps(relocated, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assets.append((SOURCE384_CONFIG, raw))
    return assets, dict(schema="terminal-source384-assets@1", config=relocated,
        config_path=SOURCE384_CONFIG, config_sha256=hashlib.sha256(raw).hexdigest(),
        source_training_data_included=False)


def validate_source384_binding(manifest):
    """Validate relocated selection and the entire selected archive inventory."""
    binding = manifest.get("source384")
    if binding is None:
        return None
    if (type(binding) is not dict or set(binding) != {"schema", "config", "config_path", "config_sha256", "source_training_data_included"}
            or binding["schema"] != "terminal-source384-assets@1"
            or binding["config_path"] != SOURCE384_CONFIG or binding["source_training_data_included"] is not False):
        raise ValueError("closed Source384 deployment binding required")
    config = validate_source384_config(binding["config"])
    embedding_path = SOURCE384_EMBEDDING + config["embedding_revision"]
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if (config["checkpoint_path"] != ROOT + "/" + SOURCE384_PATH
            or config["embedding_snapshot"] != ROOT + "/" + embedding_path
            or binding["config_sha256"] != hashlib.sha256(raw).hexdigest()):
        raise ValueError("Source384 relocated identity differs")
    files = manifest.get("files")
    if type(files) is not list or any(type(row) is not dict or type(row.get("path")) is not str for row in files):
        raise ValueError("Source384 archive inventory required")
    by_path = {row["path"]: row for row in files}
    if len(by_path) != len(files):
        raise ValueError("duplicate runtime archive member")
    wanted = [(SOURCE384_PATH, config["checkpoint_sha256"], None), (SOURCE384_CONFIG, binding["config_sha256"], len(raw))]
    wanted += [(embedding_path + "/" + row["name"], row["sha256"], row["bytes"]) for row in config["embedding_assets"]]
    if {name for name in by_path if name.startswith("models/source384/")} != {row[0] for row in wanted}:
        raise ValueError("unexpected Source384 archive member")
    from ipfs_datasets_py.logic.formalization.autoencoder import structured_source_384 as structured
    for name, digest, size in wanted:
        row = by_path.get(name, {})
        if (row.get("sha256") != digest or type(row.get("bytes")) is not int or row["bytes"] <= 0
                or (size is not None and row["bytes"] != size)
                or (name == SOURCE384_PATH and row["bytes"] > structured.MAX_BYTES)):
            raise ValueError("Source384 archive member missing or changed")
    if any(manifest.get(key) is not None for key in ("security_initializer", "canonical_cve_training",
            "security_checkpoint", "formula_decoder", "header_protocol")):
        raise ValueError("Source384 and legacy Security model selections are mutually exclusive")
    return binding


def verify_source384_archive(archive: Path, manifest: dict):
    if validate_source384_binding(manifest) is not None:
        _verify_model_archive(archive, manifest, "models/source384/")


def _source384_header_checker_probe(binding):
    """Verify the selected runtime-installed binary; never infer or update its pin."""
    if binding is None or binding["config"].get("header_applicability") is None:
        return None
    expected = binding["config"]["header_applicability"]
    # JSON data is a Python string literal here; the command caller uses shlex.
    return (
        "import json\n"
        "from ipfs_accelerate_py.agent_supervisor.runtime.header_intent_applicability import resolve_header_checker\n"
        "profile=json.loads(" + repr(json.dumps(expected, sort_keys=True, allow_nan=False)) + ")\n"
        "print(json.dumps(resolve_header_checker(profile),sort_keys=True))\n"
    )


def verify_intent_action_384_archive(archive: Path, manifest: dict):
    """Check selected inert model bytes before uploading the runtime archive."""
    binding = validate_intent_action_384_binding(manifest)
    if binding is None:
        return
    _verify_model_archive(archive, manifest, "models/intent-action-384/")


def _verify_model_archive(archive, manifest, prefix):
    """Shared regular-member verifier for both closed portable model selections."""
    wanted = {row["path"]: row for row in manifest["files"]
              if row["path"].startswith(prefix)}
    observed = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not member.name.startswith(prefix):
                continue
            row = wanted.get(member.name)
            if (row is None or member.name in observed or not member.isfile()
                    or member.size != row["bytes"]):
                raise ValueError("unexpected or changed portable model archive member")
            observed.add(member.name)
            digest = hashlib.sha256()
            count = 0
            with bundle.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    count += len(block)
                    digest.update(block)
            if count != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("portable model archive member digest differs")
    if observed != set(wanted):
        raise ValueError("portable model archive member missing")

def _intent_projection_request_assets(request, checkpoint):
    """Transport one inert, task-bound request; runtime verifies its candidate."""
    if request is None:
        return [], None
    from ipfs_datasets_py.logic.intent_ir.formalize.projection_request import validate_projection_request_shape

    validate_projection_request_shape(request)
    if (type(checkpoint) is not dict or checkpoint.get("schema") not in
            {"intent-roundtrip-checkpoint/v1", "intent-copy-roundtrip-checkpoint/v1"}
            or request["checkpoint_sha256"] != checkpoint.get("sha256")):
        raise ValueError("Intent projection request requires its matching semantic roundtrip checkpoint")
    raw = json.dumps(request, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(raw) > 65536:
        raise ValueError("Intent projection request exceeds its portable byte bound")
    binding = {"schema": request["schema"], "path": INTENT_PROJECTION_REQUEST_PATH,
        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
        **{key: request[key] for key in ("request_sha256", "checkpoint_sha256",
                                        "instruction_sha256", "source_ir_sha256")}}
    return [(INTENT_PROJECTION_REQUEST_PATH, raw)], binding


def _security_training_assets(*, security_initializer, canonical_cve_export,
                              canonical_cve_manifest_sha256):
    """Validate native contracts and capture only portable derived artifacts."""
    assets = []
    bindings = {"security_initializer": None, "canonical_cve_training": None}
    if bool(canonical_cve_export) != bool(canonical_cve_manifest_sha256):
        raise ValueError("canonical CVE export and independent manifest SHA256 are required together")
    if security_initializer is not None:
        from ipfs_datasets_py.logic.formalization.autoencoder.security.codebase_autoencoder_transfer import validate_legal_shared_weight_fork

        initializer = validate_legal_shared_weight_fork(expected_receipt=security_initializer, replay_source=True)
        root = Path(security_initializer["output"])
        for name, key, maximum in (("manifest.json", "manifest_sha256", 32_000),
                                   ("initializer.json", "initializer_sha256", 4_000_000)):
            assets.append((SECURITY_INITIALIZER_PATH + "/" + name,
                _portable_asset_bytes(root / name, expected_sha256=security_initializer[key], maximum=maximum)))
        descriptor = {**security_initializer, "output": ROOT + "/" + SECURITY_INITIALIZER_PATH}
        assets.append((SECURITY_INITIALIZER_DESCRIPTOR,
            json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()))
        bindings["security_initializer"] = {
            "path": SECURITY_INITIALIZER_PATH, "descriptor_path": SECURITY_INITIALIZER_DESCRIPTOR,
            "descriptor": descriptor, "source_checkpoint_included": False,
            "host_source_replayed": True,
            "source_lineage": {key: initializer[key] for key in (
                "source_state_digest", "source_revision", "source_architecture", "source_schema",
                "source_float_precision", "source_validation", "source_checkpoint_sha256")},
        }
    if canonical_cve_export is not None:
        from ipfs_datasets_py.logic.formalization.autoencoder.security.security_cve_canonical_export import load_canonical_cve_training

        root = Path(canonical_cve_export).absolute()
        loaded = load_canonical_cve_training(root, expected_manifest_sha256=canonical_cve_manifest_sha256)
        manifest = loaded["manifest"]
        if (any(manifest.get(key) is not False for key in (
                "raw_bodies_persisted", "raw_body_excerpts_exported", "legal_state_modified",
                "proof_authoritative", "grants_execution_authority", "target_code_executed"))
                or type(manifest.get("provider_calls")) is not int or manifest["provider_calls"] != 0
                or any(row.get("raw_body_persisted") is not False for row in manifest["selected_rows"])):
            raise ValueError("only derived non-authoritative canonical CVE training assets may be packaged")
        assets.append((CANONICAL_CVE_PATH + "/manifest.json",
            _portable_asset_bytes(root / "manifest.json", expected_sha256=canonical_cve_manifest_sha256,
                                  maximum=8 * 1024 * 1024)))
        for name in ("canonical-records.json", "training-pairs.json"):
            assets.append((CANONICAL_CVE_PATH + "/" + name,
                _portable_asset_bytes(root / name, expected_sha256=manifest["artifacts"][name]["sha256"],
                                      maximum=64 * 1024 * 1024)))
        bindings["canonical_cve_training"] = {
            "path": CANONICAL_CVE_PATH, "manifest_sha256": canonical_cve_manifest_sha256,
            "schema": manifest["schema"], "namespace": manifest["namespace"],
            "canonical_dataset_cid": manifest["canonical_dataset_cid"],
            "canonical_record_count": manifest["canonical_record_count"],
            "training_pair_count": manifest["training_pair_count"],
            "source_export_refs": {key: manifest[key] for key in (
                "pin", "routing_index_sha256", "graph_artifacts", "selected_rows")},
            "raw_source_included": False,
        }
    return assets, bindings


def runtime_environment() -> dict[str, str]:
    return {
        "HOME": ROOT + "/home",
        "PATH": ROOT + "/provider-bin:" + ROOT + "/venv/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": ROOT + "/source:" + ROOT + "/datasets:" + ROOT + "/kit",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "DOCTOR_COMPOSITION_LEAN": ROOT + "/toolchains/lean/bin/lean",
    }


def native_codex_exposure_script(root: str = ROOT) -> str:
    """Expose and execute-check the complete pinned vendor bin directory as root."""
    return "root_value=" + repr(root) + "\nexpected_version=" + repr(CODEX_VERSION) + "\n" + r'''
import hashlib,json,os,pathlib,shutil,stat,subprocess
if os.geteuid()!=0:raise ValueError('native Codex deployment requires container root')
root=pathlib.Path(root_value)
matches=list((root/'home/.nvm').glob('versions/node/*/lib/node_modules/@openai/codex/node_modules/@openai/codex-*/vendor/*/bin/codex'))
if len(matches)!=1:raise ValueError('one pinned native Codex executable required')
vendor=matches[0].parent
members=sorted(vendor.iterdir())
if not 2<=len(members)<=32 or not {'codex','codex-code-mode-host'} <= {p.name for p in members}:
 raise ValueError('pinned native Codex runtime is missing required executables')
for p in members:
 if p.is_symlink() or not p.is_file() or not p.stat().st_mode&0o111:
  raise ValueError('native Codex runtime must contain regular executable files')
target=root/'provider-bin'
if target.is_symlink():raise ValueError('native Codex runtime target cannot be a symlink')
target.mkdir(mode=0o755,exist_ok=True)
os.chown(target,0,0);target.chmod(0o755)
for p in members:
 destination=target/p.name
 if destination.is_symlink():raise ValueError('native Codex executable target cannot be a symlink')
 shutil.copyfile(p,destination);os.chown(destination,0,0);destination.chmod(0o755)
checks={}
for name,args in (('codex',['--version']),('codex-code-mode-host',['--help'])):
 completed=subprocess.run([str(target/name),*args],capture_output=True,text=True,timeout=10)
 if completed.returncode or (name=='codex' and completed.stdout.strip()!='codex-cli '+expected_version):
  raise ValueError('pinned native Codex executable check failed: '+name)
 checks[name]={'returncode':completed.returncode,'stdout_sha256':hashlib.sha256(completed.stdout.encode()).hexdigest()}
files=[{'name':p.name,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'sha256':hashlib.sha256((target/p.name).read_bytes()).hexdigest(),
        'uid':(target/p.name).stat().st_uid,'mode':(target/p.name).stat().st_mode&0o777} for p in members]
if any(x['uid']!=0 or x['mode']!=0o755 or x['sha256']!=x['source_sha256'] for x in files):
 raise ValueError('native Codex executable protection or vendor bytes differ')
print(json.dumps({'schema':'native-codex-runtime-bundle@1','codex_version':expected_version,
                  'files':files,'executable_checks':checks,'provider_calls':0},sort_keys=True))
'''


def build_runtime_archive(
    *,
    output: Path,
    source: Path,
    datasets: Path,
    kit: Path,
    extension_dir: Path,
    lean_toolchain: Path | None = None,
    model_snapshot: Path | None = None,
    security_initializer: dict | None = None,
    canonical_cve_export: Path | None = None,
    canonical_cve_manifest_sha256: str | None = None,
    security_checkpoint: Path | None = None,
    security_checkpoint_manifest_sha256: str | None = None,
    security_checkpoint_hub_descriptor: dict | None = None,
    security_checkpoint_cache: Path | None = None,
    formula_decoder: dict | None = None,
    header_protocol: dict | None = None,
    intent_checkpoint: dict | None = None,
    intent_projection_request: dict | None = None,
    intent_action_384_config: dict | None = None,
    source384_config: dict | None = None,
    torch_cpu_wheel: Path | None = None,
    torch_cpu_wheel_sha256: str | None = None,
    setup_cache_policy: str | None = None,
    grok_binary: Path | None = None,
) -> dict:
    """Package only source/code assets and explicitly selected native runtimes."""
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("fresh runtime archive directory required")
    if grok_binary is not None:
        from .terminal_grok_deployment import validate_grok_binary
        grok_binary = validate_grok_binary(grok_binary)
        if setup_cache_policy is not None:
            raise ValueError("Grok runtime does not admit the Codex setup cache policy")
    if setup_cache_policy is not None:
        import platform
        from .terminal_setup_cache_advice import POLICIES
        if (setup_cache_policy not in POLICIES or platform.machine() != "aarch64"
                or source384_config is None or model_snapshot is not None
                or any(value is not None for value in (intent_checkpoint,
                    intent_projection_request, intent_action_384_config))):
            raise ValueError("setup cache policy requires the supported regular Source384-only aarch64 layout")
    if (torch_cpu_wheel is None) != (torch_cpu_wheel_sha256 is None):
        raise ValueError("Torch CPU wheel path and independent SHA-256 must be paired")
    if source384_config is not None and any(value is not None for value in (
            security_initializer, canonical_cve_export, canonical_cve_manifest_sha256,
            security_checkpoint, security_checkpoint_manifest_sha256, security_checkpoint_hub_descriptor,
            security_checkpoint_cache, formula_decoder, header_protocol)):
        raise ValueError("Source384 and legacy Security model selections are mutually exclusive")
    source384_assets, source384_binding = _source384_assets(source384_config)
    frozen_selected = any(value is not None for value in (security_checkpoint,
        security_checkpoint_manifest_sha256, security_checkpoint_hub_descriptor, security_checkpoint_cache))
    if frozen_selected and any(value is not None for value in (
            security_initializer, canonical_cve_export, canonical_cve_manifest_sha256)):
        raise ValueError("frozen checkpoint and local security training assets are mutually exclusive")
    if formula_decoder is not None and not frozen_selected:
        raise ValueError("formula decoder requires the frozen security checkpoint profile")
    formula_assets, formula_binding, protocol_binding = _security_formula_assets(
        formula_decoder=formula_decoder, header_protocol=header_protocol)
    checkpoint_assets, checkpoint_binding = _security_checkpoint_assets(package=security_checkpoint,
        manifest_sha256=security_checkpoint_manifest_sha256, hub_descriptor=security_checkpoint_hub_descriptor,
        cache_root=security_checkpoint_cache)
    portable_assets, security_bindings = _security_training_assets(
        security_initializer=security_initializer, canonical_cve_export=canonical_cve_export,
        canonical_cve_manifest_sha256=canonical_cve_manifest_sha256)
    security_training_selected = bool(portable_assets)
    if intent_action_384_config is not None and any(value is not None for value in (intent_checkpoint, intent_projection_request)):
        raise ValueError("Intent384 and legacy Intent model selections are mutually exclusive")
    intent_action_assets, intent_action_binding = _intent_action_384_assets(intent_action_384_config)
    intent_assets, intent_binding = _intent_checkpoint_assets(intent_checkpoint)
    intent_request_assets, intent_request_binding = _intent_projection_request_assets(
        intent_projection_request, intent_checkpoint)
    portable_assets.extend(checkpoint_assets)
    portable_assets.extend(formula_assets)
    portable_assets.extend(intent_assets)
    portable_assets.extend(intent_request_assets)
    portable_assets.extend(intent_action_assets)
    portable_assets.extend(source384_assets)
    torch_required = bool(model_snapshot or portable_assets)
    if torch_cpu_wheel is not None and not torch_required:
        raise ValueError("Torch wheel requires an already selected Torch runtime feature")
    torch_assets, torch_binding = _torch_cpu_wheel_assets(torch_cpu_wheel, torch_cpu_wheel_sha256)
    portable_assets.extend(torch_assets)
    security_bindings["security_checkpoint"] = checkpoint_binding
    if formula_binding is not None:
        security_bindings["formula_decoder"] = formula_binding
    if protocol_binding is not None:
        security_bindings["header_protocol"] = protocol_binding
    if intent_binding is not None:
        security_bindings["intent_checkpoint"] = intent_binding
    if intent_request_binding is not None:
        security_bindings["intent_projection_request"] = intent_request_binding
    if intent_action_binding is not None:
        security_bindings["intent_action_384"] = intent_action_binding
    if source384_binding is not None:
        security_bindings["source384"] = source384_binding
    if torch_binding is not None:
        security_bindings["torch_cpu_wheel"] = torch_binding
    output.mkdir(parents=True)
    files = []

    def add(path: Path, name: str):
        if not path.is_file() or path.is_symlink():
            raise ValueError("runtime source must be a regular non-symlink file")
        files.append((path, name))

    for checkout, package, destination in (
        (Path(source), "ipfs_accelerate_py", "source"),
        (Path(datasets), "ipfs_datasets_py", "datasets"),
        (Path(kit), "ipfs_kit_py", "kit"),
    ):
        checkout = checkout.resolve(strict=True)
        for path in sorted((checkout / package).rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(checkout)
            if any(
                part in {"__pycache__", ".git", "node_modules", ".venv"} for part in relative.parts
            ):
                continue
            if path.suffix not in {".py", ".lean", ".sql", ".lark"} and not (
                path.suffix == ".json"
                and ("schemas" in relative.parts or path.name == "intelligence_index_v4_3.json")
            ):
                continue
            add(path, destination + "/" + relative.as_posix())
        if package == "ipfs_accelerate_py":
            for path in sorted(
                (checkout / "benchmarks/agent_supervisor/container_coding").glob("*.py")
            ):
                if not path.name.startswith("test_"):
                    add(path, "source/" + path.relative_to(checkout).as_posix())
    for extension in ("ducklake", "httpfs", "quack"):
        add(
            Path(extension_dir) / f"{extension}.duckdb_extension",
            f"extensions/{extension}.duckdb_extension",
        )
    if grok_binary is not None:
        from .terminal_grok_deployment import GROK_PATH
        add(grok_binary, GROK_PATH)
    if lean_toolchain is not None:
        toolchain = Path(lean_toolchain).resolve(strict=True)
        if not (toolchain / "bin/lean").is_file():
            raise ValueError("actual Lean executable required")
        for part in ("bin", "lib"):
            for path in sorted((toolchain / part).rglob("*")):
                if not path.is_file():
                    continue
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(toolchain):
                    raise ValueError("Lean asset escapes toolchain")
                add(resolved, "toolchains/lean/" + path.relative_to(toolchain).as_posix())
    if model_snapshot is not None:
        snapshot = Path(model_snapshot).resolve(strict=True)
        if not re.fullmatch(r"[0-9a-f]{40}", snapshot.name):
            raise ValueError("pinned HuggingFace snapshot revision directory required")
        allowed = {".json", ".txt", ".safetensors"}
        for path in sorted(snapshot.rglob("*")):
            if not path.is_file() or path.suffix not in allowed:
                continue
            resolved = path.resolve(strict=True)
            # HuggingFace snapshots use file symlinks into sibling blobs.
            if path.is_symlink() and not resolved.is_relative_to(snapshot.parents[1] / "blobs"):
                raise ValueError("model file points outside its cached blobs")
            add(resolved, "models/embedding/" + path.relative_to(snapshot).as_posix())
    inventory = []
    archive = output / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
        for path, name in files:
            entry = bundle.gettarinfo(str(path), arcname=name)
            entry.uid = entry.gid = 0
            entry.uname = entry.gname = "root"
            entry.mtime = 0
            entry.mode = 0o755 if os.access(path, os.X_OK) else 0o644
            with path.open("rb") as stream:
                bundle.addfile(entry, stream)
            inventory.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha(path),
                    "mode": entry.mode,
                }
            )
        for name, payload in portable_assets:
            entry = tarfile.TarInfo(name)
            entry.uid = entry.gid = 0
            entry.uname = entry.gname = "root"
            entry.mtime = 0
            entry.mode = 0o644
            entry.size = len(payload)
            bundle.addfile(entry, io.BytesIO(payload))
            inventory.append({"path": name, "bytes": len(payload),
                              "sha256": hashlib.sha256(payload).hexdigest(), "mode": entry.mode})
    manifest = {
        "schema": "terminal-supervisor-runtime-archive@1",
        "archive_sha256": _sha(archive),
        "files": inventory,
        "base_requirements": list(BASE_REQUIREMENTS),
        "learned_requirements": list(LEARNED_REQUIREMENTS) if model_snapshot else [],
        "torch_cpu_requirement": TORCH_CPU_REQUIREMENT if torch_required else "",
        "intent_action_384_requirements": list(LEARNED_REQUIREMENTS) if intent_action_binding is not None else [],
        "source384_requirements": list(LEARNED_REQUIREMENTS) if source384_binding is not None else [],
        "security_training_requirements": list(SECURITY_TRAINING_REQUIREMENTS) if security_training_selected else [],
        "security_inference_requirements": list(SECURITY_TRAINING_REQUIREMENTS) if checkpoint_binding is not None else [],
        **security_bindings,
        "codex_version": CODEX_VERSION,
        "runtime_python_version": RUNTIME_PYTHON_VERSION,
        "model_snapshot_revision": Path(model_snapshot).resolve().name if model_snapshot else None,
        "credentials_in_archive": False,
        "task_inputs_in_archive": intent_request_binding is not None,
        **({"task_modeling_premises_included": True} if intent_request_binding is not None else {}),
    }
    if grok_binary is not None:
        from .terminal_grok_deployment import grok_binding, validate_grok_binding
        manifest["grok_cli_assets"] = grok_binding()
        validate_grok_binding(manifest, required=True)
    validate_intent_action_384_binding(manifest)
    validate_source384_binding(manifest)
    validate_torch_cpu_wheel_binding(manifest)
    if setup_cache_policy is not None:
        from .terminal_setup_cache_advice import binding_for_manifest
        manifest["setup_cache"] = binding_for_manifest(manifest, setup_cache_policy)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if setup_cache_policy is not None:
        from .terminal_setup_cache_advice import select_setup_cache
        selection = select_setup_cache(output, setup_cache_policy)
        (output / "setup-cache-selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    return manifest


async def deploy_supervisor(
    environment,
    *,
    archive_dir: Path,
    output: Path,
    auth_json: Path | None = None,
    install_codex: bool = True,
    setup_cache_selection: dict | None = None,
    isolated_uv_bootstrap: bool = False,
    provider: str = "codex_cli",
) -> dict:
    """Use actual Harbor exec/upload APIs; does not call any model or verifier."""
    if provider not in {"codex_cli", "grok_cli"}:
        raise ValueError("explicit supported container provider required")
    if provider == "grok_cli" and (install_codex or setup_cache_selection is not None or auth_json is None):
        raise ValueError("Grok deployment requires its auth file, no Codex install and no Codex cache policy")
    if auth_json is not None:
        from .terminal_grok_deployment import validate_auth_file
        auth_json = validate_auth_file(auth_json)
    from .terminal_setup_cache_advice import validate_setup_cache_selection, validate_setup_cache_prerequisites
    validate_setup_cache_selection(archive_dir, setup_cache_selection)
    validate_setup_cache_prerequisites(setup_cache_selection, install_codex=install_codex, auth_json=auth_json)
    manifest = json.loads((Path(archive_dir) / "manifest.json").read_text())
    if install_codex:
        require_runtime_cli_version(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    archive = Path(archive_dir) / "runtime.tar.gz"
    if _sha(archive) != manifest["archive_sha256"]:
        raise ValueError("runtime archive changed")
    from .terminal_grok_deployment import verify_grok_archive, grok_exposure_script
    grok_binding = verify_grok_archive(archive, manifest, required=provider == "grok_cli")
    torch_binding = verify_torch_cpu_wheel_archive(archive, manifest)
    verify_intent_action_384_archive(archive, manifest)
    verify_source384_archive(archive, manifest)
    started = time.monotonic()
    steps = []

    async def execute(label, command, *, user="root", env=None, timeout=300, record=True):
        result = await environment.exec(
            command=command, cwd="/app", user=user, env=env, timeout_sec=timeout
        )
        steps.append({"step": label, "return_code": result.return_code})
        if record:
            (output / (label + ".log")).write_text((result.stdout or "") + (result.stderr or ""))
        if result.return_code:
            raise RuntimeError(f"container deployment {label} failed; see retained log")
        return result

    before = json.loads(
        (await execute("original-inputs", "python3 -I -c " + shlex.quote(INPUT_SNAPSHOT))).stdout
    )
    await execute("root-create", "test ! -e " + ROOT + " && install -d -m 0755 " + ROOT)
    await environment.upload_file(archive, ROOT + "/runtime.tar.gz")
    await execute(
        "archive-extract", f"tar -xzf {ROOT}/runtime.tar.gz -C {ROOT} && rm {ROOT}/runtime.tar.gz"
    )
    if manifest["learned_requirements"]:
        revision = manifest.get("model_snapshot_revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("runtime archive lacks its pinned model snapshot revision")
        await execute(
            "model-revision-layout",
            f"mv {ROOT}/models/embedding {ROOT}/models/{revision} && "
            f"ln -s {revision} {ROOT}/models/embedding",
        )
    await execute(
        "source-generation",
        f"git -C {ROOT}/source init -q && git -C {ROOT}/source add . && "
        f'git -C {ROOT}/source -c user.name="Supervisor deployment" -c user.email=deployment@localhost '
        f'commit -qm "Exact deployed runtime source" && '
        f"git config --system --add safe.directory {ROOT}/source",
    )
    await execute(
        "system-install",
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends z3 cvc5 bubblewrap ripgrep curl ca-certificates procps sudo",
        timeout=600,
    )
    # Native supervisor contracts use Python 3.12 (including hashable
    # mappingproxy defaults); keep the original task interpreter untouched.
    if type(isolated_uv_bootstrap) is not bool:
        raise ValueError("isolated uv bootstrap selection must be boolean")
    uv_command = "uv"
    uv_install = "python3 -m pip install --no-cache-dir uv==0.9.24"
    if isolated_uv_bootstrap:
        uv_command = ROOT + "/bootstrap/bin/uv"
        uv_install = ("python3 -m venv " + ROOT + "/bootstrap && " + ROOT
                      + "/bootstrap/bin/python -m pip install --no-cache-dir uv==0.9.24")
    await execute(
        "python-runtime-install",
        uv_install + " && UV_PYTHON_INSTALL_DIR=" + PYTHON_INSTALL_DIR + " " + uv_command
        + " python install " + RUNTIME_PYTHON_VERSION,
        timeout=300,
    )
    await execute(
        "python-create",
        "UV_PYTHON_INSTALL_DIR="
        + PYTHON_INSTALL_DIR
        + " " + uv_command + " venv --python "
        + RUNTIME_PYTHON_VERSION
        + " --seed "
        + ROOT
        + "/venv",
    )
    if manifest.get("torch_cpu_requirement") or manifest["learned_requirements"]:
        torch_target = TORCH_CPU_REQUIREMENT
        if torch_binding is not None:
            torch_target = ROOT + "/" + torch_binding["path"]
            await execute("torch-cpu-wheel-verify", PYTHON + " -P -c " + shlex.quote(_torch_cpu_wheel_probe(torch_binding)),
                env=runtime_environment())
        await execute(
            "torch-cpu-install",
            PYTHON
            + " -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "
            + shlex.quote(torch_target),
            timeout=600,
        )
    requirements = manifest["base_requirements"] + [
        item for item in manifest["learned_requirements"] if not item.startswith("torch==")
    ] + manifest.get("security_training_requirements", []) + manifest.get("security_inference_requirements", [])
    requirements += manifest.get("intent_action_384_requirements", [])
    requirements += manifest.get("source384_requirements", [])
    requirements = list(dict.fromkeys(requirements))
    await execute(
        "python-install",
        PYTHON + " -m pip install --no-cache-dir " + shlex.join(requirements),
        timeout=600,
    )
    checker_probe = _source384_header_checker_probe(validate_source384_binding(manifest))
    if checker_probe is not None:
        await execute("source384-header-checker", PYTHON + " -P -c " + shlex.quote(checker_probe),
            env=runtime_environment(), timeout=30)
    await execute(
        "account-create",
        f"useradd --create-home --home-dir {ROOT}/home --shell /bin/bash supervisor && "
        f"install -d -o supervisor -g supervisor -m 0700 {ROOT}/state {ROOT}/codex-auth {ROOT}/home/.codex && "
        f"chown -R supervisor:supervisor /app",
    )
    setup_extensions = """import duckdb,pathlib,shutil
con=duckdb.connect(); platform=con.execute('PRAGMA platform').fetchone()[0]
root=pathlib.Path('/opt/ipfs-supervisor')
target=root/'home/.duckdb/extensions'/('v'+duckdb.__version__)/platform
target.mkdir(parents=True,exist_ok=True)
for name in ('httpfs','quack','ducklake'):
 p=target/(name+'.duckdb_extension');shutil.copyfile(root/'extensions'/p.name,p);con.execute('LOAD '+name)
print(duckdb.__version__,platform)
"""
    await execute(
        "native-extensions",
        PYTHON + " -I -c " + shlex.quote(setup_extensions),
        user="supervisor",
        env=runtime_environment(),
    )
    old_default = environment.default_user
    environment.default_user = "supervisor"
    try:
        if install_codex:
            from harbor.agents.installed.codex import Codex

            codex = Codex(
                logs_dir=output / "codex-install", model_name="gpt-6.1-sol", version=CODEX_VERSION
            )
            await asyncio.wait_for(codex.install(environment), timeout=600)
            await execute("codex-native-expose", "python3 -I -c " + shlex.quote(native_codex_exposure_script()))
            version = (
                await execute(
                    "codex-version",
                    "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; codex --version",
                    user="supervisor",
                    env=runtime_environment(),
                )
            ).stdout.strip()
            if version != "codex-cli " + CODEX_VERSION:
                raise ValueError("native Codex pin differs")
        if provider == "grok_cli":
            await execute("grok-native-expose", "python3 -I -c " + shlex.quote(grok_exposure_script()))
        if auth_json is not None:
            auth_json = Path(auth_json)
            if (
                auth_json.is_symlink()
                or not auth_json.is_file()
                or auth_json.stat().st_size > 65536
            ):
                raise ValueError("one bounded regular auth.json file required")
            # No file contents, hashes, token-bearing environment or host home mount.
            auth_directory = "/grok-auth" if provider == "grok_cli" else "/codex-auth"
            config_directory = "/home/.grok" if provider == "grok_cli" else "/home/.codex"
            await execute("provider-auth-directory", f"install -d -o supervisor -g supervisor -m 0700 {ROOT}{auth_directory} {ROOT}{config_directory}", record=False)
            await environment.upload_file(auth_json, ROOT + auth_directory + "/auth.json")
            await execute(
                "auth-permissions",
                f"chown supervisor:supervisor {ROOT}{auth_directory}/auth.json && chmod 0600 {ROOT}{auth_directory}/auth.json && "
                f"ln -s {ROOT}{auth_directory}/auth.json {ROOT}{config_directory}/auth.json",
                record=False,
            )
            if provider == "codex_cli":
                await execute("auth-status", "codex login status", user="supervisor",
                    env=runtime_environment(), record=False)
    finally:
        environment.default_user = old_default
    imported = json.loads(
        (
            await execute(
                "native-imports",
                PYTHON + " -P -c " + shlex.quote(NATIVE_IMPORT_PROBE),
                user="supervisor",
                env=runtime_environment(),
            )
        ).stdout
    )
    await execute(
        "empty-native-start",
        PYTHON
        + " -P -m ipfs_accelerate_py.agent_supervisor.entrypoints.isolated_benchmark_runtime --directory "
        + ROOT
        + "/state/empty-start --timeout-ms 20000",
        user="supervisor",
        env=runtime_environment(),
    )
    after = json.loads(
        (
            await execute(
                "retained-inputs",
                "python3 -I -c " + shlex.quote(INPUT_SNAPSHOT),
                user="supervisor",
                env=runtime_environment(),
            )
        ).stdout
    )
    if before != after:
        raise ValueError("deployment changed original task source or Git status")
    result = {
        "schema": "terminal-native-supervisor-deployment@1",
        "qualified": True,
        "archive_sha256": manifest["archive_sha256"],
        "original_inputs": before,
        "retained_inputs": after,
        "task_source_preserved": True,
        "runtime_root": ROOT,
        "runtime_environment": runtime_environment(),
        "python": PYTHON,
        "imports": imported,
        "steps": steps,
        "codex_installed": install_codex,
        "provider": provider,
        "grok_installed": provider == "grok_cli",
        "grok_cli_assets": grok_binding if provider == "grok_cli" else None,
        "auth_file_transferred": auth_json is not None,
        "credential_contents_recorded": False,
        "provider_calls": 0,
        "official_verifier_executed": False,
        "benchmark_result": False,
        "seconds": time.monotonic() - started,
    }
    (output / "deployment.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


async def qualify_original_container(
    *,
    task_dir: Path,
    archive_dir: Path,
    output: Path,
    auth_json: Path | None = None,
    install_codex: bool = True,
    keep_container: bool = False,
    resource_profile: str | None = None,
    source384_context: bool = False,
    source384_warm_recovery: str | None = None,
    intent_requirement_contract: Path | None = None,
    setup_cache_selection: dict | None = None,
    provider: str = "codex_cli",
) -> dict:
    from harbor.environments.docker.docker import DockerEnvironment
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths
    from .terminal_source384_qualification import resource_options, observe_resources, qualify_context
    from .benchmark_resource_profile import PROFILES

    options = resource_options(resource_profile)
    from .terminal_source384_warm_recovery import validate_selection
    validate_selection(resource_profile, source384_warm_recovery)
    if source384_warm_recovery is not None and source384_context is not True:
        raise ValueError("warm recovery requires Source384 qualification")
    if type(source384_context) is not bool:
        raise ValueError("Source384 context qualification switch must be boolean")
    manifest = json.loads((Path(archive_dir) / "manifest.json").read_text())
    verify_torch_cpu_wheel_archive(Path(archive_dir) / "runtime.tar.gz", manifest)
    if source384_context and (resource_profile not in PROFILES or validate_source384_binding(manifest) is None):
        raise ValueError("Source384 qualification requires pinned assets and the explicit common profile")

    from .full_supervisor_benchmark import (
        load_intent_requirements_for_instruction, validate_header_planning_selection,
    )
    requirements = load_intent_requirements_for_instruction(intent_requirement_contract,
        Path(task_dir).resolve(strict=True) / "instruction.md")
    binding = validate_source384_binding(manifest)
    if (requirements is not None or (binding and binding["config"].get("header_applicability"))) and not source384_context:
        raise ValueError("reviewed intent qualification requires Source384 context")
    validate_header_planning_selection(binding, requirements, "full")

    from .terminal_setup_cache_advice import (
        validate_setup_cache_selection, validate_setup_cache_prerequisites, apply_setup_cache_advice,
    )
    validate_setup_cache_selection(archive_dir, setup_cache_selection)
    validate_setup_cache_prerequisites(setup_cache_selection, install_codex=install_codex,
        auth_json=auth_json, resource_profile=resource_profile)
    if setup_cache_selection is not None and not source384_context:
        raise ValueError("selected cache qualification requires the complete Source384 context")

    task = Task(task_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    trial = TrialPaths(output / "harbor")
    trial.mkdir()
    environment = DockerEnvironment(
        environment_dir=task.paths.environment_dir,
        environment_name=task.short_name,
        session_id="ipfs-native-deploy-" + str(int(time.time())),
        trial_paths=trial,
        task_env_config=task.config.environment,
        keep_containers=keep_container,
        **options,
    )
    selected = resource_profile is not None or source384_context
    report = dict(schema="terminal-original-container-profile-qualification@1", qualified=False,
        resource_profile=resource_profile, source384_requested=source384_context,
        provider_calls=0, official_verifier_executed=False, benchmark_result=False)
    try:
        await environment.start(force_build=True)
        deployment = await deploy_supervisor(
            environment,
            archive_dir=archive_dir,
            output=output / "deployment",
            auth_json=auth_json,
            install_codex=install_codex,
            provider=provider,
            **({"setup_cache_selection": setup_cache_selection} if setup_cache_selection is not None else {}),
        )
        if setup_cache_selection is not None:
            from .container_worker_deployment import deploy_worker_boundary
            boundary = output / "worker-boundary"
            await deploy_worker_boundary(environment, output=boundary)
            report["setup_cache"] = await apply_setup_cache_advice(environment,
                archive_dir=archive_dir, expected=setup_cache_selection,
                boundary_output=boundary, output=output / "setup-cache", resource_profile=resource_profile)
        if not selected:
            return deployment
        report["deployment"] = deployment
        if resource_profile is not None:
            report["resources_before"] = await observe_resources(environment, output=output, profile=resource_profile)
        if source384_context:
            report["source384"] = await qualify_context(environment, task_dir=task_dir,
                output=output, manifest=manifest, profile=resource_profile,
                **({"source384_warm_recovery": source384_warm_recovery} if source384_warm_recovery is not None else {}),
                **({"intent_requirement_contract": requirements} if requirements is not None else {}))
            signed = report["source384"]["signed_source_hashes"]
            if any(signed.get(name) != row["sha256"] for name, row in deployment["original_inputs"]["files"].items()):
                raise ValueError("Source384 qualification did not consume the complete original container sources")
            report["original_container_sources_consumed"] = True
            final_resources = output / "final-resources"
            final_resources.mkdir()
            report["resources_after"] = await observe_resources(environment, output=final_resources, profile=resource_profile)
        report["qualified"] = True
        return report
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc)[:2048])
        raise
    finally:
        try:
            if selected:
                (output / "qualification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        finally:
            if not keep_container:
                await environment.stop(delete=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("bundle")
    for name in ("output", "source", "datasets", "kit", "extension-dir"):
        build.add_argument("--" + name, type=Path, required=True)
    build.add_argument("--lean-toolchain", type=Path)
    build.add_argument("--grok-binary", type=Path, help="Explicit pinned regular native Grok binary")
    build.add_argument("--model-snapshot", type=Path)
    build.add_argument("--torch-cpu-wheel", type=Path, help="Local pinned CPython 3.12 Torch CPU wheel")
    build.add_argument("--torch-cpu-wheel-sha256", help="Independent SHA-256 of the selected Torch CPU wheel")
    build.add_argument("--source384-config", type=Path, help="Pinned offline shared Source384 parent configuration")
    from .terminal_setup_cache_advice import POLICIES, read_selection
    build.add_argument("--setup-cache-policy", choices=POLICIES,
        help="Explicit pinned setup advice after worker-boundary installation")
    build.add_argument("--security-initializer", type=Path, help="independently admitted host fork descriptor JSON")
    build.add_argument("--canonical-cve-export", type=Path)
    build.add_argument("--canonical-cve-manifest-sha256")
    build.add_argument("--security-checkpoint", type=Path)
    build.add_argument("--security-checkpoint-manifest-sha256")
    build.add_argument("--security-checkpoint-hub-descriptor", type=Path)
    build.add_argument("--security-checkpoint-cache", type=Path)
    build.add_argument("--formula-decoder-descriptor", type=Path,
        help="pinned four-file frozen formula package descriptor; requires frozen security checkpoint")
    build.add_argument("--header-protocol-descriptor", type=Path,
        help="explicit reviewed header protocol; requires formula decoder")
    build.add_argument("--intent-action-384-config", type=Path,
        help="Explicit local shared Intent384 checkpoint and pinned GTE snapshot config")
    build.add_argument("--intent-checkpoint-descriptor", type=Path,
        help="pinned shared Intent structural or semantic roundtrip checkpoint descriptor")
    build.add_argument("--intent-projection-request", type=Path,
        help="inert source-bound projection request for the selected semantic checkpoint")
    deploy = sub.add_parser("qualify")
    for name in ("task-dir", "archive-dir", "output"):
        deploy.add_argument("--" + name, type=Path, required=True)
    deploy.add_argument("--auth-json", type=Path)
    deploy.add_argument("--no-codex", action="store_true")
    deploy.add_argument("--provider", choices=["codex_cli", "grok_cli"], default="codex_cli")
    deploy.add_argument("--keep-container", action="store_true")
    from .benchmark_resource_profile import PROFILES
    deploy.add_argument("--resource-profile", choices=PROFILES,
        help="Explicit common limits; default retains original task resources")
    deploy.add_argument("--setup-cache-selection", type=Path,
        help="Independently retained closed policy and manifest SHA selection")
    deploy.add_argument("--intent-requirement-contract", type=Path,
        help="Reviewed public-instruction mapping for Source384 header qualification")
    deploy.add_argument("--source384-context", action="store_true",
        help="Qualify actual pinned Source384 prepare/initial-context/replay without provider calls")
    deploy.add_argument("--source384-warm-recovery", choices=["source384-warm-admission-recovery@1"],
        help="Explicit bounded validation-only memory-admission recovery during qualification")
    args = parser.parse_args()
    if args.command == "bundle":
        result = build_runtime_archive(
            output=args.output,
            source=args.source,
            datasets=args.datasets,
            kit=args.kit,
            extension_dir=args.extension_dir,
            lean_toolchain=args.lean_toolchain,
            grok_binary=args.grok_binary,
            model_snapshot=args.model_snapshot,
            torch_cpu_wheel=args.torch_cpu_wheel,
            torch_cpu_wheel_sha256=args.torch_cpu_wheel_sha256,
            setup_cache_policy=args.setup_cache_policy,
            security_initializer=(_load_initializer_descriptor(args.security_initializer)
                                  if args.security_initializer else None),
            canonical_cve_export=args.canonical_cve_export,
            canonical_cve_manifest_sha256=args.canonical_cve_manifest_sha256,
            security_checkpoint=args.security_checkpoint,
            security_checkpoint_manifest_sha256=args.security_checkpoint_manifest_sha256,
            security_checkpoint_hub_descriptor=(_load_initializer_descriptor(args.security_checkpoint_hub_descriptor)
                                               if args.security_checkpoint_hub_descriptor else None),
            security_checkpoint_cache=args.security_checkpoint_cache,
            formula_decoder=(_load_initializer_descriptor(args.formula_decoder_descriptor)
                             if args.formula_decoder_descriptor else None),
            header_protocol=(_load_initializer_descriptor(args.header_protocol_descriptor)
                             if args.header_protocol_descriptor else None),
            intent_action_384_config=(load_intent_action_384_config(args.intent_action_384_config)
                                     if args.intent_action_384_config else None),
            source384_config=(load_source384_config(args.source384_config) if args.source384_config else None),
            intent_checkpoint=(_load_initializer_descriptor(args.intent_checkpoint_descriptor)
                               if args.intent_checkpoint_descriptor else None),
            intent_projection_request=(_load_initializer_descriptor(args.intent_projection_request)
                                       if args.intent_projection_request else None),
        )
        print(
            json.dumps({"archive_sha256": result["archive_sha256"], "files": len(result["files"])})
        )
    else:
        result = asyncio.run(
            qualify_original_container(
                task_dir=args.task_dir,
                archive_dir=args.archive_dir,
                output=args.output,
                auth_json=args.auth_json,
                install_codex=not args.no_codex,
                provider=args.provider,
                keep_container=args.keep_container,
                resource_profile=args.resource_profile,
                source384_context=args.source384_context,
                source384_warm_recovery=args.source384_warm_recovery,
                intent_requirement_contract=args.intent_requirement_contract,
                setup_cache_selection=(read_selection(args.setup_cache_selection) if args.setup_cache_selection else None),
            )
        )
        print(
            json.dumps(
                {"qualified": result["qualified"], "provider_calls": result["provider_calls"]}
            )
        )


if __name__ == "__main__":
    main()
