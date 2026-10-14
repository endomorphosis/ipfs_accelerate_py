#!/usr/bin/env python3
"""Read-only, metadata-only preparatory capability inventory for LAC-S2-006.

Materialization writes one task-owned immutable result. Validation replays
source identities without rerunning volatile probes. Successful inventory
validation grants no runtime, semantic, model, or corpus admission.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "legal-autoformalization-foundations-capabilities/v1"
SOURCE_SCHEMA = "legal-autoformalization-foundations-sources/v1"
TASK = "LAC-S2-006"
RELATIVE_OUTPUT = "data/legal_autoformalization_foundations/LAC-S2-006/result.json"
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_BYTES = 65536
OPTIMIZER = "ipfs_datasets_py/optimizers/logic_theorem_optimizer/"
AUTOENCODER = "ipfs_datasets_py/logic/formalization/autoencoder/"
METADATA_SOURCES = {
    "registry": OPTIMIZER + "autoencoder_runtime_registry.py",
    "raw384": OPTIMIZER + "autoencoder_embedding_runtime.py",
    "stages384": AUTOENCODER + "legal_native_conditioning.py",
    "native768": AUTOENCODER + "gte_multilingual_profile.py",
    "producer768": AUTOENCODER + "source_embeddings_768.py",
    "complete768": AUTOENCODER + "source_embeddings_768_complete.py",
    "native4096": AUTOENCODER + "source_embeddings_4096.py",
    "retained_raw384": AUTOENCODER + "arbitrary_native_span_runtime.py",
}
SOLVER_FLAGS = {"z3": "--version", "cvc5": "--version", "vampire": "--version",
                "eprover": "--version"}
VERSION_PREFIXES = {"lean": r"Lean (?:\(version|version)", "lake": r"Lake version",
    "z3": r"Z3 version", "cvc5": r"(?:This is )?cvc5 version",
    "vampire": r"Vampire", "eprover": r"(?:E |E Prover)"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def exact_path(value, *, root=None):
    path = Path(value)
    if not path.is_absolute():
        need(root is not None and ".." not in path.parts, "relative input needs a bound source root")
        path = root / path
    need(".." not in path.parts, "path traversal refused")
    for part in (path, *path.parents):
        need(not part.is_symlink(), "immutable input path aliases refused")
    need(path.resolve(strict=True) == path, "canonical input path required")
    return path


def read_file(path):
    path = exact_path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_FILE_BYTES,
             "bounded regular metadata file required")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        after = os.fstat(descriptor)
        identity = lambda entry: (entry.st_dev, entry.st_ino, entry.st_size,
                                  entry.st_mtime_ns, entry.st_ctime_ns)
        need(identity(before) == identity(after) == identity(path.stat()), "input changed while reading")
        need(len(raw) <= MAX_FILE_BYTES, "metadata byte cap exceeded")
        return raw, {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    finally:
        os.close(descriptor)


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def bounded_command(argv, *, cwd, timeout, environment):
    """Bound output on disk, avoiding unbounded capture_output memory."""
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(argv, cwd=cwd, env=environment,
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True)
        deadline = time.monotonic() + timeout
        state = "completed"
        while process.poll() is None:
            if output.tell() > MAX_OUTPUT_BYTES:
                state = "output_limit"
                break
            if time.monotonic() >= deadline:
                state = "timeout"
                break
            time.sleep(0.02)
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)
        output.seek(0)
        raw = output.read(MAX_OUTPUT_BYTES + 1)
    if len(raw) > MAX_OUTPUT_BYTES:
        state = "output_limit"
        raw = raw[:MAX_OUTPUT_BYTES]
    return {"returncode": process.returncode, "state": state,
            "output_sha256": hashlib.sha256(raw).hexdigest(),
            "output_bytes": len(raw), "text": raw.decode(errors="replace")}


def probe_environment(directory, executable):
    return {"PATH": str(Path(executable).parent) + ":/usr/bin:/bin",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "HOME": str(directory),
        "XDG_CACHE_HOME": str(directory), "ELAN_HOME": str(directory / "no-elan"),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def git(root, *arguments, raw=False):
    command = bounded_command(["git", "-C", str(root), *arguments], cwd=root,
        timeout=10, environment={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
                                  "GIT_OPTIONAL_LOCKS": "0"})
    need(command["state"] == "completed" and command["returncode"] == 0,
         "read-only repository identity command failed")
    return command["text"] if raw else command["text"].strip()


def constants(raw):
    """Read only a restricted literal AST; never import or execute source."""
    result = {}

    def evaluate(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return result[node.id]
        if isinstance(node, ast.Tuple):
            return tuple(evaluate(value) for value in node.elts)
        if isinstance(node, ast.List):
            return [evaluate(value) for value in node.elts]
        if isinstance(node, ast.Dict):
            return {evaluate(key): evaluate(value) for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add)):
            left, right = evaluate(node.left), evaluate(node.right)
            return left * right if isinstance(node.op, ast.Mult) else left + right
        if isinstance(node, ast.JoinedStr):
            return "".join(str(evaluate(value.value)) if isinstance(value, ast.FormattedValue)
                           else evaluate(value) for value in node.values)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args:
            return {value.arg: evaluate(value.value) for value in node.keywords if value.arg}
        raise ValueError("nonliteral declaration")

    for statement in ast.parse(raw).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            try:
                result[statement.targets[0].id] = evaluate(statement.value)
            except (ValueError, KeyError, TypeError):
                pass
    return result


def profiles(metadata):
    registry, raw, stages = (metadata[key] for key in ("registry", "raw384", "stages384"))
    native768, native4096 = metadata["native768"], metadata["native4096"]
    need(list(stages.get("STAGES", ())) == ["raw384", "core384", "trained384"], "native stage identities differ")
    need(raw.get("PINNED_REVISION") and native768.get("MODEL_REV") and native4096.get("PROFILE_ID"),
         "native producer identities missing")
    entries = [{"profile_id": "legal_ir:" + registry["LEARNED_FORMULA_VERSION"],
        "runtime_id": "legal_ir:" + registry["LEARNED_FORMULA_VERSION"],
        "representation": "source_text", "dimension": None,
        "dimension_role": "no_vector_input", "tokenizer": {"binding": "checkpoint_source_vocabulary_unbound"},
        "projection": {"kind": "source_encoder_from_exact_checkpoint_unbound"}}]
    kinds = {"raw384": "native_gte_source_embedding", "core384": "source_parser_sparse_core_projection",
             "trained384": "saved_learned_residual_projection"}
    for stage in stages["STAGES"]:
        entries.append({"profile_id": stages["RECEIPT_SCHEMA"] + ":" + stage,
            "runtime_id": "legal_ir:current_v2", "representation": kinds[stage],
            "stage": stage, "dimension": 384, "dimension_role": "input_embedding",
            "model": {"id": "thenlper/gte-small", "revision": raw["PINNED_REVISION"]},
            "tokenizer": {"max_tokens": raw["MAX_TOKENS"],
                "declared_assets": {name: {"bytes": value[0], "sha256": value[1]}
                    for name, value in raw["_PINNED_ASSETS"].items() if "token" in name or name == "vocab.txt"}},
            "projection": {"kind": kinds[stage],
                "checkpoint_binding": "not_required" if stage == "raw384" else "exact_package_receipt_unbound"},
            "upstream_stage": {"raw384": None, "core384": "raw384", "trained384": "core384"}[stage]})
    entries.append({"profile_id": native768["PROFILE_ID"], "runtime_id": "source_embeddings_768_complete",
        "representation": "native_multilingual_gte_cls_l2", "dimension": 768,
        "dimension_role": "input_embedding", "model": {"id": native768["MODEL_REPO"], "revision": native768["MODEL_REV"],
            "code_revision": native768["CODE_REV"]},
        "tokenizer": {"max_tokens": native768["PROFILE"]["max_tokens"],
            "sha256": native768["PUBLISHED_ASSETS"][("model", "tokenizer.json")]["sha256"]},
        "projection": {"kind": "native_cls_l2_no_384_projection"}})
    entries.append({"profile_id": native4096["PROFILE_ID"], "runtime_id": native4096["PRODUCER_ID"],
        "representation": "leanstral_source4096", "dimension": native4096["DIMENSION"],
        "dimension_role": "input_embedding", "model": {"sha256": native4096["MODEL_SHA256"], "cid": native4096["MODEL_CID"]},
        "tokenizer": {"max_tokens": native4096["MAX_TOKENS"], "binding": "operation_bound_receipt_unbound"},
        "projection": {"kind": "last_l2_native4096"}})
    for entry in entries:
        entry.update(evidence_kind="source_declaration_only", model_loaded=False,
            tokenizer_executed=False, projection_executed=False, width_implies_compatibility=False,
            runtime_qualified=False)
        entry["identity_sha256"] = digest(entry)
    return entries


def capture_immutable(sources_path):
    source_raw, source_pin = read_file(sources_path)
    source = decode(source_raw)
    need(source.get("schema") == SOURCE_SCHEMA, "foundations source schema differs")
    datasets = source["datasets"]
    root = exact_path(datasets["root"])
    need(root.is_dir() and re.fullmatch(r"[0-9a-f]{40}", datasets["commit"]), "exact datasets repository commit required")
    need(git(root, "rev-parse", "HEAD") == datasets["commit"], "datasets HEAD differs from source pin")
    if datasets.get("expected_branch"):
        need(git(root, "branch", "--show-current") == datasets["expected_branch"], "datasets branch differs")
    inputs = []
    metadata = {}
    for role, relative in METADATA_SOURCES.items():
        path = exact_path(relative, root=root)
        raw, pin = read_file(path)
        committed = git(root, "show", datasets["commit"] + ":" + relative, raw=True).encode()
        need(raw == committed, "runtime metadata differs from pinned commit")
        pin.update(role=role, repo_relative_path=relative)
        inputs.append(pin)
        metadata[role] = constants(raw)
    declared = source.get("immutable_inputs", [])
    need(isinstance(declared, list) and len(declared) <= 128, "bounded immutable input list required")
    for spec in [*declared,
                 dict(source["prior_bootstrap_completion"], role="prior_bootstrap_completion"),
                 dict(source["original_plan"], role="original_plan")]:
        need(re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256", "")), "immutable input SHA256 required")
        try:
            path = exact_path(spec["path"], root=root)
            _, pin = read_file(path)
        except FileNotFoundError:
            if spec.get("required", True):
                raise
            inputs.append({"path": spec["path"], "role": spec.get("role", "metadata"), "status": "missing_optional", "sha256": spec["sha256"]})
            continue
        need(pin["sha256"] == spec["sha256"], "immutable metadata SHA256 differs")
        if "bytes" in spec:
            need(pin["bytes"] == spec["bytes"], "immutable metadata byte count differs")
        pin.update(role=spec.get("role", "metadata"), evidence_kind="declared_or_prior_receipt_not_live_probe")
        inputs.append(pin)
    _, script_pin = read_file(Path(__file__).resolve())
    if source.get("capabilities_source_sha256"):
        need(script_pin["sha256"] == source["capabilities_source_sha256"], "capability driver differs from launch pin")
    script_pin.pop("path")
    script_pin.update(repo_relative_path="scripts/legal_autoformalization_capabilities.py")
    source_pin.pop("path")
    source_pin.update(repo_relative_path=str(Path(sources_path).resolve().relative_to(ROOT)))
    core = {"source_manifest": source_pin, "driver": script_pin,
        "datasets": {"root": str(root), "commit": datasets["commit"], "branch": datasets.get("expected_branch")},
        "input_artifacts": inputs, "runtime_profiles": profiles(metadata),
        "registered_legal_versions": list(metadata["registry"]["LEGAL_VERSIONS"]),
        "scope": "listed metadata only; no transitive closure or checkpoint payload verified"}
    core["input_digest_sha256"] = digest(core)
    return source, core


def probe_tool(name, source, *, which=shutil.which, runner=bounded_command):
    selected = source.get("tool_paths", {}).get(name) or which(name)
    result = {"name": name, "evidence_kind": "actual_local_subprocess_probe", "available": bool(selected),
              "version_verified": False, "blocker": None}
    if not selected:
        result["blocker"] = name + "_executable_missing"
        return result
    try:
        path = Path(selected).resolve(strict=True)
    except (OSError, RuntimeError):
        result.update(available=False, blocker=name + "_executable_missing")
        return result
    result.update(executable_path=str(path))
    if (path.parent.name == "bin" and path.parent.parent.name == ".elan") or path.name == "elan":
        result["blocker"] = name + "_elan_shim_requires_explicit_cached_toolchain"
        return result
    if not path.is_file() or not os.access(path, os.X_OK):
        result["blocker"] = name + "_not_executable"
        return result
    # Executable identity hashing is separate from checkpoint/data payloads.
    checksum = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    result["executable_sha256"] = checksum.hexdigest()
    try:
        with tempfile.TemporaryDirectory(prefix="lac-capability-probe-") as temporary:
            directory = Path(temporary)
            command = runner([str(path), "--version"], cwd=directory,
                timeout=min(10, source.get("probe_timeout_seconds", 5)),
                environment=probe_environment(directory, path))
    except (OSError, subprocess.SubprocessError) as error:
        result.update(blocker=name + "_version_probe_failed", error_type=type(error).__name__)
        return result
    result.update({key: command[key] for key in ("returncode", "state", "output_sha256", "output_bytes")})
    need((path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns), "tool executable changed during probe")
    text = command["text"].strip()
    # Only a conventional bounded version line enters the manifest; arbitrary
    # stderr and environment values are never serialized.
    version_line = next((line for line in text.splitlines()
                         if re.fullmatch(r"[A-Za-z0-9 .,:+()_/@=-]{1,256}", line)
                         and re.match(VERSION_PREFIXES[name], line, re.IGNORECASE)
                         and re.search(r"\d+\.\d+", line)), None)
    result["version_verified"] = command["state"] == "completed" and command["returncode"] == 0 and version_line is not None
    if result["version_verified"]:
        result["version"] = version_line
    else:
        result["blocker"] = name + "_version_probe_failed"
    return result


def probe_docker(source, *, which=shutil.which, runner=bounded_command):
    request = source.get("docker", {})
    images = request.get("images", [])
    need(len(images) <= 8, "bounded Docker image list required")
    result = {"evidence_kind": "actual_local_image_inspect_only", "runtime_available": False,
              "images": [], "blocker": None}
    selected = which("docker")
    if not selected:
        result["blocker"] = "docker_executable_missing"
        return result
    try:
        path = Path(selected).resolve(strict=True)
    except OSError:
        result["blocker"] = "docker_executable_missing"
        return result
    information = path.stat()
    if path not in (Path("/usr/bin/docker"), Path("/usr/local/bin/docker")) or information.st_uid != 0 or information.st_mode & 0o022:
        result["blocker"] = "docker_executable_untrusted"
        return result
    result["runtime_available"] = True
    need(request.get("socket", "unix:///var/run/docker.sock") == "unix:///var/run/docker.sock", "only trusted local Docker socket supported")
    for image in images:
        reference = image["reference"]
        need(isinstance(reference, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:@-]{0,255}", reference), "invalid local image reference")
        with tempfile.TemporaryDirectory(prefix="lac-docker-inspect-") as temporary:
            directory = Path(temporary)
            try:
                command = runner([str(path), "--host=unix:///var/run/docker.sock", "--config", temporary,
                    "image", "inspect", "--format", "{{json .}}", reference], cwd=directory,
                    timeout=min(10, source.get("probe_timeout_seconds", 5)),
                    environment=probe_environment(directory, path))
            except (OSError, subprocess.SubprocessError):
                command = {"returncode": None, "state": "probe_error", "text": ""}
        row = {"reference": reference, "present": False, "identity_matches": False,
               "returncode": command["returncode"], "state": command["state"]}
        if command["returncode"] == 0 and command["state"] == "completed":
            try:
                observed = decode(command["text"])
                row.update(present=True, image_id=observed["Id"], os=observed["Os"], architecture=observed["Architecture"])
                labels = observed.get("Config", {}).get("Labels") or {}
                row["identity_matches"] = all((not image.get(field) or image[field] == row[key])
                    for field, key in (("expected_id", "image_id"), ("os", "os"), ("architecture", "architecture"))) and all(
                    labels.get(key) == value for key, value in image.get("labels", {}).items())
            except (ValueError, TypeError, KeyError, AttributeError):
                row["blocker"] = "docker_image_inspect_invalid"
        if not row["present"]:
            row["blocker"] = "pinned_docker_image_missing_or_unobservable"
        elif not row["identity_matches"]:
            row["blocker"] = "pinned_docker_image_identity_mismatch"
        result["images"].append(row)
    return result


def collect(sources_path):
    source, core = capture_immutable(sources_path)
    need(type(source.get("probe_timeout_seconds", 5)) is int
         and 1 <= source.get("probe_timeout_seconds", 5) <= 10,
         "probe timeout must be an exact integer in 1..10 seconds")
    required = source.get("required_tools", {"lean": True, "lake": True})
    solvers = required.get("solvers", [])
    need(isinstance(solvers, list) and set(solvers) <= set(SOLVER_FLAGS), "unsupported solver version probe")
    names = ["lean", "lake", *sorted(set(solvers))]
    observed = time.time_ns()
    ttl = source.get("observation_ttl_seconds", 900)
    need(type(ttl) is int and 1 <= ttl <= 3600, "observation TTL must be 1..3600 seconds")
    result = {"schema": SCHEMA, "campaign_id": source["campaign_id"], "task_id": TASK,
        "original_task_acceptance": "preparatory_metadata_inventory_only",
        "immutable": core, "observations": {"observed_at_epoch_nanoseconds": observed,
            "expires_at_epoch_nanoseconds": observed + ttl * 1_000_000_000,
            "tools": [probe_tool(name, source) for name in names], "docker": probe_docker(source)},
        "authority": {"legal_semantic_acceptance": False, "runtime_qualified": False,
            "model_loaded": False, "checkpoint_loaded": False, "training_executed": False,
            "provider_invoked": False, "network_requested": False,
            "docker_mutations": False, "services_started": False}}
    result["capability_gates"] = capability_gates(result, source, observed)
    result["manifest_sha256"] = digest(result)
    # Recheck immutable inputs after all probes; never fold changed source
    # bytes into the same capability observation generation.
    need(capture_immutable(sources_path)[1] == core, "immutable sources moved during probes")
    return result


def capability_gates(result, source, now_ns):
    observations = result["observations"]
    blockers = []
    if not observations["observed_at_epoch_nanoseconds"] <= now_ns <= observations["expires_at_epoch_nanoseconds"]:
        blockers.append("live_capability_observations_expired_or_future")
    required = source.get("required_tools", {"lean": True, "lake": True})
    for tool in observations["tools"]:
        name = tool["name"]
        # A diagnostic label never supplies positive capability evidence.
        # Removing it cannot turn a missing or failed probe into availability.
        if tool.get("available") is not True:
            blocker = name + "_executable_missing"
        elif (tool.get("version_verified") is not True
                or tool.get("state") != "completed" or tool.get("returncode") != 0):
            allowed = {name + suffix for suffix in ("_version_probe_failed", "_not_executable",
                       "_elan_shim_requires_explicit_cached_toolchain")}
            blocker = tool.get("blocker") if tool.get("blocker") in allowed else name + "_version_probe_failed"
        else:
            blocker = None
        if blocker and (required.get(name, False) or name in required.get("solvers", [])):
            blockers.append(blocker)
    docker = observations["docker"]
    if source.get("docker", {}).get("required", True):
        if docker.get("runtime_available") is not True:
            blockers.append("docker_executable_untrusted" if docker.get("blocker") == "docker_executable_untrusted"
                            else "docker_executable_missing")
        for image in docker["images"]:
            if image.get("present") is not True:
                blockers.append("pinned_docker_image_missing_or_unobservable:" + image["reference"])
            elif (image.get("identity_matches") is not True
                  or image.get("state") != "completed" or image.get("returncode") != 0):
                blockers.append("pinned_docker_image_identity_mismatch:" + image["reference"])
    # Local metadata cannot authenticate a loaded projection or translator.
    return {"status": "blocked" if blockers else "observed_tools_available_only",
        "blockers": blockers, "evaluated_at_epoch_nanoseconds": now_ns,
        "runtime_semantic_admission": False, "checkpoint_projection_bindings_complete": False}


def validate_manifest(result, sources_path, *, now_ns=None):
    need(isinstance(result, dict) and result.get("schema") == SCHEMA and result.get("task_id") == TASK,
         "capability manifest schema or task differs")
    body = dict(result)
    need(body.pop("manifest_sha256", None) == digest(body), "capability manifest integrity differs")
    source, core = capture_immutable(sources_path)
    need(result["immutable"] == core and result["campaign_id"] == source["campaign_id"], "capability immutable replay differs")
    observations = result["observations"]
    need(type(observations.get("observed_at_epoch_nanoseconds")) is int
         and type(observations.get("expires_at_epoch_nanoseconds")) is int
         and observations["expires_at_epoch_nanoseconds"] - observations["observed_at_epoch_nanoseconds"]
             == source.get("observation_ttl_seconds", 900) * 1_000_000_000,
         "capability observation lifetime differs")
    expected_tools = ["lean", "lake", *sorted(set(source.get("required_tools", {}).get("solvers", [])))]
    need([row.get("name") for row in observations["tools"]] == expected_tools, "capability tool population differs")
    need(all(row.get("evidence_kind") == "actual_local_subprocess_probe"
             and type(row.get("available")) is bool
             and type(row.get("version_verified")) is bool for row in observations["tools"]),
         "capability evidence kind differs")
    for row in observations["tools"]:
        if row.get("version_verified"):
            need(row.get("available") is True and row.get("state") == "completed"
                 and row.get("returncode") == 0 and isinstance(row.get("version"), str),
                 "successful tool probe has no version evidence")
    need(type(observations["docker"].get("runtime_available")) is bool,
         "Docker runtime availability evidence missing")
    for row in observations["docker"]["images"]:
        need(type(row.get("present")) is bool and type(row.get("identity_matches")) is bool,
             "Docker image availability evidence missing")
        if row.get("identity_matches"):
            need(row.get("present") is True and row.get("state") == "completed"
                 and row.get("returncode") == 0, "Docker image match has no successful inspect evidence")
    need([row.get("reference") for row in observations["docker"]["images"]]
             == [row["reference"] for row in source.get("docker", {}).get("images", [])]
             or observations["docker"].get("runtime_available") is False,
         "Docker capability population differs")
    need(all(value is False for value in result["authority"].values()), "capability inventory cannot confer authority")
    current = capability_gates(result, source, time.time_ns() if now_ns is None else now_ns)
    return {"valid": True, "task_id": TASK, "status": "validated_preparatory_capability_inventory",
        "input_digest_sha256": core["input_digest_sha256"], "capability_gates": current,
        "live_probes_reexecuted": False, "original_task_acceptance": "partial"}


def output_path(requested=None):
    expected = ROOT / RELATIVE_OUTPUT
    candidate = Path(requested) if requested else expected
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    need(candidate == expected, "output must be the exact LAC-S2-006 task-owned result path")
    for part in (candidate, *candidate.parents):
        need(not part.is_symlink(), "output aliases refused")
    return candidate


def materialize(sources_path, output=None):
    path = output_path(output)
    if path.exists():
        raw, pin = read_file(path)
        return {**validate_manifest(decode(raw), sources_path), "materialized": False,
                "output": str(path), "sha256": pin["sha256"]}
    result = collect(sources_path)
    checked = validate_manifest(result, sources_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, prefix=".capabilities-") as stream:
        temporary = Path(stream.name)
        stream.write(canonical(result) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError:
        return materialize(sources_path, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {**checked, "materialized": True, "output": str(path), "sha256": read_file(path)[1]["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("materialize", "validate"))
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    sources_path = Path(args.sources)
    if not sources_path.is_absolute():
        sources_path = ROOT / sources_path
    if args.command == "materialize":
        answer = materialize(sources_path, args.output)
    else:
        path = output_path(args.manifest)
        answer = validate_manifest(decode(read_file(path)[0]), sources_path)
    print(json.dumps(answer, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"valid": False, "task_id": TASK, "error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(2)
