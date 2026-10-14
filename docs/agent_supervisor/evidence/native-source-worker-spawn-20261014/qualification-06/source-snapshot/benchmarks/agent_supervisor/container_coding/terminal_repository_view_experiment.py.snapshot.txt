"""Offline repository-view round trips on reviewed public Terminal Bench files.

The repository boundary is the explicit reviewed COPY fixture tree, never the
container filesystem or a hidden verifier. No benchmark code is executed and no
model, network service, or container is used. Host package storage and synthetic
JSON API transcripts are measured separately.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import shutil
import stat

from . import terminal_profile_catalog as catalog
from .terminal_source_view_experiment import (
    DEFAULT_TASKS, MAX_INPUTS, MAX_TASKS, MAX_TOTAL_BYTES,
    _metrics, _relative, _render, _snapshot, local_token_counter,
)

SCHEMA = "terminal-repository-view-experiment@1"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _save(path: Path, value: dict) -> None:
    path.write_text(_render(value), encoding="utf-8")


def _tree(root: Path) -> dict:
    """Observe only our disposable tree; preserve path, kind, mode and bytes."""
    result = {"": {"kind": "directory", "mode": stat.S_IMODE(root.stat().st_mode)}}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError("the public experiment does not admit symlinks")
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            result[relative] = {"kind": "directory", "mode": mode}
        elif path.is_file():
            raw = path.read_bytes()
            result[relative] = {"kind": "file", "mode": mode, "bytes": len(raw), "sha256": _sha(raw)}
        else:
            raise ValueError("unexpected special file in disposable experiment tree")
    return result


def _read_bytes(response: dict) -> bytes:
    if response["encoding"] == "utf-8":
        return response["content"].encode("utf-8")
    if response["encoding"] == "base64":
        return base64.b64decode(response["content"], validate=True)
    raise ValueError("repository read returned an unsupported content encoding")


def _ast(raw: bytes | str) -> str:
    return ast.dump(ast.parse(raw, type_comments=True), include_attributes=False)


def _storage(package: Path) -> dict:
    files = [path for path in package.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    manifest = (package / "manifest.json").stat().st_size
    return {"package_bytes": total, "manifest_mapping_bytes": manifest,
        "unique_blob_bytes": total - manifest, "package_files": len(files),
        "sent_in_read_protocol": False,
        "scope": "host-side lossless source, compact views and mapping; not model input"}


def _refusals(*, package: Path, digest: str, manifest: dict, output: Path) -> dict:
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_view import (
        RepositoryViewError, load_repository_view, restore_repository_view,
    )
    wrong_digest = "0" * 64 if digest != "0" * 64 else "1" * 64
    errors = {}
    try:
        load_repository_view(package, expected_manifest_sha256=wrong_digest)
    except RepositoryViewError as exc:
        errors["wrong_manifest"] = exc.code
    damaged = output / "damaged-package"
    shutil.copytree(package, damaged)
    row = next(entry for entry in manifest["entries"].values() if entry["kind"] == "file")
    blob = damaged / "blobs" / row["original_sha256"]
    blob.chmod(0o600)
    blob.write_bytes(blob.read_bytes() + b"\nqualification tamper\n")
    destination = output / "refused-restoration"
    try:
        restore_repository_view(damaged, destination, expected_manifest_sha256=digest)
    except RepositoryViewError as exc:
        errors["damaged_blob"] = exc.code
    # Retain the rejection receipt, rather than an extra damaged source package.
    damaged.chmod(0o700)
    for directory in damaged.rglob("*"):
        if directory.is_dir():
            directory.chmod(0o700)
    shutil.rmtree(damaged)
    return {"wrong_manifest_rejected": "wrong_manifest" in errors,
        "damaged_blob_rejected": "damaged_blob" in errors,
        "failed_restore_published_no_destination": not destination.exists(),
        "errors": errors}


def _read_protocol(*, package: Path, digest: str, paths: list[str], representation: str) -> tuple[list[dict], dict]:
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_view import read_repository_file
    messages, responses = [], {}
    for path in paths:
        request = {"operation": "read_repository_file", "expected_manifest_sha256": digest,
            "path": path, "representation": representation}
        response = read_repository_file(package, **{key: value for key, value in request.items() if key != "operation"})
        messages.extend((request, response))
        responses[path] = response
    return messages, responses


def _public_edits(*, package: Path, digest: str, manifest: dict, reads: dict,
                  output: Path, representation: str, counter) -> dict:
    """Append an authored pass to the first module-level function, if present."""
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_revision import (
        RepositoryRevisionError, revise_repository_view,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_view import restore_repository_view
    operations = []
    expected_asts = {}
    for path, response in reads.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(_read_bytes(response), type_comments=True)
        except (SyntaxError, ValueError, UnicodeError):
            continue
        node = next((item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if node is None:
            continue
        node.body.append(ast.Pass())
        replacement = ast.unparse(ast.fix_missing_locations(node)) + "\n"
        operations.append({"operation": "replace_symbol", "path": path,
            "expected_sha256": manifest["entries"][path]["original_sha256"],
            "symbol": node.name, "replacement": replacement})
        expected_asts[path] = ast.dump(tree, include_attributes=False)
    if not operations:
        return {"disposition": "no_module_level_function_for_authored_probe", "edit_count": 0,
            "messages": [], "checks": {}, "final_source_tree": None}
    successor = output / "successor-package"
    request = {"operation": "revise_repository_view", "expected_manifest_sha256": digest, "operations": operations}
    receipt = revise_repository_view(package, successor, expected_manifest_sha256=digest, operations=operations)
    refreshed_digest = receipt["successor_manifest_sha256"]
    restored = output / "successor-original"
    restore_repository_view(successor, restored, expected_manifest_sha256=refreshed_digest)
    after_messages, after = _read_protocol(package=successor, digest=refreshed_digest,
        paths=[row["path"] for row in operations], representation=representation)
    stale_error = None
    try:
        revise_repository_view(successor, output / "refused-stale-revision",
            expected_manifest_sha256=refreshed_digest, operations=[operations[0]])
    except RepositoryRevisionError as exc:
        stale_error = exc.code
    changed = {row["path"] for row in operations}
    checks = {
        "only_authored_ast_pass_added": all(_ast((restored / path).read_bytes()) == expected for path, expected in expected_asts.items()),
        "refresh_delivers_changed_ast": all(_ast(_read_bytes(after[path])) == expected for path, expected in expected_asts.items()),
        "unchanged_files_byte_exact": all(_sha((restored / path).read_bytes()) == row["original_sha256"]
            for path, row in manifest["entries"].items() if row["kind"] == "file" and path not in changed),
        "revision_reports_unchanged_files_preserved": receipt["unchanged_files_preserved"] is True,
        "stale_edit_rejected": stale_error == "stale_source",
        "stale_edit_published_no_package": not (output / "refused-stale-revision").exists(),
        "manifest_refreshed": refreshed_digest != digest,
    }
    messages = [request, receipt, *after_messages]
    return {"disposition": "authored_syntax_probe_only", "edit_count": len(operations),
        "checks": checks, "stale_error": stale_error, "messages": messages,
        "protocol": _metrics(messages, counter), "final_source_tree": _tree(restored),
        "benchmark_source_executed": False, "behavioral_equivalence_claimed": False}


def _task_experiment(*, name: str, sources: dict[str, bytes], output: Path, metadata: dict, counter) -> dict:
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_view import (
        create_repository_view, load_repository_view, restore_repository_view,
    )
    output.mkdir()
    repository = output / "disposable-source"
    _snapshot(repository, sources)
    before_tree = _tree(repository)
    package = output / "package"
    receipt = create_repository_view(repository, package)
    digest = receipt["manifest_sha256"]
    manifest = load_repository_view(package, expected_manifest_sha256=digest)
    package_before = _tree(package)
    # This is the core distinction from a view that depends on canonical files.
    shutil.rmtree(repository)
    source_absent = not repository.exists()
    original, compact = output / "restored-original", output / "restored-compact"
    restore_repository_view(package, original, expected_manifest_sha256=digest, representation="original")
    restore_repository_view(package, compact, expected_manifest_sha256=digest, representation="compact")
    original_tree, compact_tree = _tree(original), _tree(compact)
    files = {path: row for path, row in manifest["entries"].items() if row["kind"] == "file"}
    transformed = sorted(path for path, row in files.items() if row["rendering"] == "python-ast-compact")
    passthrough = sorted(set(files) - set(transformed))
    checks = {
        "original_source_removed_before_restoration": source_absent,
        "all_reviewed_files_packaged": set(files) == set(sources),
        "original_tree_byte_mode_exact": original_tree == before_tree,
        "compact_tree_paths_modes_exact": {path: (row["kind"], row["mode"]) for path, row in compact_tree.items()}
            == {path: (row["kind"], row["mode"]) for path, row in before_tree.items()},
        "compact_materialization_matches_manifest": all(_sha((compact / path).read_bytes()) == row["compact_sha256"] for path, row in files.items()),
        "transformed_python_ast_equal": all(_ast(sources[path]) == _ast((compact / path).read_bytes()) for path in transformed),
        "passthrough_files_byte_exact": all((compact / path).read_bytes() == sources[path] for path in passthrough),
    }
    protocol_results = {}
    for representation in ("original", "compact"):
        arm = output / representation
        arm.mkdir()
        messages, reads = _read_protocol(package=package, digest=digest, paths=sorted(files), representation=representation)
        view_root = original if representation == "original" else compact
        checks[f"{representation}_reads_match_materialized_files"] = all(
            _read_bytes(reads[path]) == (view_root / path).read_bytes() for path in files)
        edits = _public_edits(package=package, digest=digest, manifest=manifest,
            reads=reads, output=arm, representation=representation, counter=counter)
        transcript = {"read_messages": messages, "edit_messages": edits.pop("messages")}
        _save(arm / "protocol.json", transcript)
        protocol_results[representation] = {"read_protocol": _metrics(messages, counter), "edits": edits,
            "read_edit_refresh_protocol": _metrics(messages + transcript["edit_messages"], counter)}
    refusal = _refusals(package=package, digest=digest, manifest=manifest, output=output)
    checks.update({key: value for key, value in refusal.items() if key != "errors"})
    checks["parent_package_unchanged"] = _tree(package) == package_before
    checks["paired_final_canonical_trees_identical"] = (
        protocol_results["original"]["edits"]["final_source_tree"] == protocol_results["compact"]["edits"]["final_source_tree"])
    result = {"task_name": name, **metadata, "coverage": "complete reviewed public COPY fixture tree; not the hidden container filesystem",
        "fixture_mode_policy": "new snapshot uses process-default file/directory modes; does not reproduce image COPY ownership or permissions",
        "input_sha256": {path: _sha(raw) for path, raw in sorted(sources.items())},
        "manifest_sha256": digest, "source_tree_sha256": receipt["source_tree_sha256"],
        "file_count": len(files), "transformed_paths": transformed, "passthrough_paths": passthrough,
        "compact_reads": {"requested_files": len(files), "compacted_python_content_delivered": len(transformed),
            "passthrough_content_delivered": len(passthrough)},
        "original_file_bytes": sum(len(raw) for raw in sources.values()),
        "compact_file_bytes": sum(row["compact_bytes"] for row in files.values()),
        "host_storage": _storage(package), "checks": checks, "refusal_codes": refusal["errors"],
        "protocol": protocol_results, "provider_calls": 0, "containers_started": 0,
        "benchmark_source_executed": False, "official_reward": None, "task_solved": None,
        "qualified": all(checks.values()) and all(all(arm["edits"]["checks"].values()) for arm in protocol_results.values()),
        "disposition": "roundtrip_with_compact_python" if transformed else "roundtrip_passthrough_only"}
    _save(output / "qualification.json", result)
    return result


def _authored_behavior(output: Path) -> dict:
    """Execute only a tiny fixture authored here, never public benchmark source."""
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_view import create_repository_view, restore_repository_view
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_revision import revise_repository_view
    output.mkdir()
    source = output / "source"
    originals = {"fixture.py": b"# Canonical comment remains outside the edit.\ndef increment(value):\n    return value + 1\n\nSENTINEL = 'unchanged'\n",
        "keep.bin": b"\x00\xff\x10", "rename.txt": b"rename me\n", "delete.txt": b"delete me\n"}
    _snapshot(source, originals)
    package = output / "package"
    receipt = create_repository_view(source, package)
    shutil.rmtree(source)
    successor = output / "successor"
    revised = revise_repository_view(package, successor, expected_manifest_sha256=receipt["manifest_sha256"], operations=[
        {"operation": "replace_symbol", "path": "fixture.py", "expected_sha256": _sha(originals["fixture.py"]),
            "symbol": "increment", "replacement": "def increment(value):\n    return value + 2\n"},
        {"operation": "write_file", "path": "added.txt", "expected_sha256": None, "content": "created\n"},
        {"operation": "rename_file", "path": "rename.txt", "destination": "renamed.txt", "expected_sha256": _sha(originals["rename.txt"])},
        {"operation": "delete_file", "path": "delete.txt", "expected_sha256": _sha(originals["delete.txt"])},
    ])
    restored = output / "restored"
    restore_repository_view(successor, restored, expected_manifest_sha256=revised["successor_manifest_sha256"])
    namespace = {}
    exec(compile((restored / "fixture.py").read_bytes(), "authored-fixture", "exec"), namespace)
    checks = {"behavior_change_verified": namespace["increment"](5) == 7,
        "sentinel_preserved": namespace["SENTINEL"] == "unchanged",
        "outside_comment_preserved": (restored / "fixture.py").read_bytes().startswith(b"# Canonical comment remains outside the edit.\n"),
        "binary_unchanged": (restored / "keep.bin").read_bytes() == originals["keep.bin"],
        "addition_materialized": (restored / "added.txt").read_bytes() == b"created\n",
        "rename_materialized": not (restored / "rename.txt").exists() and (restored / "renamed.txt").read_bytes() == originals["rename.txt"],
        "deletion_materialized": not (restored / "delete.txt").exists(),
        "original_source_absent": not source.exists()}
    result = {"checks": checks, "qualified": all(checks.values()), "operations": 4,
        "benchmark_source_executed": False, "executed_source": "only the inline authored increment fixture",
        "provider_calls": 0, "revision": revised}
    _save(output / "qualification.json", result)
    return result


def run_experiment(*, dataset: Path, output: Path, tasks=DEFAULT_TASKS, tokenizer_cache: Path | None = None) -> dict:
    """Validate reviewed inputs before output creation; work exclusively on copies."""
    tasks = tuple(tasks)
    if not tasks or len(tasks) > MAX_TASKS or len(set(tasks)) != len(tasks):
        raise ValueError("tasks must be a bounded, nonempty unique selection")
    dataset = Path(dataset).resolve(strict=True)
    loaded, total = [], 0
    for name in tasks:
        profile = catalog.reviewed_profile(dataset=dataset, task_name=name)
        rows = profile["bindings"]["inputs"]
        if not rows or len(rows) > MAX_INPUTS:
            raise ValueError("public profile input count must be bounded and nonempty")
        sources = {}
        for row in rows:
            _relative(row["path"])
            raw = catalog._read(dataset / name / "environment" / row["environment_path"], row["sha256"])
            total += len(raw)
            if total > MAX_TOTAL_BYTES:
                raise ValueError("public profile aggregate bytes exceed experiment bound")
            if row["path"] in sources:
                raise ValueError("duplicate public input path")
            sources[row["path"]] = raw
        loaded.append((name, profile, sources))
    counter, tokenizer = local_token_counter(tokenizer_cache)
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("experiment output must be a new canonical directory")
    if output == dataset or dataset in output.parents:
        raise ValueError("experiment output must be outside the source dataset")
    output.mkdir(parents=True)
    rows = [_task_experiment(name=name, sources=sources, output=output / name, counter=counter,
        metadata={"dataset_revision_reviewed": profile["dataset_revision_reviewed"],
            "public_metadata_sha256": profile["bindings"]["public_metadata"],
            "source_boundary": profile["source_boundary"], "hidden_verifier_or_solution_read": False})
        for name, profile, sources in loaded]
    authored = _authored_behavior(output / "authored-behavior")
    unchanged = all(catalog._read(dataset / name / "environment" / row["environment_path"], row["sha256"]) == sources[row["path"]]
        for name, profile, sources in loaded for row in profile["bindings"]["inputs"])
    result = {"schema": SCHEMA, "tasks": rows, "authored_behavior": authored,
        "qualified": unchanged and authored["qualified"] and all(row["qualified"] for row in rows),
        "original_dataset_unchanged": unchanged, "local_tokenizer": tokenizer,
        "provider_calls": 0, "containers_started": 0, "provider_usage_tokens": None, "official_reward": None,
        "measurement_scope": "same complete reviewed file selection; synthetic sorted compact JSON API requests and complete responses; filesystem package routing is host-side",
        "storage_scope": "host package originals, compact blobs and manifest mapping counted separately; full package not sent to a model",
        "limitations": [
            "No live model solving, provider usage, task success rate, hidden verifier, or official reward is measured.",
            "Coverage is the reviewed public COPY fixture tree, not a live container or arbitrary whole repository inventory.",
            "Original restoration requires the complete self-contained package, including retained originals; compact text alone is insufficient.",
            "AST agreement and compilation are syntax/structure checks, not proofs of behavioral equivalence or reflection/debugging fidelity.",
            "Public edit probes append an authored pass statement to the first module-level function; public source is never executed.",
            "Model reasoning, instructions, shell/provider framing, retries, planning and subsequent raw expansion are outside transcript totals.",
            "Deterministic local API calls demonstrate actual compact reads; they do not establish native agent tool adoption.",
            "DuckDB selection, semantic capsule admission, proof solving and context history are not varied.",
        ]}
    _save(output / "qualification.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--tokenizer-cache", type=Path)
    args = parser.parse_args(argv)
    result = run_experiment(dataset=args.dataset, output=args.output, tasks=args.tasks or DEFAULT_TASKS,
        tokenizer_cache=args.tokenizer_cache)
    print(_render({"schema": SCHEMA, "qualified": result["qualified"], "provider_calls": 0,
        "tasks": [row["task_name"] for row in result["tasks"]], "output": str(args.output)}), end="")
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
