"""Qualify indexes on explicitly permitted files from an actual task image.

The image must contain public task inputs only. This does not mount the task
definition, verifier, solution, credentials or host workspace in the container.
It performs no coding/model trial and never invokes the official verifier.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess


READ_INPUTS = """import base64, hashlib, json, pathlib, sys
root = pathlib.Path('/app').resolve(strict=True)
result = {}
for name in json.loads(sys.argv[1]):
    relative = pathlib.PurePosixPath(name)
    path = root / name
    if relative.is_absolute() or '..' in relative.parts or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('permitted input escapes /app')
    with path.open('rb') as stream:
        raw = stream.read(2000001)
    if len(raw) > 2000000:
        raise ValueError('source byte bound exceeded')
    result[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'base64': base64.b64encode(raw).decode('ascii')}
print(json.dumps(result))
"""


def qualify(image: str, output: Path, paths: list[str], query: str) -> dict:
    from benchmarks.agent_supervisor.container_coding.vector_index_preflight import qualify as vectors

    if not 1 <= len(paths) <= 64 or len(set(paths)) != len(paths):
        raise ValueError("one to 64 unique public source paths required")
    for name in paths:
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name
                or not name.endswith(".py") or relative.parts[0] in {".git", ".runtime"}):
            raise ValueError("only normalized relative Python source paths are accepted")
    output = output.absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("a new output directory without symlinks is required")
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True).strip()
    raw = subprocess.check_output([
        "docker", "run", "--rm", "--network", "none", "--read-only", "--cpus", "1",
        "--memory", "256m", "--pids-limit", "64", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--entrypoint", "python3", image_id,
        "-I", "-c", READ_INPUTS, json.dumps(paths),
    ], timeout=60)
    if len(raw) > 180_000_000:
        raise ValueError("container input receipt exceeds bound")
    inputs = json.loads(raw)
    if set(inputs) != set(paths):
        raise ValueError("container did not return the exact permitted input set")
    output.mkdir(parents=True)
    repository = output / "permitted-inputs"
    repository.mkdir()
    hashes = {}
    for name, row in inputs.items():
        content = base64.b64decode(row["base64"], validate=True)
        digest = hashlib.sha256(content).hexdigest()
        if digest != row["sha256"] or len(content) > 2_000_000:
            raise ValueError("container source receipt differs from delivered bytes")
        destination = repository / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        hashes[name] = digest
    indexed = vectors(repository, output / "vectors", paths, query)
    if indexed["source_sha256"] != hashes:
        raise ValueError("index inputs differ from the actual container inputs")
    result = {
        "schema": "terminal-container-input-index-qualification@1", "status": "qualified",
        "image_id": image_id, "container_root": "/app", "source_sha256": hashes,
        "complete_permitted_scope": indexed["complete_permitted_scope"],
        "symbols": indexed["symbols"], "index_id": indexed["index_id"],
        "ducklake": indexed["ducklake"], "index_seconds": indexed["seconds"],
        "embedding_mode": "lexical", "model_calls": 0, "official_verifier_executed": False,
        "full_indexed_arm": False, "terminal_bench_result": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file", action="append", required=True, dest="paths")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(args.image, args.output, args.paths, args.query), indent=2))
