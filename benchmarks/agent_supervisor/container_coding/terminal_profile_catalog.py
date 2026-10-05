"""Pinned public task profiles and model-free native preparation qualification.

Only reviewed instruction, Dockerfile, task metadata and explicit public COPY
inputs are opened. This module never opens a task's verifier or solution tree.
Qualification is not a container run, a task solution, or an official reward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import tomllib

from .terminal_task_profile import (DATA_SCHEMA, task_profile_bytes,
    instruction_sha256, validate_task_profile, validate_task_data)

CATALOG = Path(__file__).with_name("task_profiles") / "terminal_bench_2_public.json"


def _read(path: Path, expected: str, *, maximum: int = 262144) -> bytes:
    if path.is_symlink() or path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("reviewed public input must be a canonical regular file")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("reviewed public input changed or exceeds its byte bound")
    return raw


def reviewed_profile(*, dataset: Path, task_name: str) -> dict:
    catalog = json.loads(CATALOG.read_text())
    if task_name not in catalog["tasks"]:
        raise ValueError("task has no reviewed public profile")
    dataset = Path(dataset).resolve(strict=True)
    row = catalog["tasks"][task_name]
    task = dataset / task_name
    public = {name: _read(task / name, digest) for name, digest in row["public_metadata"].items()}
    data = []
    for item in row["inputs"]:
        raw = _read(task / "environment" / item["environment_path"], item["sha256"])
        if item["media_type"] is not None:
            validate_task_data(raw, item["media_type"])
            data.append({key: item[key] for key in ("path", "media_type")})
    profile = validate_task_profile(dict(schema=DATA_SCHEMA,
        instruction_sha256=instruction_sha256(public["instruction.md"].decode("utf-8")),
        input_paths=[item["path"] for item in row["inputs"]],
        data_inputs=data, outputs=row["outputs"]))
    return dict(profile=profile, bindings=row,
        dataset_revision_reviewed=catalog["dataset_commit"],
        native_agent_seconds=tomllib.loads(public["task.toml"].decode())["agent"]["timeout_sec"],
        source_boundary="reviewed public instruction, metadata and explicit environment COPY inputs only",
        hidden_verifier_or_solution_read=False, official_reward=None,
        source384_inference_qualified=False, full_supervisor_execution_qualified=False)


def qualify_public_profile(*, dataset: Path, task_name: str, output: Path) -> dict:
    """Exercise real signed prepare and index hydration on exact public bytes."""
    selected = reviewed_profile(dataset=dataset, task_name=task_name)
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("qualification output must be a new canonical directory")
    output.mkdir(parents=True)
    root = output / "app"
    root.mkdir()
    task = Path(dataset).resolve(strict=True) / task_name
    for item in selected["bindings"]["inputs"]:
        raw = _read(task / "environment" / item["environment_path"], item["sha256"])
        destination = root / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    def git(*args):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    git("init", "-q")
    git("add", "--all")
    git("-c", "user.name=Public profile qualification", "-c", "user.email=benchmark@example.invalid",
        "commit", "--allow-empty", "-qm", "Exact reviewed public inputs")
    instruction = output / "instruction.md"
    instruction.write_bytes(_read(task / "instruction.md", selected["bindings"]["public_metadata"]["instruction.md"]))
    (output / "profile.json").write_bytes(task_profile_bytes(selected["profile"]))
    from . import terminal_indexed_preparation as preparation
    started = time.monotonic()
    prepared = preparation.prepare(repository=root, instruction=instruction,
        state=output / "state", task_profile=selected["profile"], disable_intent_autoencoder=True)
    context = preparation.initial_context(state=output / "state")
    result = dict(schema="terminal-public-profile-qualification@1", task_name=task_name,
        **selected, signed_input_count=len(prepared["manifest"]["payload"]["sources"]),
        profile_sha256=hashlib.sha256(task_profile_bytes(selected["profile"])).hexdigest(),
        original_input_sha256={name: row["sha256"] for name, row in
            json.loads((output / "state/original-image.json").read_text())["sources"].items()},
        indexed_symbols=context["indexed_symbols"], disposition=context.get("disposition", "indexed"),
        full_capsules=context["full_capsules"], provider_calls=0,
        seconds=time.monotonic()-started, initial_context_qualified=True,
        input_reconstruction="exact reviewed COPY bytes, not a live image inventory",
        task_data_semantics_verified=False, completion_authority=False)
    (output / "qualification.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args(argv)
    if args.qualify:
        result = qualify_public_profile(dataset=args.dataset, task_name=args.task, output=args.output)
        print(json.dumps({key: result[key] for key in ("task_name", "indexed_symbols", "provider_calls", "seconds")}))
    else:
        result = reviewed_profile(dataset=args.dataset, task_name=args.task)
        with args.output.open("xb") as stream:
            stream.write(task_profile_bytes(result["profile"]))
        print(json.dumps({"task_name": args.task, "profile": str(args.output), "official_reward": None}))


if __name__ == "__main__":
    main()
