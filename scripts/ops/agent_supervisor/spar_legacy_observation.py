"""Observe the accepted SPAR source in fresh processes with its own imports.

The capture implementation may live in a candidate checkout. Importing the
accepted operator into that process would reuse candidate packages already in
sys.modules. Source qualification therefore runs in an isolated interpreter;
status uses the complete native CLI and its normal reader admission contract.
This module never starts or stops an owner, edits a source, or reads a vault.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

OPERATOR = "scripts/materialize_semantic_preserving_remodularization_program.py"
MAX_OUTPUT = 16 * 1024 * 1024


class NativeObservationError(RuntimeError):
    pass


def native_observation_cid(value):
    """Hash the complete native status without the smaller queue-receipt limit."""
    try:
        body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as exc:
        raise NativeObservationError("native observation is not bounded JSON") from exc
    if len(body) > MAX_OUTPUT:
        raise NativeObservationError("native observation exceeds bound")
    return "sha256:" + hashlib.sha256(body).hexdigest()


class NativeObservationClient:
    def __init__(self, root):
        self.root = Path(root).absolute()
        self.config_path = None
        self._paths = None

    @staticmethod
    def _identity(value):
        body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        if len(body) > MAX_OUTPUT:
            raise NativeObservationError("native identity exceeds bound")
        return "sha256:" + hashlib.sha256(body).hexdigest()

    def _call(self, argv):
        result = subprocess.run(argv, cwd=self.root, capture_output=True,
                                timeout=90, check=False)
        # Do not forward arbitrary operator diagnostics or any private material.
        if result.returncode or len(result.stdout) > MAX_OUTPUT:
            raise NativeObservationError("native isolated observation failed")
        try:
            value = json.loads(result.stdout)
        except (ValueError, UnicodeError) as exc:
            raise NativeObservationError("native isolated observation malformed") from exc
        if type(value) is not dict:
            raise NativeObservationError("native isolated observation is not an object")
        return value

    def _observe(self, action, config_path):
        return self._call([
            "/usr/bin/python3", "-I", str(Path(__file__).resolve()),
            "--root", str(self.root), "--config", str(config_path), "--action", action,
        ])

    def _load_config(self, config_path):
        self.config_path = Path(config_path).absolute()
        value = self._observe("describe", self.config_path)
        board = SimpleNamespace(**value["board"])
        board.repo_root = Path(board.repo_root)
        board.path = lambda name: Path(value["resolved_paths"][str(name)])
        self._paths = {k: Path(v) for k, v in value["paths"].items()}
        return board, {}

    def _runtime_paths(self, board):
        if self._paths is None or Path(board.repo_root) != self.root:
            raise NativeObservationError("native isolated path binding differs")
        return dict(self._paths)

    def source_binding(self, config_path):
        return self._observe("source", config_path)

    def authoritative_status(self, config_path):
        # Complete accepted native command. Credentials stay inside its existing
        # operator; no token is copied into this capture process.
        return self._call([
            "/usr/bin/python3", "-I", str(self.root / OPERATOR),
            "--config", str(config_path), "authoritative-status",
        ])


def observe(root, config_path, action):
    root = Path(root).absolute()
    config_path = Path(config_path).absolute()
    # The CLI's -I invocation gives us no candidate package imports. Select the
    # accepted root before executing any native operator or repository module.
    sys.path.insert(0, str(root))
    native = runpy.run_path(str(root / OPERATOR), run_name="_spar_native_observation")
    before = config_path.read_bytes()
    board, config = native["_load_config"](config_path)
    if Path(board.repo_root).absolute() != root:
        raise NativeObservationError("native observation repository differs")
    if action == "describe":
        runtime = dict(board.runtime_paths)
        value = {
            "board": {"repo_root": str(board.repo_root),
                      "board_namespace": board.board_namespace,
                      "task_prefix": board.task_prefix,
                      "max_lanes": board.max_lanes,
                      "runtime_paths": runtime,
                      "payload": {"runtime_paths": dict(board.payload["runtime_paths"]),
                                  "merge_target_branch": board.payload.get("merge_target_branch", "")}},
            "paths": {key: str(path) for key, path in native["_runtime_paths"](board).items()},
            "resolved_paths": {str(value): str(board.path(value)) for value in
                [*runtime.values(), *board.payload["runtime_paths"].values()]},
        }
    elif action == "source":
        head, tree = native["_assert_clean_current_tree"](config)
        value = {"head": head, "tree": tree,
                 "forest": native["_source_forest"](config, head=head),
                 "config_sha256": hashlib.sha256(before).hexdigest()}
        if native["_assert_clean_current_tree"](config) != (head, tree):
            raise NativeObservationError("native source changed during observation")
    else:
        raise NativeObservationError("unknown native observation action")
    if config_path.read_bytes() != before:
        raise NativeObservationError("native configuration changed during observation")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", choices=("describe", "source"), required=True)
    args = parser.parse_args()
    try:
        value = observe(args.root, args.config, args.action)
    except Exception:
        # Public failure class only; private native diagnostics are not exported.
        print(json.dumps({"error": "native_observation_rejected"}))
        return 1
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
