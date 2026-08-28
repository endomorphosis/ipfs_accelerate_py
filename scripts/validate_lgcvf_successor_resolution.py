#!/usr/bin/env python3
"""Create once or validate an LGCVF successor-resolution artifact.

The authority callback is deliberately injected as ``MODULE:CALLABLE``.  It
must authenticate S001/S002 receipts and return the versioned, content-addressed
verdict consumed by ``lgcvf_successor_resolution``.  This wrapper neither signs
receipts nor nominates a trust root.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.validation.lgcvf_successor_resolution import (
    AuthorityReceiptValidator,
    LgcvfSuccessorResolutionError,
    build_successor_resolution,
    validate_successor_resolution,
)

EVIDENCE_ROOT: Final[Path] = (
    ROOT
    / "data"
    / "agent_supervisor"
    / "logic_governed_compositional_verification_fabric"
)
DEFAULT_OUTPUT: Final[Path] = EVIDENCE_ROOT / "successor_resolution.json"
DEFAULT_PREDECESSOR: Final[Path] = EVIDENCE_ROOT / "successor_tasks.json"
DEFAULT_QUALIFICATION: Final[Path] = (
    EVIDENCE_ROOT / "independent_qualification_result.json"
)
DEFAULT_BENCHMARK: Final[Path] = EVIDENCE_ROOT / "benchmark_result.json"
DEFAULT_S001_RECEIPT: Final[Path] = (
    EVIDENCE_ROOT / "external_qualification_r_and_d_receipt.v2.json"
)
DEFAULT_S002_RECEIPT: Final[Path] = (
    EVIDENCE_ROOT / "production_authorization_r_and_d_receipt.v2.json"
)
MAX_JSON_BYTES: Final[int] = 16 * 1024 * 1024


class ResolutionCliError(RuntimeError):
    """The CLI input, callback, source root, or append-only output is invalid."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        encoded = path.read_bytes()
        if len(encoded) > MAX_JSON_BYTES:
            raise ResolutionCliError(f"{label} exceeds the bounded JSON size")
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except ResolutionCliError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResolutionCliError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ResolutionCliError(f"{label} root is not an object")
    return value


def _git(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository), *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ResolutionCliError(f"Git source-root query failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-1_000:]
        raise ResolutionCliError(
            f"Git source-root query returned {completed.returncode}: {detail}"
        )
    value = completed.stdout.strip()
    if not value or "\n" in value:
        raise ResolutionCliError("Git source-root query returned an ambiguous object")
    return value


def _revision(repository: Path, value: str) -> tuple[str, str]:
    if (
        not value
        or value.startswith("-")
        or any(character.isspace() for character in value)
    ):
        raise ResolutionCliError(
            "Git revision is empty, option-like, or contains whitespace"
        )
    commit = _git(repository, "rev-parse", "--verify", f"{value}^{{commit}}")
    tree = _git(repository, "rev-parse", "--verify", f"{commit}^{{tree}}")
    return commit, tree


def _source_roots(
    *,
    accelerator_revision: str,
    datasets_revision: str,
) -> dict[str, dict[str, str]]:
    accelerator_head, accelerator_tree = _revision(ROOT, accelerator_revision)
    datasets_root = ROOT / "ipfs_datasets_py"
    datasets_head, datasets_tree = _revision(datasets_root, datasets_revision)
    datasets_gitlink = _git(
        ROOT,
        "rev-parse",
        "--verify",
        f"{accelerator_head}:ipfs_datasets_py",
    )
    if datasets_head != datasets_gitlink:
        raise ResolutionCliError(
            "datasets revision differs from the selected accelerator gitlink"
        )
    return {
        "ipfs_accelerate_py": {
            "head": accelerator_head,
            "tree": accelerator_tree,
        },
        "ipfs_datasets_py": {
            "head": datasets_head,
            "tree": datasets_tree,
            "gitlink": datasets_gitlink,
        },
    }


def _import_callable(specification: str) -> Callable[..., Any]:
    module_name, separator, attribute_path = specification.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ResolutionCliError(
            "authority validator must use the MODULE:CALLABLE form"
        )
    try:
        candidate: Any = importlib.import_module(module_name)
        for component in attribute_path.split("."):
            if not component or component.startswith("_"):
                raise AttributeError(attribute_path)
            candidate = getattr(candidate, component)
    except (ImportError, AttributeError) as exc:
        raise ResolutionCliError(
            f"authority validator is unavailable: {specification}"
        ) from exc
    if not callable(candidate):
        raise ResolutionCliError("authority validator target is not callable")
    return candidate


def _authority_validator(
    specification: str,
    configuration: Mapping[str, Any] | None,
) -> AuthorityReceiptValidator:
    callback = _import_callable(specification)
    if configuration is None:
        return callback

    def configured(**arguments: Any) -> Mapping[str, Any]:
        result = callback(**arguments, configuration=dict(configuration))
        if not isinstance(result, Mapping):
            raise ResolutionCliError(
                "configured authority validator returned no object"
            )
        return result

    return configured


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
    except FileExistsError as exc:
        raise ResolutionCliError(
            f"append-only resolution already exists and will not be replaced: {path}"
        ) from exc
    except OSError as exc:
        raise ResolutionCliError(f"resolution cannot be created: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--emit",
        action="store_true",
        help="build, validate, and exclusively create the resolution artifact",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="validate an already-created resolution artifact without writing",
    )
    parser.add_argument("--resolution", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--predecessor", type=Path, default=DEFAULT_PREDECESSOR)
    parser.add_argument("--qualification", type=Path, default=DEFAULT_QUALIFICATION)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--s001-receipt", type=Path, default=DEFAULT_S001_RECEIPT)
    parser.add_argument("--s002-receipt", type=Path, default=DEFAULT_S002_RECEIPT)
    parser.add_argument(
        "--authority-validator",
        required=True,
        metavar="MODULE:CALLABLE",
        help=(
            "injected authenticated-receipt adapter implementing the successor "
            "authority callback contract"
        ),
    )
    parser.add_argument(
        "--authority-validator-config",
        type=Path,
        help="optional JSON object passed to the adapter as configuration=...",
    )
    parser.add_argument(
        "--accelerator-revision",
        required=True,
        help="exact semantic accelerator commit; HEAD is accepted only when explicit",
    )
    parser.add_argument(
        "--datasets-revision",
        required=True,
        help="exact datasets commit selected by the accelerator gitlink",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _parser()
    options = parser.parse_args(arguments)
    try:
        predecessor = _load_object(options.predecessor, label="predecessor")
        qualification = _load_object(options.qualification, label="qualification")
        benchmark = _load_object(options.benchmark, label="benchmark")
        receipts = {
            "LGCVF-S001": _load_object(options.s001_receipt, label="S001 receipt"),
            "LGCVF-S002": _load_object(options.s002_receipt, label="S002 receipt"),
        }
        configuration = (
            _load_object(
                options.authority_validator_config,
                label="authority validator configuration",
            )
            if options.authority_validator_config is not None
            else None
        )
        validator = _authority_validator(options.authority_validator, configuration)
        roots = _source_roots(
            accelerator_revision=options.accelerator_revision,
            datasets_revision=options.datasets_revision,
        )
        if options.emit:
            resolution = build_successor_resolution(
                predecessor=predecessor,
                qualification=qualification,
                benchmark=benchmark,
                source_roots=roots,
                authority_receipts=receipts,
                authority_validator=validator,
            )
            _write_once(options.resolution, resolution)
            result = {
                "schema": "lgcvf-successor-resolution-cli-result@1",
                "created": True,
                "valid": True,
                "path": str(options.resolution),
                "resolution_cid": resolution["resolution_cid"],
            }
        else:
            resolution = _load_object(options.resolution, label="successor resolution")
            validation = validate_successor_resolution(
                resolution,
                predecessor=predecessor,
                qualification=qualification,
                benchmark=benchmark,
                expected_source_roots=roots,
                authority_receipts=receipts,
                authority_validator=validator,
            )
            result = {
                "schema": "lgcvf-successor-resolution-cli-result@1",
                "created": False,
                **validation,
            }
    except (ResolutionCliError, LgcvfSuccessorResolutionError) as exc:
        parser.exit(1, f"LGCVF successor resolution rejected: {exc}\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
