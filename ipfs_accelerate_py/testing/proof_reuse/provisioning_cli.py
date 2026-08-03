"""Explicit setup-facing provisioning for optional proof-reuse capabilities.

This module is an operator command, not an installation hook.  Importing or
installing :mod:`ipfs_accelerate_py` never calls it.  When invoked explicitly,
it delegates to :class:`ProofReuseLazyDependencyInstaller` and therefore keeps
the existing allowlists, consent gates, locks, timeouts, receipts, and typed
RUN/DEFERRED fallbacks.

NLTK's Python distribution is ordinary package metadata.  This command only
provisions its allowlisted data resources.  Groth16 is a Cargo-native
capability, not a Python distribution, and this command never performs trusted
setup or generates proving/verifying keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Final

PROOF_REUSE_PROVISION_COMMAND_INTERFACE: Final = "ProofReuseProvisionCommand@1"
PROOF_REUSE_RUNTIME_MIN_PYTHON: Final = (3, 12)
REASON_UNSUPPORTED_PYTHON_RUNTIME: Final = "unsupported_python_runtime"


def _normalized_runtime_version(
    runtime_version: Sequence[int] | None = None,
) -> tuple[int, int, int]:
    """Return a bounded numeric runtime version without importing providers."""

    raw: Any = sys.version_info if runtime_version is None else runtime_version
    try:
        values = tuple(int(value) for value in tuple(raw)[:3])
    except (TypeError, ValueError, OverflowError):
        return (0, 0, 0)
    return (values + (0, 0, 0))[:3]


def _runtime_supported(runtime_version: Sequence[int] | None = None) -> bool:
    detected = _normalized_runtime_version(runtime_version)
    return detected[:2] >= PROOF_REUSE_RUNTIME_MIN_PYTHON


def _runtime_nltk_policy() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load the NLTK policy only after the reviewed Python floor is met."""

    from .services import (
        DEFAULT_NLTK_DATA_RESOURCES,
        NLTK_DATA_RESOURCE_ALLOWLIST,
    )

    return (
        tuple(DEFAULT_NLTK_DATA_RESOURCES),
        tuple(sorted(NLTK_DATA_RESOURCE_ALLOWLIST)),
    )


def get_default_lazy_dependency_installer() -> Any:
    """Resolve the runtime installer lazily after the version boundary."""

    from .lazy_dependencies import (
        get_default_lazy_dependency_installer as _get_default_installer,
    )

    return _get_default_installer()


def _parser(
    *,
    nltk_resource_choices: Sequence[str] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ipfs-accelerate-proof-reuse-provision",
        description=(
            "Explicitly provision bounded optional proof-reuse capabilities. "
            "With no capability flag, both NLTK data and the native Groth16 "
            "backend are requested. Existing environment consent gates still "
            "apply."
        ),
    )
    parser.add_argument(
        "--nltk-data",
        action="store_true",
        help="request the allowlisted NLTK data resources",
    )
    resource_options: dict[str, Any] = {}
    if nltk_resource_choices is not None:
        resource_options["choices"] = tuple(nltk_resource_choices)
    parser.add_argument(
        "--nltk-resource",
        action="append",
        default=[],
        help="request one allowlisted NLTK resource (repeatable)",
        **resource_options,
    )
    parser.add_argument(
        "--groth16-native",
        action="store_true",
        help="request the reviewed native Groth16 binary",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="return a nonzero status when a requested capability is unavailable",
    )
    return parser


def _unsupported_runtime_report(
    *,
    runtime_version: Sequence[int] | None,
    nltk_data: bool,
    groth16_native: bool,
    nltk_resources: Sequence[str] = (),
) -> dict[str, Any]:
    """Return the typed fallback without importing proof-runtime modules."""

    detected = _normalized_runtime_version(runtime_version)
    required = ">=" + ".".join(
        str(value) for value in PROOF_REUSE_RUNTIME_MIN_PYTHON
    )
    detected_text = ".".join(str(value) for value in detected)
    bounded_resources = [str(value)[:64] for value in tuple(nltk_resources)[:16]]
    runtime_failure = {
        "available": False,
        "reason_code": REASON_UNSUPPORTED_PYTHON_RUNTIME,
        "capability": "proof_reuse_runtime",
        "installed": False,
        "action": "RUN_OR_DEFERRED",
        "diagnostics": {
            "required_python": required,
            "detected_python": detected_text,
            "runtime_dependency_import_attempted": False,
        },
    }
    results: dict[str, Any] = {"runtime": runtime_failure}
    if nltk_data:
        results["nltk_data"] = {
            **runtime_failure,
            "capability": "nltk_data",
            "action": "RUN",
        }
    if groth16_native:
        results["groth16_native"] = {
            **runtime_failure,
            "capability": "groth16_native",
            "action": "DEFERRED",
        }
    return {
        "interface": PROOF_REUSE_PROVISION_COMMAND_INTERFACE,
        "requested": {
            "nltk_data": bool(nltk_data),
            "groth16_native": bool(groth16_native),
            "nltk_resources": bounded_resources if nltk_data else [],
        },
        "ready": False,
        "action": "RUN_OR_DEFERRED",
        "results": results,
        "dependency_plan": {
            "available": False,
            "reason_code": REASON_UNSUPPORTED_PYTHON_RUNTIME,
            "required_python": required,
            "detected_python": detected_text,
        },
        "network_attempted": False,
        "process_started": False,
        "trusted_setup_attempted": False,
    }


def _typed_failure(capability: str, exc: BaseException) -> dict[str, Any]:
    """Return bounded public diagnostics for an optional-boundary exception."""

    return {
        "available": False,
        "reason_code": "provisioner_exception",
        "capability": capability,
        "installed": False,
        "action": "DEFERRED" if capability == "groth16_native" else "RUN",
        "diagnostics": {"error_type": type(exc).__name__[:96]},
    }


def _resolution_payload(value: Any, capability: str) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception as exc:  # noqa: BLE001 - optional boundary.
            return _typed_failure(capability, exc)
        if isinstance(payload, Mapping):
            return dict(payload)
    return {
        "available": bool(getattr(value, "available", False)),
        "reason_code": str(getattr(value, "reason_code", "invalid_resolution"))[
            :96
        ],
        "capability": capability,
        "installed": bool(getattr(value, "installed", False)),
        "action": str(getattr(value, "action", "RUN"))[:16],
        "diagnostics": {},
    }


def _safe_plan(installer: Any) -> dict[str, Any]:
    dependency_plan = getattr(installer, "dependency_plan", None)
    if not callable(dependency_plan):
        return {"available": False, "reason_code": "dependency_plan_unavailable"}
    try:
        payload = dependency_plan()
    except Exception as exc:  # noqa: BLE001 - diagnostic command must be finite.
        return _typed_failure("dependency_plan", exc)
    return dict(payload) if isinstance(payload, Mapping) else {
        "available": False,
        "reason_code": "invalid_dependency_plan",
    }


def provision(
    *,
    nltk_data: bool,
    groth16_native: bool,
    nltk_resources: Sequence[str] | None = None,
    installer: Any = None,
    runtime_version: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Provision requested capabilities and return one bounded typed report."""

    if not _runtime_supported(runtime_version):
        return _unsupported_runtime_report(
            runtime_version=runtime_version,
            nltk_data=nltk_data,
            groth16_native=groth16_native,
            nltk_resources=nltk_resources or (),
        )

    if nltk_resources is None:
        try:
            nltk_resources, _choices = _runtime_nltk_policy()
        except Exception as exc:  # noqa: BLE001 - optional runtime boundary.
            return {
                "interface": PROOF_REUSE_PROVISION_COMMAND_INTERFACE,
                "requested": {
                    "nltk_data": bool(nltk_data),
                    "groth16_native": bool(groth16_native),
                    "nltk_resources": [],
                },
                "ready": False,
                "action": "RUN_OR_DEFERRED",
                "results": {"runtime": _typed_failure("runtime", exc)},
                "dependency_plan": {
                    "available": False,
                    "reason_code": "runtime_policy_unavailable",
                },
                "trusted_setup_attempted": False,
            }

    try:
        selected_installer = installer or get_default_lazy_dependency_installer()
    except Exception as exc:  # noqa: BLE001 - fail gracefully before provisioning.
        return {
            "interface": PROOF_REUSE_PROVISION_COMMAND_INTERFACE,
            "requested": {
                "nltk_data": bool(nltk_data),
                "groth16_native": bool(groth16_native),
            },
            "ready": False,
            "action": "RUN_OR_DEFERRED",
            "results": {"installer": _typed_failure("installer", exc)},
            "dependency_plan": {
                "available": False,
                "reason_code": "installer_unavailable",
            },
            "trusted_setup_attempted": False,
        }

    results: dict[str, Any] = {}
    readiness: list[bool] = []
    if nltk_data:
        try:
            resolution = selected_installer.ensure_nltk_data(
                tuple(nltk_resources)
            )
            result = _resolution_payload(resolution, "nltk_data")
        except Exception as exc:  # noqa: BLE001 - preserve RUN behavior.
            result = _typed_failure("nltk_data", exc)
        results["nltk_data"] = result
        readiness.append(result.get("available") is True)

    if groth16_native:
        try:
            resolution = selected_installer.ensure_groth16_native_backend()
            result = _resolution_payload(resolution, "groth16_native")
        except Exception as exc:  # noqa: BLE001 - preserve DEFERRED behavior.
            result = _typed_failure("groth16_native", exc)
        results["groth16_native"] = result
        readiness.append(result.get("available") is True)
        inspect_runtime = getattr(selected_installer, "inspect_groth16_runtime", None)
        if callable(inspect_runtime):
            try:
                runtime = inspect_runtime()
                results["groth16_runtime"] = (
                    dict(runtime)
                    if isinstance(runtime, Mapping)
                    else {
                        "ready": False,
                        "reason_code": "invalid_runtime_status",
                    }
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic only.
                results["groth16_runtime"] = _typed_failure(
                    "groth16_runtime", exc
                )

    ready = bool(readiness) and all(readiness)
    return {
        "interface": PROOF_REUSE_PROVISION_COMMAND_INTERFACE,
        "requested": {
            "nltk_data": bool(nltk_data),
            "groth16_native": bool(groth16_native),
            "nltk_resources": list(nltk_resources) if nltk_data else [],
        },
        "ready": ready,
        "action": "READY" if ready else "RUN_OR_DEFERRED",
        "results": results,
        "dependency_plan": _safe_plan(selected_installer),
        "trusted_setup_attempted": False,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_version: Sequence[int] | None = None,
) -> int:
    if not _runtime_supported(runtime_version):
        # Parse only the stable command shape.  Resource choices live in the
        # Python-3.12+ services module and must not be imported on an older
        # interpreter merely to format an unavailable result.
        args = _parser(nltk_resource_choices=None).parse_args(argv)
        request_nltk = bool(args.nltk_data or args.nltk_resource)
        request_groth16 = bool(args.groth16_native)
        if not request_nltk and not request_groth16:
            request_nltk = True
            request_groth16 = True
        report = _unsupported_runtime_report(
            runtime_version=runtime_version,
            nltk_data=request_nltk,
            groth16_native=request_groth16,
            nltk_resources=tuple(args.nltk_resource),
        )
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 2 if args.require_ready else 0

    default_resources, resource_choices = _runtime_nltk_policy()
    args = _parser(nltk_resource_choices=resource_choices).parse_args(argv)
    request_nltk = bool(args.nltk_data or args.nltk_resource)
    request_groth16 = bool(args.groth16_native)
    if not request_nltk and not request_groth16:
        request_nltk = True
        request_groth16 = True
    resources = tuple(args.nltk_resource) or default_resources
    report = provision(
        nltk_data=request_nltk,
        groth16_native=request_groth16,
        nltk_resources=resources,
        runtime_version=runtime_version,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    if args.require_ready and report.get("ready") is not True:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - console entry point.
    raise SystemExit(main())


__all__ = [
    "PROOF_REUSE_PROVISION_COMMAND_INTERFACE",
    "PROOF_REUSE_RUNTIME_MIN_PYTHON",
    "REASON_UNSUPPORTED_PYTHON_RUNTIME",
    "main",
    "provision",
]
