"""Configure, but never start, the optional owner-side repository source service.

The installed isolation guard remains authoritative. This helper accepts no
private capacity from worker arguments and supplies no fallback scheduler. Its
caller must retain the returned lifecycle before explicitly starting it.
"""
from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
import time


MAX_ARGUMENTS = 256
MAX_ARGUMENT_BYTES = 8192
MAX_TOTAL_ARGUMENT_BYTES = 65536
_POLICY = "--repository-source-policy"
# These prefixes were unique before owner descriptor flags were introduced.
# --re was already ambiguous with --reasoning-effort. Preserve only legacy
# no-op selections; an active owner selection must still use the exact flag.
_LEGACY_POLICY_FLAGS = frozenset(_POLICY[:end] for end in range(len("--rep"), len(_POLICY) + 1))
_REPRESENTATIONS = {"repository-source-raw@1": "original",
                    "repository-source-compact@1": "compact"}
_VALUES = {_POLICY, "--provider", "--purpose", "--semantic-repository",
           "--source-view-advisory-policy", "--source-view-profile", "--source-edit-mode"}
_OTHER_ROUTES = {
    "--preflight", "--semantic-metadata-census-only", "--census-only", "--validate",
    "--doctor-candidate-artifact", "--doctor-candidate-sha256", "--doctor-task-cid",
    "--doctor-contract-artifact", "--doctor-contract-sha256", "--doctor-contract-task-cid",
    "--doctor-residual-artifact", "--doctor-residual-sha256", "--doctor-residual-task-cid",
    "--finite-repository-artifact", "--finite-repository-sha256", "--finite-repository-task-cid",
    "--finite-proof-query-context", "--finite-proof-query-sha256", "--finite-proof-query-context-cid",
    "--semantic-decision-policy-artifact", "--semantic-decision-policy-sha256",
    "--semantic-decision-nomination-artifact", "--semantic-decision-nomination-sha256",
    "--semantic-decision-task-id", "--semantic-decision-typed-nomination",
    "--semantic-decision-context-bundle-artifact", "--semantic-decision-context-bundle-sha256",
    "--semantic-decision-task-cid", "--source-view-descriptor", "--source-view-descriptor-sha256",
}
_PRIVATE = {
    "--repository-source-owner-descriptor", "--repository-source-owner-descriptor-sha256",
    "--repository-source-parent-lease", "--repository-source-scheduler-policy-sha256",
    "--repository-source-scheduler-state-path", "--repository-source-deadline-monotonic",
    "--repository-source-resource-grant", "--repository-source-resource-grant-sha256",
    "--resource-grant", "--resource-grant-sha256", "--parent-lease", "--parent-token",
    "--scheduler-state-path", "--scheduler-policy-sha256", "--expected-policy-sha256",
    "--lease-key", "--lease-token", "--state-owner-token",
}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _literal_argv(argv):
    _need(type(argv) is list and len(argv) <= MAX_ARGUMENTS,
          "bounded literal source-owner argv required")
    total = 0
    for item in argv:
        _need(type(item) is str and "\0" not in item, "literal source-owner arguments required")
        try:
            size = len(item.encode("utf-8"))
        except UnicodeError as error:
            raise ValueError("UTF-8 source-owner arguments required") from error
        _need(size <= MAX_ARGUMENT_BYTES, "source-owner argument exceeds byte bound")
        total += size + 1
    _need(total <= MAX_TOTAL_ARGUMENT_BYTES, "source-owner argv exceeds total byte bound")
    return [item.partition("=")[0] for item in argv if item.startswith("--")]


def _reject_abbreviations(flags, relevant):
    for flag in flags:
        _need(flag != "--", "source-owner arguments cannot terminate option parsing")
        if flag not in relevant:
            _need(not any(name.startswith(flag) for name in relevant),
                  "source-owner relevant option must use its exact spelling")


def _argument(argv, name, default=None):
    from .numerical_obligation_broker import _argument as existing_argument
    return existing_argument(argv, name, default)


def load_source_owner_configuration(boundary):
    from .repository_source_owner_configuration import load_source_owner_configuration as load
    return load(boundary)


def _new_lifecycle(**kwargs):
    from .repository_source_owner_lifecycle import RepositorySourceOwnerLifecycle
    return RepositorySourceOwnerLifecycle(**kwargs)


def _canonical_path(value):
    _need(type(value) is str and value and len(value.encode("utf-8")) <= 4096,
          "explicit bounded source-owner semantic repository required")
    path = PurePosixPath(value)
    _need(path.is_absolute() and str(path) == value and value != "/"
          and not value.startswith("//") and ".." not in path.parts
          and not any(ord(char) < 32 or ord(char) == 127 or char == "\\" for char in value),
          "canonical absolute source-owner semantic repository required")
    return Path(value)


def configure_repository_source_owner_dispatch(*, argv, boundary, workspace, deadline_monotonic):
    """Return an unstarted lifecycle, or None for the legacy source route.

    This is a bounded adapter for the existing owner entrypoint, not a parser,
    launcher or resource owner. Other ordinary worker options retain their
    existing parser and validation responsibilities.
    """
    flags = _literal_argv(argv)
    _reject_abbreviations([flag for flag in flags if flag not in _LEGACY_POLICY_FLAGS],
                          {_POLICY, *_PRIVATE})
    _need(not any(flag in _PRIVATE for flag in flags),
          "source-owner descriptors and private capabilities are owner-created")
    selections = []
    for index, item in enumerate(argv):
        flag, separator, _ = item.partition("=")
        if flag in _LEGACY_POLICY_FLAGS:
            # Reuse the established missing-value rule without letting its
            # duplicate rejection erase historical repeated all-legacy flags.
            occurrence = argv[index:index + (1 if separator else 2)]
            selections.append((flag, _argument(occurrence, flag)))
    if not selections or all(value == "legacy" for _, value in selections):
        return None
    _need(len(selections) == 1 and selections[0][0] == _POLICY,
          "active source-owner policy requires one exact option")
    policy = selections[0][1]
    _need(policy in _REPRESENTATIONS, "unsupported repository source policy")
    relevant = _VALUES | _OTHER_ROUTES | _PRIVATE
    _reject_abbreviations(flags, relevant)
    _need(all(flags.count(flag) <= 1 for flag in relevant),
          "source-owner relevant option is duplicated")
    _need(not any(flag in _OTHER_ROUTES for flag in flags),
          "source owner requires the exclusive ordinary coding route")
    _need(_argument(argv, "--provider", "codex_cli") == "codex_cli"
          and _argument(argv, "--purpose", "coding") == "coding",
          "source owner requires Codex coding")
    _need(_argument(argv, "--source-view-advisory-policy", "legacy") == "legacy"
          and _argument(argv, "--source-view-profile", "disabled") == "disabled"
          and _argument(argv, "--source-edit-mode", "disabled") == "disabled",
          "source owner excludes other source views and source editing")
    repository = _canonical_path(_argument(argv, "--semantic-repository"))
    _need(type(deadline_monotonic) in (int, float)
          and time.monotonic() < deadline_monotonic <= time.monotonic() + 1800,
          "existing bounded absolute source-owner deadline required")
    _need(math.isfinite(deadline_monotonic), "finite source-owner deadline required")
    configuration = load_source_owner_configuration(boundary)
    _need(type(configuration) is dict,
          "selected source route requires explicit deployed native resource configuration")
    # The protected loader owns the closed configuration contract. Pass only
    # its explicit fields, never argv/environment capacity or new defaults.
    return _new_lifecycle(repository=repository, workspace=workspace, boundary=boundary,
        scheduler_state_path=configuration["scheduler_state_path"],
        expected_policy_sha256=configuration["scheduler_policy_sha256"],
        deadline_monotonic=deadline_monotonic,
        admission_timeout_seconds=configuration["admission_timeout_seconds"],
        representation=_REPRESENTATIONS[policy])
