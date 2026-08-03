"""Deterministic review of implementation/validation failures.

After a candidate fails proposal validation or pre-merge checks, the supervisor
reviews the failure and either:

* **accept** — a bounded, fail-closed override when the only residual issue is a
  previously justified scope expansion that revalidates cleanly; or
* **guide_rescue** — attach structured follow-up guidance for the rescue branch
  and the next implementation attempt.

This module never treats natural-language model rationales as authority and
never authorizes secret, protected-path, submodule, or test-weakening failures.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .proof.formal_verification_contracts import canonical_json, content_identity
from .validation.validation_commands import infer_validation_impact_paths


FAILURE_REVIEW_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/implementation-failure-review@1"
)
FAILURE_REVIEW_POLICY_VERSION = "deterministic-failure-review-v2"

# Failures that may never be accepted by the reviewer.
_HARD_DENY_FINDING_CODES = frozenset(
    {
        "secret_change_forbidden",
        "protected_path_forbidden",
        "submodule_boundary_forbidden",
        "symlink_boundary_forbidden",
        "hardlink_boundary_forbidden",
        "test_deletion_forbidden",
        "test_weakening_forbidden",
        "validation_weakening_forbidden",
        "unsafe_path",
        "forged_authority_claim",
        "authority_mismatch",
        "stale_baseline",
        "stale_proposal_replay",
    }
)

# Scope-only findings that can be resolved by justified companion expansion.
_SCOPE_RELATED_FINDING_CODES = frozenset(
    {
        "path_outside_scope",
    }
)

# Proposal-gate size / bulk findings that need size-aware rescue guidance.
# Defaults mirror ProposalValidationPolicy in proposal_validation.py.
_SIZE_RELATED_FINDING_CODES = frozenset(
    {
        "output_too_large",
        "patch_too_large",
        "patch_parse_error",
        "large_file_forbidden",
    }
)
DEFAULT_PROPOSAL_MAX_PATCH_BYTES = 2_000_000
DEFAULT_PROPOSAL_MAX_OUTPUT_BYTES = 2_500_000
DEFAULT_PROPOSAL_MAX_FILE_BYTES = 1_000_000


class FailureReviewDecision(str, Enum):
    """Outcome of one deterministic failure review."""

    ACCEPT = "accept"
    GUIDE_RESCUE = "guide_rescue"
    REJECT = "reject"


class FailureReviewReason(str, Enum):
    """Bounded reason codes for failure-review receipts."""

    SCOPE_EXPANSION_JUSTIFIED = "scope_expansion_justified"
    SCOPE_EXPANSION_DENIED = "scope_expansion_denied"
    HARD_DENY_FINDINGS = "hard_deny_findings"
    INCOMPLETE_EXPECTED_OUTPUTS = "incomplete_expected_outputs"
    PROPOSAL_GATE_FAILED = "proposal_gate_failed"
    VALIDATION_COMMAND_FAILED = "validation_command_failed"
    ENVIRONMENT_VALIDATION_UNAVAILABLE = "environment_validation_unavailable"
    LARGE_OR_UNDECLARED_REFACTOR = "large_or_undeclared_refactor"
    TASK_SCOPE_CONTRACT_REVISION_REQUIRED = (
        "task_scope_contract_revision_required"
    )
    EMPTY_OR_NO_CHANGE = "empty_or_no_change"
    GENERIC_IMPLEMENTATION_FAILURE = "generic_implementation_failure"
    NO_ACTIONABLE_EVIDENCE = "no_actionable_evidence"


def _normalize_path(value: Any) -> str:
    path = str(value or "").strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if (
        not path
        or path.startswith("/")
        or "\0" in path
        or ".." in PurePosixPath(path).parts
    ):
        return ""
    return PurePosixPath(path).as_posix()


def _normalized_paths(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                path
                for value in values
                if (path := _normalize_path(value))
            }
        )
    )


def _as_str_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        text = str(values).strip()
        return (text,) if text else ()
    if not isinstance(values, (list, tuple, set, frozenset)):
        return ()
    return tuple(
        sorted(
            {
                str(item).strip()
                for item in values
                if str(item).strip()
            }
        )
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finding_codes_from_validation(
    validation_result: Mapping[str, Any],
) -> tuple[str, ...]:
    codes: set[str] = set()
    for key in ("reason_codes", "finding_codes"):
        codes.update(_as_str_tuple(validation_result.get(key)))
    proposal = _mapping(validation_result.get("proposal_gate"))
    codes.update(_as_str_tuple(proposal.get("reason_codes")))
    codes.update(_as_str_tuple(proposal.get("finding_codes")))
    proposal_validation = _mapping(
        validation_result.get("proposal_validation")
    )
    findings = proposal_validation.get("findings") or ()
    if isinstance(findings, Sequence) and not isinstance(findings, (str, bytes)):
        for finding in findings:
            if not isinstance(finding, Mapping):
                continue
            code = finding.get("code")
            if isinstance(code, Mapping):
                code = code.get("value")
            text = str(code or "").strip()
            if text:
                codes.add(text)
    return tuple(sorted(codes))


def _changed_paths_from_validation(
    validation_result: Mapping[str, Any],
) -> tuple[str, ...]:
    proposal_paths: set[str] = set()
    for container_key in ("proposal_gate", "proposal_validation"):
        container = _mapping(validation_result.get(container_key))
        for key in ("changed_paths", "changed_files", "paths"):
            proposal_paths.update(
                path
                for path in _normalized_paths(container.get(key) or ())
            )
    # Nested proposal object.
    proposal_validation = _mapping(
        validation_result.get("proposal_validation")
    )
    proposal = _mapping(proposal_validation.get("proposal"))
    proposal_paths.update(
        _normalized_paths(proposal.get("changed_paths") or ())
    )
    if proposal_paths:
        return tuple(sorted(proposal_paths))

    # Legacy validation records may not contain a proposal projection. Keep
    # their selection paths as a best-effort fallback, but never union them
    # with authoritative proposal paths: selection.changed_files also includes
    # read-only validation impact paths and therefore is not candidate-edit
    # evidence.
    selection = _mapping(validation_result.get("selection"))
    selection_paths: set[str] = set()
    for key in ("changed_paths", "changed_files", "paths"):
        selection_paths.update(
            path
            for path in _normalized_paths(selection.get(key) or ())
        )
    return tuple(sorted(selection_paths))


def _failed_commands_from_validation(
    validation_result: Mapping[str, Any],
) -> tuple[str, ...]:
    commands: list[str] = []
    raw = validation_result.get("failed_commands")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Mapping):
                command = str(item.get("command") or item.get("cmd") or "").strip()
            else:
                command = str(item or "").strip()
            if command:
                commands.append(command)
    results = validation_result.get("results")
    if isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
        for item in results:
            if not isinstance(item, Mapping):
                continue
            if item.get("passed") is False or int(item.get("returncode") or 0) != 0:
                command = str(item.get("command") or "").strip()
                if command:
                    commands.append(command)
    return tuple(dict.fromkeys(commands))


def _scope_projection(
    validation_result: Mapping[str, Any],
) -> dict[str, Any]:
    for key in ("scope_adjudication",):
        payload = _mapping(validation_result.get(key))
        if payload:
            return dict(payload)
    proposal_validation = _mapping(
        validation_result.get("proposal_validation")
    )
    nested = _mapping(proposal_validation.get("scope_adjudication"))
    return dict(nested) if nested else {}


def _scope_contract_gap_paths(
    scope_adjudication: Mapping[str, Any],
    *,
    validation_commands: Sequence[str],
    validation_result: Mapping[str, Any],
) -> tuple[str, ...]:
    """Find denied test companions that need protected task revision."""

    validation_paths = set(
        _normalized_paths(
            validation_result.get("validation_impact_paths") or ()
        )
    )
    for command in validation_commands:
        validation_paths.update(
            _normalized_paths(infer_validation_impact_paths(str(command)))
        )
        try:
            tokens = shlex.split(str(command), posix=True)
        except ValueError:
            tokens = []
        for index, token in enumerate(tokens):
            raw_prefix = ""
            if token in {"--prefix", "--cwd", "--dir", "cd"}:
                if index + 1 < len(tokens):
                    raw_prefix = tokens[index + 1]
            elif token.startswith(("--prefix=", "--cwd=", "--dir=")):
                raw_prefix = token.split("=", 1)[1]
            prefix = _normalize_path(raw_prefix)
            if prefix:
                validation_paths.add(prefix)
    if not validation_paths:
        return ()

    contract_gap_reasons = {
        "test_change_unverifiable",
        "test_without_regression_evidence",
    }
    decisions = scope_adjudication.get("decisions") or ()
    if not isinstance(decisions, Sequence) or isinstance(
        decisions, (str, bytes)
    ):
        return ()

    paths: list[str] = []
    for decision in decisions:
        if not isinstance(decision, Mapping):
            continue
        if str(decision.get("verdict") or "").strip() != "denied":
            continue
        if not set(_as_str_tuple(decision.get("reason_codes"))).intersection(
            contract_gap_reasons
        ):
            continue
        path = _normalize_path(decision.get("path"))
        if not path or not any(
            path == target or path.startswith(target.rstrip("/") + "/")
            for target in validation_paths
        ):
            continue
        if path not in paths:
            paths.append(path)
    return tuple(sorted(paths))


def _is_environment_failure(
    validation_result: Mapping[str, Any],
    *,
    log_excerpt: str = "",
) -> bool:
    reason = str(validation_result.get("reason") or "").strip().lower()
    error = str(validation_result.get("error") or "").strip().lower()
    text = "\n".join(
        part
        for part in (
            reason,
            error,
            log_excerpt,
            "\n".join(_failed_commands_from_validation(validation_result)),
        )
        if part
    ).lower()
    needles = (
        "no module named pytest",
        "modulenotfounderror: no module named 'pytest'",
        "command not found",
        "pytest: not found",
        "environment_validation_unavailable",
        "validation_configuration_failed",
    )
    return any(needle in text for needle in needles)


def _path_under_or_equal(path: str, declared: str) -> bool:
    """Return True when ``path`` is ``declared`` or a descendant of it."""

    declared_norm = declared.rstrip("/")
    if not declared_norm:
        return False
    return path == declared_norm or path.startswith(declared_norm + "/")


def _path_owned_by_expected(path: str, expected: Sequence[str]) -> bool:
    """True when path equals or lives under any declared expected output."""

    return any(_path_under_or_equal(path, declared) for declared in expected)


def _expected_output_satisfied(
    declared: str,
    *,
    changed: set[str],
    workspace_path: Path | None,
) -> bool:
    """Return True when a declared file or directory output was produced.

    Directory outputs (for example ``tests/fixtures/foo``) are satisfied when
    any changed path is that directory or a descendant. Exact file outputs
    require an exact changed path. Workspace presence alone never counts as
    producing the output — the attempt must still touch declared ownership.
    """

    declared_norm = declared.rstrip("/")
    if not declared_norm:
        return False
    if declared_norm in changed:
        return True
    # Directory ownership: any descendant change satisfies the declared tree.
    prefix = declared_norm + "/"
    if any(path.startswith(prefix) for path in changed):
        return True
    # Optional: if the declared path itself was listed with a trailing slash
    # style only via descendants already handled above.
    _ = workspace_path  # reserved for future hermetic workspace probes
    return False


def _missing_expected_outputs(
    *,
    expected_outputs: Sequence[str],
    changed_paths: Sequence[str],
    workspace_path: Path | None,
) -> tuple[str, ...]:
    expected = _normalized_paths(expected_outputs)
    changed = _normalized_paths(changed_paths)
    missing: list[str] = []
    for path in expected:
        if _expected_output_satisfied(
            path,
            changed=changed,
            workspace_path=workspace_path,
        ):
            continue
        # Unchanged expected outputs remain "missing work" so rescue guidance
        # can say "create or update" rather than treating them as out of scope.
        missing.append(path)
    return tuple(missing)


def _size_related_findings(finding_codes: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        code for code in finding_codes if code in _SIZE_RELATED_FINDING_CODES
    )


def _guidance_lines(
    *,
    decision: FailureReviewDecision,
    reason_codes: Sequence[str],
    finding_codes: Sequence[str],
    missing_outputs: Sequence[str],
    out_of_scope_paths: Sequence[str],
    justified_paths: Sequence[str],
    denied_paths: Sequence[str],
    contract_gap_paths: Sequence[str],
    failed_commands: Sequence[str],
    expected_outputs: Sequence[str],
    validation_environment_guidance: str = "",
) -> list[str]:
    lines = [
        "# Implementation failure review",
        "",
        f"- Decision: `{decision.value}`",
        f"- Reason codes: {', '.join(f'`{code}`' for code in reason_codes) or 'none'}",
        f"- Finding codes: {', '.join(f'`{code}`' for code in finding_codes) or 'none'}",
        "",
        "## Follow-up guidance",
    ]
    if decision is FailureReviewDecision.ACCEPT:
        lines.extend(
            [
                "",
                "The supervisor accepted this candidate after a bounded failure review.",
                "No rescue rewrite is required for the reviewed findings.",
            ]
        )
        if justified_paths:
            lines.append(
                "Justified companion paths that were admitted: "
                + ", ".join(f"`{path}`" for path in justified_paths)
            )
        return lines

    lines.append("")
    lines.append("Do **not** widen scope casually. Stay inside the task contract.")
    if expected_outputs:
        lines.append("")
        lines.append("### Declared task outputs (exact edit authority)")
        for path in expected_outputs:
            lines.append(f"- `{path}`")
    if missing_outputs:
        lines.append("")
        lines.append("### Missing or unfinished expected outputs")
        lines.append(
            "Implement **every** declared output before finishing the attempt:"
        )
        for path in missing_outputs:
            lines.append(f"- create/update `{path}`")
    if out_of_scope_paths or denied_paths:
        lines.append("")
        lines.append("### Out-of-scope / denied paths")
        lines.append(
            "These paths are outside task-owned scope or were denied by "
            "deterministic adjudication. Prefer in-place edits of declared "
            "outputs; do not invent new modules or rename files unless both "
            "names are listed in Outputs/Predicted files."
        )
        for path in dict.fromkeys((*out_of_scope_paths, *denied_paths)):
            lines.append(f"- `{path}`")
        if justified_paths:
            lines.append("")
            lines.append(
                "Justified companion paths (import/test-linked) that may be "
                "re-admitted on revalidation:"
            )
            for path in justified_paths:
                lines.append(f"- `{path}`")
    if contract_gap_paths:
        lines.append("")
        lines.append("### Task-scope contract revision required")
        lines.append(
            "The proposal remains rejected. For each exact companion below, "
            "either revert the change or have protected-board authority add "
            "that exact path to Outputs / Predicted files before retrying. "
            "A broad validation command is diagnostic evidence, not edit "
            "authority."
        )
        for path in contract_gap_paths:
            lines.append(f"- `{path}`")
    if failed_commands:
        lines.append("")
        lines.append("### Failed validation commands")
        for command in failed_commands:
            lines.append(f"- `{command}`")
        lines.append(
            "Re-run these commands after edits and keep them green before exit."
        )
    if validation_environment_guidance:
        lines.extend(
            [
                "",
                "### Authoritative validation boundary",
                validation_environment_guidance,
            ]
        )
    if FailureReviewReason.ENVIRONMENT_VALIDATION_UNAVAILABLE.value in reason_codes:
        lines.append("")
        lines.append("### Environment")
        lines.append(
            "Validation could not import or execute the test runner. Fix the "
            "hermetic validation environment (pytest on PYTHONPATH) rather than "
            "changing product code around the tool failure."
        )
    if FailureReviewReason.LARGE_OR_UNDECLARED_REFACTOR.value in reason_codes:
        lines.append("")
        lines.append("### Refactor constraints")
        lines.append(
            "Large refactors are allowed **only inside declared output paths**. "
            "Do not extract helpers into new undeclared files; do not touch "
            "submodule gitlinks (for example `ipfs_accelerate_py/`); do not "
            "delete or weaken tests."
        )
    size_findings = _size_related_findings(finding_codes)
    if size_findings:
        lines.append("")
        lines.append("### Proposal size / bulk limits")
        lines.append(
            "The proposal gate rejected this candidate for size or patch bulk "
            f"({', '.join(f'`{code}`' for code in size_findings)}). "
            "Pytest green does **not** bypass admission."
        )
        lines.append(
            "Default admission budgets (strict-proposal-v1): "
            f"patch ≤ {DEFAULT_PROPOSAL_MAX_PATCH_BYTES} bytes, "
            f"provider output ≤ {DEFAULT_PROPOSAL_MAX_OUTPUT_BYTES} bytes, "
            f"single file ≤ {DEFAULT_PROPOSAL_MAX_FILE_BYTES} bytes."
        )
        lines.append(
            "Shrink the candidate **before** re-running the same dump:"
        )
        lines.extend(
            [
                "- Prefer compact recipes / generators over bulk golden dumps.",
                "- Rebuild large envelopes at test load time from smaller seeds.",
                "- Avoid duplicating full formal artifacts per case when one "
                "shared fixture plus variants suffices.",
                "- Split only if the task board declares separate Outputs; do "
                "not invent undeclared modules to dodge size limits.",
            ]
        )
    if FailureReviewReason.HARD_DENY_FINDINGS.value in reason_codes:
        lines.append("")
        lines.append("### Hard deny")
        lines.append(
            "Secret, protected-path, submodule, symlink, or test-weakening "
            "findings cannot be accepted. Remove those changes entirely."
        )
    lines.append("")
    lines.append("### Next attempt checklist")
    checklist = [
        "1. Touch only declared Outputs / Predicted files (plus justified companions).",
        "2. Deliver every listed expected output file or directory tree.",
        "3. Keep validation commands passing.",
        "4. Avoid renames, submodule edits, and undeclared new modules.",
    ]
    if size_findings:
        checklist.append(
            "5. Stay under proposal size budgets; use compact fixtures/"
            "generators instead of re-emitting oversized dumps."
        )
    lines.extend(checklist)
    return lines


@dataclass(frozen=True)
class ImplementationFailureReviewReceipt:
    """Content-addressed decision for one failed implementation attempt."""

    task_id: str
    attempt: int
    decision: FailureReviewDecision
    reason_codes: tuple[str, ...]
    finding_codes: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    changed_paths: tuple[str, ...]
    missing_expected_outputs: tuple[str, ...]
    out_of_scope_paths: tuple[str, ...]
    justified_paths: tuple[str, ...]
    denied_paths: tuple[str, ...]
    contract_gap_paths: tuple[str, ...]
    failed_commands: tuple[str, ...]
    guidance_markdown: str
    next_attempt_prompt_addendum: str
    policy_version: str = FAILURE_REVIEW_POLICY_VERSION
    proof_authoritative: bool = False
    completion_authoritative: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", str(self.task_id or "").strip())
        if not self.task_id:
            raise ValueError("task_id is required")
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if not isinstance(self.decision, FailureReviewDecision):
            object.__setattr__(
                self,
                "decision",
                FailureReviewDecision(str(self.decision)),
            )
        for name in (
            "reason_codes",
            "finding_codes",
            "expected_outputs",
            "changed_paths",
            "missing_expected_outputs",
            "out_of_scope_paths",
            "justified_paths",
            "denied_paths",
            "contract_gap_paths",
            "failed_commands",
        ):
            object.__setattr__(self, name, _as_str_tuple(getattr(self, name)))
        object.__setattr__(
            self,
            "expected_outputs",
            _normalized_paths(self.expected_outputs),
        )
        object.__setattr__(
            self,
            "changed_paths",
            _normalized_paths(self.changed_paths),
        )
        object.__setattr__(
            self,
            "missing_expected_outputs",
            _normalized_paths(self.missing_expected_outputs),
        )
        object.__setattr__(
            self,
            "out_of_scope_paths",
            _normalized_paths(self.out_of_scope_paths),
        )
        object.__setattr__(
            self,
            "justified_paths",
            _normalized_paths(self.justified_paths),
        )
        object.__setattr__(
            self,
            "denied_paths",
            _normalized_paths(self.denied_paths),
        )
        object.__setattr__(
            self,
            "contract_gap_paths",
            _normalized_paths(self.contract_gap_paths),
        )
        guidance = str(self.guidance_markdown or "").strip()
        if not guidance:
            raise ValueError("guidance_markdown is required")
        if len(guidance.encode("utf-8")) > 32_768:
            raise ValueError("guidance_markdown exceeds 32 KiB")
        object.__setattr__(self, "guidance_markdown", guidance)
        addendum = str(self.next_attempt_prompt_addendum or "").strip()
        if len(addendum.encode("utf-8")) > 16_384:
            raise ValueError("next_attempt_prompt_addendum exceeds 16 KiB")
        object.__setattr__(self, "next_attempt_prompt_addendum", addendum)
        object.__setattr__(
            self,
            "policy_version",
            str(self.policy_version or FAILURE_REVIEW_POLICY_VERSION).strip(),
        )
        if self.proof_authoritative or self.completion_authoritative:
            raise ValueError(
                "failure review cannot claim proof or completion authority"
            )

    @property
    def accepted(self) -> bool:
        return self.decision is FailureReviewDecision.ACCEPT

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FAILURE_REVIEW_SCHEMA,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "decision": self.decision.value,
            "reason_codes": list(self.reason_codes),
            "finding_codes": list(self.finding_codes),
            "expected_outputs": list(self.expected_outputs),
            "changed_paths": list(self.changed_paths),
            "missing_expected_outputs": list(self.missing_expected_outputs),
            "out_of_scope_paths": list(self.out_of_scope_paths),
            "justified_paths": list(self.justified_paths),
            "denied_paths": list(self.denied_paths),
            "contract_gap_paths": list(self.contract_gap_paths),
            "failed_commands": list(self.failed_commands),
            "guidance_markdown": self.guidance_markdown,
            "next_attempt_prompt_addendum": self.next_attempt_prompt_addendum,
            "policy_version": self.policy_version,
            "accepted": self.accepted,
            "proof_authoritative": False,
            "completion_authoritative": False,
        }

    def to_record(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["receipt_id"] = self.receipt_id
        canonical_json(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ImplementationFailureReviewReceipt":
        if not isinstance(payload, Mapping):
            raise ValueError("failure review receipt must be an object")
        if payload.get("schema") != FAILURE_REVIEW_SCHEMA:
            raise ValueError("unsupported failure review schema")
        result = cls(
            task_id=str(payload.get("task_id") or ""),
            attempt=int(payload.get("attempt") or 0),
            decision=FailureReviewDecision(str(payload.get("decision") or "")),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            finding_codes=tuple(payload.get("finding_codes") or ()),
            expected_outputs=tuple(payload.get("expected_outputs") or ()),
            changed_paths=tuple(payload.get("changed_paths") or ()),
            missing_expected_outputs=tuple(
                payload.get("missing_expected_outputs") or ()
            ),
            out_of_scope_paths=tuple(payload.get("out_of_scope_paths") or ()),
            justified_paths=tuple(payload.get("justified_paths") or ()),
            denied_paths=tuple(payload.get("denied_paths") or ()),
            contract_gap_paths=tuple(
                payload.get("contract_gap_paths") or ()
            ),
            failed_commands=tuple(payload.get("failed_commands") or ()),
            guidance_markdown=str(payload.get("guidance_markdown") or ""),
            next_attempt_prompt_addendum=str(
                payload.get("next_attempt_prompt_addendum") or ""
            ),
            policy_version=str(
                payload.get("policy_version") or FAILURE_REVIEW_POLICY_VERSION
            ),
        )
        if payload.get("receipt_id") not in (None, "", result.receipt_id):
            raise ValueError("failure review receipt identity is forged")
        if payload.get("accepted") not in (None, result.accepted):
            raise ValueError("failure review accepted flag is inconsistent")
        return result


def review_implementation_failure(
    *,
    task_id: str,
    attempt: int,
    expected_outputs: Sequence[str] = (),
    validation_result: Mapping[str, Any] | None = None,
    changed_paths: Sequence[str] = (),
    workspace_path: Path | str | None = None,
    log_excerpt: str = "",
    proposal_accepted: bool | None = None,
    scope_adjudication: Mapping[str, Any] | None = None,
    validation_commands: Sequence[str] = (),
    validation_environment_guidance: str = "",
) -> ImplementationFailureReviewReceipt:
    """Review one failed implementation/validation attempt.

    Acceptance is intentionally narrow: only a fully justified scope expansion
    with no hard-deny findings and no remaining command failures can flip the
    decision to ``accept``. Every other actionable case returns rescue guidance.
    """

    validation = _mapping(validation_result)
    environment_guidance = str(
        validation_environment_guidance or ""
    ).strip()
    if len(environment_guidance.encode("utf-8")) > 8_192:
        raise ValueError(
            "validation_environment_guidance exceeds 8 KiB"
        )
    finding_codes = _finding_codes_from_validation(validation)
    changed = _normalized_paths(
        (*changed_paths, *_changed_paths_from_validation(validation))
    )
    expected = _normalized_paths(expected_outputs)
    workspace = Path(workspace_path) if workspace_path else None
    missing = _missing_expected_outputs(
        expected_outputs=expected,
        changed_paths=changed,
        workspace_path=workspace,
    )
    scope = dict(scope_adjudication or {}) or _scope_projection(validation)
    justified = _normalized_paths(scope.get("justified_paths") or ())
    denied = _normalized_paths(scope.get("denied_paths") or ())
    contract_gap_paths = _scope_contract_gap_paths(
        scope,
        validation_commands=validation_commands,
        validation_result=validation,
    )
    scope_accepted = scope.get("accepted") is True
    out_of_scope = tuple(
        path
        for path in changed
        if expected and not _path_owned_by_expected(path, expected)
    )
    failed_commands = _failed_commands_from_validation(validation)
    reason = str(validation.get("reason") or "").strip()
    error = str(validation.get("error") or "").strip()
    proposal_failed = (
        proposal_accepted is False
        or error == "proposal_validation_failed"
        or reason in {"proposal_gate_failed", "proposal_validation_failed"}
        or (
            not validation.get("passed", True)
            and "proposal" in reason
        )
    )
    size_findings = _size_related_findings(finding_codes)

    reason_codes: list[str] = []
    hard_denies = tuple(
        code for code in finding_codes if code in _HARD_DENY_FINDING_CODES
    )
    if hard_denies:
        reason_codes.append(FailureReviewReason.HARD_DENY_FINDINGS.value)
    if finding_codes and set(finding_codes) <= _SCOPE_RELATED_FINDING_CODES:
        if scope_accepted and justified:
            reason_codes.append(
                FailureReviewReason.SCOPE_EXPANSION_JUSTIFIED.value
            )
        elif denied or out_of_scope:
            reason_codes.append(
                FailureReviewReason.SCOPE_EXPANSION_DENIED.value
            )
    elif out_of_scope or denied:
        reason_codes.append(FailureReviewReason.SCOPE_EXPANSION_DENIED.value)
    if contract_gap_paths:
        reason_codes.append(
            FailureReviewReason.TASK_SCOPE_CONTRACT_REVISION_REQUIRED.value
        )
    if missing and len(missing) == len(expected) and not changed:
        reason_codes.append(FailureReviewReason.EMPTY_OR_NO_CHANGE.value)
    elif missing:
        reason_codes.append(
            FailureReviewReason.INCOMPLETE_EXPECTED_OUTPUTS.value
        )
    if proposal_failed:
        reason_codes.append(FailureReviewReason.PROPOSAL_GATE_FAILED.value)
    if failed_commands or (
        validation.get("attempted") and validation.get("passed") is False
        and not proposal_failed
    ):
        reason_codes.append(
            FailureReviewReason.VALIDATION_COMMAND_FAILED.value
        )
    if _is_environment_failure(validation, log_excerpt=log_excerpt):
        reason_codes.append(
            FailureReviewReason.ENVIRONMENT_VALIDATION_UNAVAILABLE.value
        )
    # Large/undeclared refactor only when paths fall outside declared file or
    # directory ownership. Many files under a declared directory output (for
    # example tests/fixtures/...) are in-scope, not a refactor.
    if out_of_scope:
        reason_codes.append(
            FailureReviewReason.LARGE_OR_UNDECLARED_REFACTOR.value
        )
    if not reason_codes:
        if finding_codes or failed_commands or missing:
            reason_codes.append(
                FailureReviewReason.GENERIC_IMPLEMENTATION_FAILURE.value
            )
        else:
            reason_codes.append(
                FailureReviewReason.NO_ACTIONABLE_EVIDENCE.value
            )
    reason_codes = list(dict.fromkeys(reason_codes))

    # Fail-closed accept: only pure justified scope after successful
    # revalidation authority, with no command/environment/hard-deny issues and
    # no incomplete declared outputs.
    can_accept = (
        FailureReviewReason.SCOPE_EXPANSION_JUSTIFIED.value in reason_codes
        and scope_accepted
        and not hard_denies
        and not failed_commands
        and not missing
        and not _is_environment_failure(validation, log_excerpt=log_excerpt)
        and (
            proposal_accepted is True
            or (
                # Proposal will be revalidated by the caller after accept.
                set(finding_codes) <= _SCOPE_RELATED_FINDING_CODES
                and justified
            )
        )
    )
    if can_accept and proposal_failed and set(finding_codes) <= _SCOPE_RELATED_FINDING_CODES:
        decision = FailureReviewDecision.ACCEPT
    elif hard_denies:
        decision = FailureReviewDecision.REJECT
    else:
        decision = FailureReviewDecision.GUIDE_RESCUE

    guidance = "\n".join(
        _guidance_lines(
            decision=decision,
            reason_codes=reason_codes,
            finding_codes=finding_codes,
            missing_outputs=missing,
            out_of_scope_paths=out_of_scope,
            justified_paths=justified,
            denied_paths=denied,
            contract_gap_paths=contract_gap_paths,
            failed_commands=failed_commands,
            expected_outputs=expected,
            validation_environment_guidance=environment_guidance,
        )
    )
    addendum_lines = [
        "Prior attempt failure review "
        f"({decision.value}; reasons: {', '.join(reason_codes)}).",
    ]
    if size_findings:
        addendum_lines.append(
            "Proposal size gate failed ("
            + ", ".join(size_findings)
            + f"): keep patch ≤ {DEFAULT_PROPOSAL_MAX_PATCH_BYTES} bytes, "
            f"output ≤ {DEFAULT_PROPOSAL_MAX_OUTPUT_BYTES} bytes, "
            f"file ≤ {DEFAULT_PROPOSAL_MAX_FILE_BYTES} bytes. Prefer compact "
            "recipe/generator fixtures over bulk dumps; pytest green does not "
            "bypass admission."
        )
    if missing:
        addendum_lines.append(
            "Still required outputs: " + ", ".join(missing) + "."
        )
    ordinary_denied = tuple(
        path
        for path in dict.fromkeys((*denied, *out_of_scope))
        if path not in set(contract_gap_paths)
    )
    if ordinary_denied:
        addendum_lines.append(
            "Do not modify these out-of-scope paths: "
            + ", ".join(ordinary_denied)
            + "."
        )
    if contract_gap_paths:
        addendum_lines.append(
            "Task-scope contract revision required for: "
            + ", ".join(contract_gap_paths)
            + ". The proposal remains rejected; either revert each companion "
            "or have protected-board authority add its exact path to "
            "Outputs/Predicted before retrying."
        )
    if justified and decision is not FailureReviewDecision.ACCEPT:
        addendum_lines.append(
            "Import/test-linked companions previously justified: "
            + ", ".join(justified)
            + ". Prefer declaring them on the task board if they must stick."
        )
    if failed_commands:
        addendum_lines.append(
            "Re-run and fix: " + " | ".join(failed_commands[:4]) + "."
        )
    if environment_guidance:
        addendum_lines.append(
            "Authoritative validation environment: "
            + " ".join(environment_guidance.split())
        )
    addendum_lines.append(
        "Stay inside declared Outputs/Predicted files (files or directory "
        "trees); finish all expected outputs; avoid renames, submodule edits, "
        "and undeclared new modules."
    )

    return ImplementationFailureReviewReceipt(
        task_id=task_id,
        attempt=attempt,
        decision=decision,
        reason_codes=tuple(reason_codes),
        finding_codes=finding_codes,
        expected_outputs=expected,
        changed_paths=changed,
        missing_expected_outputs=missing,
        out_of_scope_paths=out_of_scope,
        justified_paths=justified,
        denied_paths=denied,
        contract_gap_paths=contract_gap_paths,
        failed_commands=failed_commands,
        guidance_markdown=guidance,
        next_attempt_prompt_addendum=" ".join(addendum_lines),
    )


def compact_failure_review(
    receipt: ImplementationFailureReviewReceipt,
) -> dict[str, Any]:
    """Bounded event/diagnostic projection of a failure review receipt."""

    payload = {
        "receipt_id": receipt.receipt_id,
        "task_id": receipt.task_id,
        "attempt": receipt.attempt,
        "decision": receipt.decision.value,
        "accepted": receipt.accepted,
        "reason_codes": list(receipt.reason_codes),
        "finding_codes": list(receipt.finding_codes),
        "missing_expected_outputs": list(receipt.missing_expected_outputs),
        "out_of_scope_paths": list(receipt.out_of_scope_paths),
        "justified_paths": list(receipt.justified_paths),
        "denied_paths": list(receipt.denied_paths),
        "contract_gap_paths": list(receipt.contract_gap_paths),
        "failed_commands": list(receipt.failed_commands),
        "next_attempt_prompt_addendum": receipt.next_attempt_prompt_addendum,
        "policy_version": receipt.policy_version,
        "proof_authoritative": False,
        "completion_authoritative": False,
    }
    canonical_json(payload)
    return payload


__all__ = [
    "FAILURE_REVIEW_POLICY_VERSION",
    "FAILURE_REVIEW_SCHEMA",
    "FailureReviewDecision",
    "FailureReviewReason",
    "ImplementationFailureReviewReceipt",
    "compact_failure_review",
    "review_implementation_failure",
]
