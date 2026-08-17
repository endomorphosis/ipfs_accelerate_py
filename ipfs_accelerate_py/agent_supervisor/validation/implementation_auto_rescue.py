"""Deterministic automatic rescue planning for failed implementation attempts.

When proposal admission or declared validation fails, the supervisor normally
returns ``guide_rescue`` guidance for the *next* attempt. Many recoverable
cases can instead be healed on the **same attempt**:

* declared outputs exist on disk but were never staged into the candidate
  patch (``empty_patch`` / ``expected_output_ignored_or_unstaged`` /
  ``patch_mismatch``);
* generated evidence artifacts are missing but a sibling ``materialize`` /
  ``write`` / ``generate`` CLI can be derived from a ``validate`` command;
* proposal admission already succeeded and only declared validation commands
  failed — a single focused provider repair pass on the preserved worktree can
  apply the failure-review addendum without discarding the candidate.

This module is pure planning. The implementation daemon owns workspace
mutations, provider invocation, and revalidation.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


AUTO_RESCUE_PLAN_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/implementation-auto-rescue-plan@2"
)
AUTO_RESCUE_POLICY_VERSION = "deterministic-auto-rescue-v2"

# Proposal findings that are often fixed by staging declared dirty/ignored
# outputs and re-running admission without a provider call.
STAGE_AND_REVALIDATE_FINDING_CODES = frozenset(
    {
        "empty_patch",
        "expected_output_ignored_or_unstaged",
        "patch_mismatch",
        "missing_required_field",
    }
)

STAGE_AND_REVALIDATE_REASON_CODES = frozenset(
    {
        "proposal_gate_failed",
        "empty_or_no_change",
        "incomplete_expected_outputs",
    }
)

MATERIALIZE_REASON_CODES = frozenset(
    {
        "incomplete_expected_outputs",
        "proposal_gate_failed",
        "empty_or_no_change",
    }
)

INLINE_PROVIDER_RESCUE_REASON_CODES = frozenset(
    {
        "validation_command_failed",
        "incomplete_expected_outputs",
        "proposal_gate_failed",
        "generic_implementation_failure",
        "empty_or_no_change",
    }
)

# Never auto-rescue hard security/policy failures.
HARD_DENY_REASON_CODES = frozenset(
    {
        "hard_deny_findings",
        "scope_expansion_denied",
        "task_scope_contract_revision_required",
    }
)

_VALIDATE_TOKEN_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])validate(?![A-Za-z0-9_])")
_MATERIALIZE_ALIASES = ("materialize", "write", "generate")


class AutoRescueAction(str, Enum):
    """Bounded automatic rescue actions the daemon may execute."""

    NONE = "none"
    MATERIALIZE_AND_STAGE = "materialize_and_stage"
    STAGE_AND_REVALIDATE = "stage_and_revalidate"
    STRIP_DENIED_HELPERS = "strip_denied_helpers"
    INLINE_PROVIDER_RESCUE = "inline_provider_rescue"


# Scratch helpers implementers add because they have no shell. These are
# never declared outputs; deleting them and revalidating unblocks the
# candidate without another provider attempt.
_HELPER_BASENAME_RE = re.compile(
    r"^(tmp-|_run_|_vgo|DELETE_ME)",
    re.IGNORECASE,
)


def is_undeclared_helper_path(
    path: str,
    expected_outputs: Sequence[str] = (),
) -> bool:
    """Return whether ``path`` is an undeclared self-check helper file."""

    normalized = str(path or "").replace("\\", "/").lstrip("./")
    if not normalized:
        return False
    expected = {
        str(item).replace("\\", "/").lstrip("./")
        for item in expected_outputs
        if str(item).strip()
    }
    if normalized in expected:
        return False
    name = normalized.rsplit("/", 1)[-1]
    if _HELPER_BASENAME_RE.match(name):
        return True
    lowered = name.lower()
    return "selfcheck" in lowered or lowered.startswith("tmp-")


@dataclass(frozen=True)
class AutoRescuePlan:
    """Content-free plan for one automatic rescue step."""

    action: AutoRescueAction
    reason: str
    finding_codes: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    failed_commands: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    materialize_commands: tuple[str, ...] = ()
    missing_expected_outputs: tuple[str, ...] = ()
    denied_helper_paths: tuple[str, ...] = ()
    max_provider_rescue_passes: int = 1

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": AUTO_RESCUE_PLAN_SCHEMA,
            "policy_version": AUTO_RESCUE_POLICY_VERSION,
            "action": self.action.value,
            "reason": self.reason,
            "finding_codes": list(self.finding_codes),
            "reason_codes": list(self.reason_codes),
            "failed_commands": list(self.failed_commands),
            "expected_outputs": list(self.expected_outputs),
            "materialize_commands": list(self.materialize_commands),
            "missing_expected_outputs": list(self.missing_expected_outputs),
            "denied_helper_paths": list(self.denied_helper_paths),
            "max_provider_rescue_passes": int(self.max_provider_rescue_passes),
        }


def _as_str_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        text = str(values).strip()
        return (text,) if text else ()
    if not isinstance(values, (list, tuple, set, frozenset)):
        return ()
    return tuple(
        sorted({str(item).strip() for item in values if str(item).strip()})
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _failure_review_projection(
    validation_result: Mapping[str, Any],
) -> Mapping[str, Any]:
    review = _mapping(validation_result.get("failure_review"))
    if review:
        return review
    nested = _mapping(validation_result.get("validation"))
    return _mapping(nested.get("failure_review"))


def derive_materialize_commands(
    validation_commands: Sequence[str],
) -> tuple[str, ...]:
    """Derive deterministic materialize/write commands from validate CLIs.

    Board validation lines often look like::

        python3 -m pkg.mod validate --workspace . --artifact data/...json

    When the implementer shipped a sibling ``materialize`` (or write/generate)
    subcommand, auto-rescue can invoke it without another model call.
    """

    derived: list[str] = []
    for raw in validation_commands:
        command = str(raw or "").strip()
        if not command or not _VALIDATE_TOKEN_RE.search(command):
            continue
        for alias in _MATERIALIZE_ALIASES:
            candidate = _VALIDATE_TOKEN_RE.sub(alias, command, count=1)
            if candidate != command and candidate not in derived:
                derived.append(candidate)
        # Also try token-level argv rewrite for robustness.
        try:
            argv = shlex.split(command)
        except ValueError:
            argv = []
        if argv:
            for index, token in enumerate(argv):
                if token.lower() != "validate":
                    continue
                for alias in _MATERIALIZE_ALIASES:
                    rewritten = list(argv)
                    rewritten[index] = alias
                    text = " ".join(shlex.quote(part) for part in rewritten)
                    if text not in derived:
                        derived.append(text)
                break
    return tuple(derived)


def plan_automatic_implementation_rescue(
    *,
    validation_result: Mapping[str, Any],
    expected_outputs: Sequence[str] = (),
    validation_commands: Sequence[str] = (),
    already_auto_rescued: bool = False,
    provider_rescue_passes_used: int = 0,
    stage_rescue_used: bool = False,
    materialize_rescue_used: bool = False,
    strip_helpers_used: bool = False,
    allow_provider_rescue: bool = True,
    expected_outputs_present_on_disk: bool = False,
    dirty_in_scope_paths: Sequence[str] = (),
    missing_expected_outputs: Sequence[str] = (),
) -> AutoRescuePlan:
    """Plan the next automatic rescue action for a failed attempt.

    Fail-closed defaults:

    * hard-deny / contract-gap reviews → no auto rescue
    * already exhausted automatic steps → none
    * materialize only when a validate→materialize rewrite exists
    * stage rescue only when staging can plausibly change the candidate
    * provider rescue only once per attempt, only for ``guide_rescue``
    """

    result = _mapping(validation_result)
    if result.get("passed") is True:
        return AutoRescuePlan(
            action=AutoRescueAction.NONE,
            reason="validation_already_passed",
        )
    if (
        already_auto_rescued
        and provider_rescue_passes_used >= 1
        and stage_rescue_used
        and materialize_rescue_used
        and strip_helpers_used
    ):
        return AutoRescuePlan(
            action=AutoRescueAction.NONE,
            reason="auto_rescue_budget_exhausted",
        )

    review = _failure_review_projection(result)
    decision = str(review.get("decision") or "").strip()
    reason_codes = _as_str_tuple(
        review.get("reason_codes") or result.get("reason_codes") or ()
    )
    finding_codes = _as_str_tuple(
        review.get("finding_codes")
        or result.get("finding_codes")
        or _mapping(result.get("proposal_gate")).get("finding_codes")
        or _mapping(result.get("proposal_validation")).get("finding_codes")
        or ()
    )
    proposal_validation = _mapping(result.get("proposal_validation"))
    findings = proposal_validation.get("findings") or ()
    if isinstance(findings, Sequence) and not isinstance(findings, (str, bytes)):
        for finding in findings:
            if not isinstance(finding, Mapping):
                continue
            code = finding.get("code")
            if isinstance(code, Mapping):
                code = code.get("value")
            text = str(code or "").strip()
            if text and text not in finding_codes:
                finding_codes = tuple(sorted({*finding_codes, text}))

    failed_commands = _as_str_tuple(
        review.get("failed_commands")
        or result.get("failed_commands")
        or ()
    )
    expected = _as_str_tuple(
        expected_outputs
        or review.get("expected_outputs")
        or result.get("expected_outputs")
        or ()
    )
    missing = _as_str_tuple(
        missing_expected_outputs
        or review.get("missing_expected_outputs")
        or result.get("missing_expected_outputs")
        or ()
    )
    dirty = _as_str_tuple(dirty_in_scope_paths)
    declared_validation_commands = _as_str_tuple(
        validation_commands
        or review.get("failed_commands")
        or result.get("failed_commands")
        or ()
    )
    # Prefer full failed command list for rewrites, but also accept any
    # validation command strings the caller supplies.
    materialize_commands = derive_materialize_commands(
        tuple(dict.fromkeys((*declared_validation_commands, *failed_commands)))
    )

    denied_paths = _as_str_tuple(
        review.get("denied_paths")
        or review.get("out_of_scope_paths")
        or _mapping(result.get("scope_adjudication")).get("denied_paths")
        or ()
    )
    helper_paths = tuple(
        path
        for path in denied_paths
        if is_undeclared_helper_path(path, expected)
    )
    if (
        not strip_helpers_used
        and helper_paths
        and set(helper_paths) == set(denied_paths)
        and expected_outputs_present_on_disk
        and (
            "scope_expansion_denied" in reason_codes
            or "path_outside_scope" in finding_codes
            or bool(denied_paths)
        )
    ):
        return AutoRescuePlan(
            action=AutoRescueAction.STRIP_DENIED_HELPERS,
            reason="strip_undeclared_helper_paths",
            finding_codes=finding_codes,
            reason_codes=reason_codes,
            failed_commands=failed_commands,
            expected_outputs=expected,
            missing_expected_outputs=missing,
            denied_helper_paths=helper_paths,
        )

    if decision and decision not in {"guide_rescue", ""}:
        if decision == "reject" or set(reason_codes) & HARD_DENY_REASON_CODES:
            return AutoRescuePlan(
                action=AutoRescueAction.NONE,
                reason="hard_deny_or_reject",
                finding_codes=finding_codes,
                reason_codes=reason_codes,
                failed_commands=failed_commands,
                expected_outputs=expected,
                missing_expected_outputs=missing,
            )

    if set(reason_codes) & HARD_DENY_REASON_CODES:
        return AutoRescuePlan(
            action=AutoRescueAction.NONE,
            reason="hard_deny_reason_codes",
            finding_codes=finding_codes,
            reason_codes=reason_codes,
            failed_commands=failed_commands,
            expected_outputs=expected,
            missing_expected_outputs=missing,
        )

    incomplete = bool(
        missing
        or "incomplete_expected_outputs" in reason_codes
        or "expected_output_ignored_or_unstaged" in finding_codes
    )
    proposal_failed = (
        str(result.get("reason") or "")
        in {"proposal_gate_failed", "proposal_validation_failed"}
        or str(result.get("error") or "") == "proposal_validation_failed"
        or "proposal_gate_failed" in reason_codes
        or bool(set(finding_codes) & STAGE_AND_REVALIDATE_FINDING_CODES)
    )

    # 1) Prefer materialize when expected generated artifacts are missing and a
    # validate CLI can be rewritten to materialize/write/generate.
    should_materialize = (
        not materialize_rescue_used
        and bool(materialize_commands)
        and (
            bool(missing)
            or (
                incomplete
                and not expected_outputs_present_on_disk
            )
            or (
                incomplete
                and "expected_output_ignored_or_unstaged" in finding_codes
                and not dirty
            )
        )
    )
    if should_materialize:
        return AutoRescuePlan(
            action=AutoRescueAction.MATERIALIZE_AND_STAGE,
            reason="materialize_missing_declared_artifacts",
            finding_codes=finding_codes,
            reason_codes=reason_codes,
            failed_commands=failed_commands,
            expected_outputs=expected,
            materialize_commands=materialize_commands,
            missing_expected_outputs=missing,
        )

    staging_plausible = bool(
        dirty
        or set(finding_codes) & STAGE_AND_REVALIDATE_FINDING_CODES
        or (
            expected_outputs_present_on_disk
            and (
                proposal_failed
                or incomplete
                or set(reason_codes) & STAGE_AND_REVALIDATE_REASON_CODES
            )
        )
    )
    # Prefer a cheap stage/revalidate before any provider call whenever dirty
    # declared outputs or ignored evidence may be missing from the patch.
    if (
        not stage_rescue_used
        and staging_plausible
        and (
            proposal_failed
            or incomplete
            or bool(dirty)
            or set(finding_codes) & STAGE_AND_REVALIDATE_FINDING_CODES
        )
    ):
        return AutoRescuePlan(
            action=AutoRescueAction.STAGE_AND_REVALIDATE,
            reason="stage_declared_outputs_and_revalidate",
            finding_codes=finding_codes,
            reason_codes=reason_codes,
            failed_commands=failed_commands,
            expected_outputs=expected,
            missing_expected_outputs=missing,
        )

    validation_failed = bool(
        failed_commands
        or "validation_command_failed" in reason_codes
        or str(result.get("error") or "") == "validation_command_failed"
        or str(result.get("reason") or "")
        in {
            "declared_validation_failed",
            "validation_failed",
            "validation_command_failed",
        }
    )
    # After staging/materialize, still allow one provider pass for residual
    # incomplete outputs or proposal-gate issues — not only command failures.
    residual_incomplete = bool(
        incomplete
        or proposal_failed
        or set(reason_codes) & INLINE_PROVIDER_RESCUE_REASON_CODES
    )
    if (
        allow_provider_rescue
        and provider_rescue_passes_used < 1
        and (
            validation_failed
            or residual_incomplete
            or stage_rescue_used
            or materialize_rescue_used
        )
        and (
            expected_outputs_present_on_disk
            or dirty
            or bool(expected)
            or bool(missing)
        )
        and (
            not decision
            or decision == "guide_rescue"
            or set(reason_codes) & INLINE_PROVIDER_RESCUE_REASON_CODES
        )
    ):
        return AutoRescuePlan(
            action=AutoRescueAction.INLINE_PROVIDER_RESCUE,
            reason=(
                "inline_provider_rescue_for_validation_failure"
                if validation_failed
                else "inline_provider_rescue_for_residual_incomplete_outputs"
            ),
            finding_codes=finding_codes,
            reason_codes=reason_codes,
            failed_commands=failed_commands,
            expected_outputs=expected,
            materialize_commands=materialize_commands,
            missing_expected_outputs=missing,
            max_provider_rescue_passes=1,
        )

    return AutoRescuePlan(
        action=AutoRescueAction.NONE,
        reason="no_automatic_rescue_path",
        finding_codes=finding_codes,
        reason_codes=reason_codes,
        failed_commands=failed_commands,
        expected_outputs=expected,
        materialize_commands=materialize_commands,
        missing_expected_outputs=missing,
    )


def build_inline_provider_rescue_prompt(
    *,
    base_prompt: str,
    validation_result: Mapping[str, Any],
    auto_rescue_plan: AutoRescuePlan | None = None,
) -> str:
    """Append deterministic rescue guidance onto an existing implementer prompt."""

    base = str(base_prompt or "").rstrip()
    review = _failure_review_projection(validation_result)
    addendum = str(
        validation_result.get("next_attempt_prompt_addendum")
        or review.get("next_attempt_prompt_addendum")
        or ""
    ).strip()
    failure_head = " ".join(
        str(validation_result.get("failure_head") or "").split()
    ).strip()
    failed_tests = _as_str_tuple(validation_result.get("failed_tests") or ())
    failed_commands = _as_str_tuple(
        review.get("failed_commands")
        or validation_result.get("failed_commands")
        or ()
    )
    missing = _as_str_tuple(
        review.get("missing_expected_outputs")
        or validation_result.get("missing_expected_outputs")
        or (auto_rescue_plan.missing_expected_outputs if auto_rescue_plan else ())
        or ()
    )
    materialize_commands = (
        auto_rescue_plan.materialize_commands if auto_rescue_plan is not None else ()
    )
    sections: list[str] = [
        "## Automatic same-attempt validation rescue",
        "The previous implementer pass left a candidate that failed admission "
        "or declared validation. Repair the existing worktree in place. Do not "
        "reset declared outputs that already look correct. Keep edits inside "
        "declared Outputs/Predicted files. Finish with green validation.",
        "If a generated evidence artifact is missing under data/, materialize "
        "it with the module CLI (`materialize`/`write`) then `git add` it, "
        "including force-add when the path is gitignored.",
    ]
    if auto_rescue_plan is not None and auto_rescue_plan.action is not AutoRescueAction.NONE:
        sections.append(
            f"Auto-rescue plan: `{auto_rescue_plan.action.value}` "
            f"({auto_rescue_plan.reason})."
        )
    if addendum:
        sections.append("### Prior failure review")
        sections.append(addendum)
    if missing:
        sections.append(
            "### Missing required outputs\n"
            + "\n".join(f"- `{path}`" for path in missing[:12])
        )
    if materialize_commands:
        sections.append(
            "### Suggested materialize commands\n"
            + "\n".join(f"- `{command}`" for command in materialize_commands[:6])
        )
    if failed_commands:
        sections.append(
            "### Failed commands\n"
            + "\n".join(f"- `{command}`" for command in failed_commands[:6])
        )
    if failed_tests:
        sections.append(
            "### Failed tests\n"
            + "\n".join(f"- `{node}`" for node in failed_tests[:12])
        )
    if failure_head:
        sections.append(
            "### Failure evidence\n```text\n" + failure_head[:1800] + "\n```"
        )
    rescue_block = "\n".join(sections).strip()
    if not base:
        return rescue_block + "\n"
    return f"{base}\n\n{rescue_block}\n"


__all__ = [
    "AUTO_RESCUE_PLAN_SCHEMA",
    "AUTO_RESCUE_POLICY_VERSION",
    "AutoRescueAction",
    "AutoRescuePlan",
    "INLINE_PROVIDER_RESCUE_REASON_CODES",
    "MATERIALIZE_REASON_CODES",
    "STAGE_AND_REVALIDATE_FINDING_CODES",
    "build_inline_provider_rescue_prompt",
    "derive_materialize_commands",
    "plan_automatic_implementation_rescue",
]
