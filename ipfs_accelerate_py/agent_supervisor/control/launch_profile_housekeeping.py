"""Automatic merge-base housekeeping for agent-supervisor launch profiles.

Voice-care and related programs pin ``merge_target_creation.expected_base_commit``
to a reviewed ``base_ref`` tip (usually ``origin/main``). After program merges
land on that tip, preflight fails until operators re-pin the profile and any
companion ``PINNED_BASE_COMMIT`` constants, and often until lagging agent merge
targets are fast-forwarded.

This module owns that housekeeping:

* Re-pin ``expected_base_commit`` when ``base_ref`` advances as a pure
  fast-forward of the previous pin.
* Optionally rewrite companion Python constants of the form
  ``PINNED_BASE_COMMIT = "<sha>"``.
* Optionally fast-forward a lagging merge-target branch onto the new pin when
  the branch tip is an ancestor of the new pin (no unique unmerged commits).
* Fail closed on history rewrite, divergence, or unsafe merge-target state.

No network access. Never force-pushes remotes. Never deletes branches.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

HOUSEKEEP_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/launch-profile-base-housekeep@1"
)
PLAN_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/launch-profile-base-housekeep-plan@1"
)

_OID_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)
_PINNED_CONST_RE = re.compile(
    r'^(?P<prefix>PINNED_BASE_COMMIT\s*=\s*")(?P<sha>[0-9a-fA-F]{7,40})(?P<suffix>")',
    re.MULTILINE,
)


class HousekeepError(RuntimeError):
    """Fail-closed housekeeping failure."""


@dataclass(frozen=True)
class MergeBaseHousekeepPlan:
    """Dry analysis of whether merge-base housekeeping is safe and needed."""

    schema: str = PLAN_SCHEMA
    action: str = "unchanged"
    safe: bool = True
    base_ref: str = "origin/main"
    old_pin: str = ""
    new_pin: str = ""
    merge_target_branch: str = ""
    merge_target_tip: str = ""
    merge_target_action: str = "none"
    reasons: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MergeBaseHousekeepResult:
    """Receipt for a housekeep attempt (dry-run or applied)."""

    schema: str = HOUSEKEEP_SCHEMA
    applied: bool = False
    dry_run: bool = False
    action: str = "unchanged"
    safe: bool = True
    base_ref: str = "origin/main"
    old_pin: str = ""
    new_pin: str = ""
    merge_target_branch: str = ""
    merge_target_action: str = "none"
    merge_target_tip_before: str = ""
    merge_target_tip_after: str = ""
    profile_path: str = ""
    companion_pin_paths_updated: list[str] = field(default_factory=list)
    receipt_path: str = ""
    reasons: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git(
    repo_root: Path,
    *args: str,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ("git", "-C", str(repo_root), *args),
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise HousekeepError(
            f"git {' '.join(args)} failed ({result.returncode}): {detail}"
        )
    return result


def _rev_parse(repo_root: Path, rev: str) -> str | None:
    result = _git(repo_root, "rev-parse", "--verify", f"{rev}^{{commit}}")
    if result.returncode != 0:
        return None
    tip = result.stdout.strip()
    if not _OID_RE.fullmatch(tip):
        return None
    return tip


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    result = _git(repo_root, "merge-base", "--is-ancestor", ancestor, descendant)
    return result.returncode == 0


def _ref_exists(repo_root: Path, branch: str) -> bool:
    result = _git(
        repo_root,
        "show-ref",
        "--verify",
        "--quiet",
        f"refs/heads/{branch}",
    )
    return result.returncode == 0


def _load_profile(profile_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HousekeepError(f"cannot read launch profile {profile_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HousekeepError(f"launch profile must be a JSON object: {profile_path}")
    return payload


def _creation_block(profile: Mapping[str, Any]) -> dict[str, Any]:
    creation = profile.get("merge_target_creation")
    if not isinstance(creation, dict):
        raise HousekeepError("profile merge_target_creation must be an object")
    return dict(creation)


def plan_merge_base_housekeeping(
    repo_root: Path | str,
    profile: Mapping[str, Any],
    *,
    update_merge_target: bool = True,
) -> MergeBaseHousekeepPlan:
    """Analyze whether the launch profile pin can be auto-reconciled."""

    root = Path(repo_root).resolve()
    reasons: list[str] = []
    errors: list[str] = []

    creation = profile.get("merge_target_creation")
    if not isinstance(creation, dict):
        return MergeBaseHousekeepPlan(
            action="fail",
            safe=False,
            errors=("profile merge_target_creation must be an object",),
        )

    base_ref = str(creation.get("base_ref") or "origin/main").strip() or "origin/main"
    old_pin = str(creation.get("expected_base_commit") or "").strip()
    merge_target = str(profile.get("merge_target_branch") or "").strip()

    if not old_pin or not _OID_RE.fullmatch(old_pin):
        return MergeBaseHousekeepPlan(
            action="fail",
            safe=False,
            base_ref=base_ref,
            old_pin=old_pin,
            merge_target_branch=merge_target,
            errors=("expected_base_commit is missing or not a commit oid",),
        )

    old_resolved = _rev_parse(root, old_pin)
    if old_resolved is None:
        return MergeBaseHousekeepPlan(
            action="fail",
            safe=False,
            base_ref=base_ref,
            old_pin=old_pin,
            merge_target_branch=merge_target,
            errors=(f"pinned base commit is unavailable: {old_pin}",),
        )
    old_pin = old_resolved

    new_pin = _rev_parse(root, base_ref)
    if new_pin is None:
        return MergeBaseHousekeepPlan(
            action="fail",
            safe=False,
            base_ref=base_ref,
            old_pin=old_pin,
            merge_target_branch=merge_target,
            errors=(f"base_ref is unavailable: {base_ref}",),
        )

    merge_target_tip = ""
    merge_target_action = "none"
    if merge_target and _ref_exists(root, merge_target):
        tip = _rev_parse(root, merge_target)
        if tip is None:
            errors.append(f"merge target tip is unavailable: {merge_target}")
        else:
            merge_target_tip = tip

    if old_pin == new_pin:
        # Pin already tracks base_ref. Still reconcile lagging merge target.
        if (
            update_merge_target
            and merge_target
            and merge_target_tip
            and merge_target_tip != new_pin
        ):
            if _is_ancestor(root, merge_target_tip, new_pin):
                merge_target_action = "fast_forward"
                reasons.append(
                    "merge target is behind base_ref tip; safe to fast-forward"
                )
                return MergeBaseHousekeepPlan(
                    action="ff_merge_target",
                    safe=True,
                    base_ref=base_ref,
                    old_pin=old_pin,
                    new_pin=new_pin,
                    merge_target_branch=merge_target,
                    merge_target_tip=merge_target_tip,
                    merge_target_action=merge_target_action,
                    reasons=tuple(reasons),
                )
            if _is_ancestor(root, new_pin, merge_target_tip):
                reasons.append(
                    "merge target is ahead of base_ref tip (agent work present); leave as-is"
                )
                return MergeBaseHousekeepPlan(
                    action="unchanged",
                    safe=True,
                    base_ref=base_ref,
                    old_pin=old_pin,
                    new_pin=new_pin,
                    merge_target_branch=merge_target,
                    merge_target_tip=merge_target_tip,
                    merge_target_action="leave_ahead",
                    reasons=tuple(reasons),
                )
            return MergeBaseHousekeepPlan(
                action="fail",
                safe=False,
                base_ref=base_ref,
                old_pin=old_pin,
                new_pin=new_pin,
                merge_target_branch=merge_target,
                merge_target_tip=merge_target_tip,
                merge_target_action="fail",
                errors=(
                    "merge target has diverged from base_ref; human reconciliation required",
                ),
            )
        reasons.append("expected_base_commit already matches base_ref tip")
        return MergeBaseHousekeepPlan(
            action="unchanged",
            safe=True,
            base_ref=base_ref,
            old_pin=old_pin,
            new_pin=new_pin,
            merge_target_branch=merge_target,
            merge_target_tip=merge_target_tip,
            merge_target_action="none",
            reasons=tuple(reasons),
        )

    # Pin drift: only auto-repin when base_ref is a pure FF of the old pin.
    if not _is_ancestor(root, old_pin, new_pin):
        return MergeBaseHousekeepPlan(
            action="fail",
            safe=False,
            base_ref=base_ref,
            old_pin=old_pin,
            new_pin=new_pin,
            merge_target_branch=merge_target,
            merge_target_tip=merge_target_tip,
            errors=(
                "base_ref tip is not a fast-forward of expected_base_commit; "
                "refusing automatic re-pin (possible history rewrite or divergent tip)",
            ),
        )

    reasons.append(
        f"base_ref advanced as fast-forward from {old_pin[:12]} to {new_pin[:12]}"
    )
    action = "repin"

    if update_merge_target and merge_target and merge_target_tip:
        if merge_target_tip == new_pin:
            merge_target_action = "unchanged"
            reasons.append("merge target already at new pin")
        elif _is_ancestor(root, merge_target_tip, new_pin):
            merge_target_action = "fast_forward"
            action = "repin_and_ff_merge_target"
            reasons.append(
                "merge target is behind new pin; will fast-forward after re-pin"
            )
        elif _is_ancestor(root, new_pin, merge_target_tip):
            merge_target_action = "leave_ahead"
            reasons.append(
                "merge target is ahead of new pin; re-pin only (agent work present)"
            )
        else:
            return MergeBaseHousekeepPlan(
                action="fail",
                safe=False,
                base_ref=base_ref,
                old_pin=old_pin,
                new_pin=new_pin,
                merge_target_branch=merge_target,
                merge_target_tip=merge_target_tip,
                merge_target_action="fail",
                reasons=tuple(reasons),
                errors=(
                    "merge target has diverged from base_ref; human reconciliation required",
                ),
            )
    elif update_merge_target and merge_target and not merge_target_tip:
        merge_target_action = "none"
        reasons.append("merge target branch absent; pin-only housekeeping")

    if errors:
        return MergeBaseHousekeepPlan(
            action="fail",
            safe=False,
            base_ref=base_ref,
            old_pin=old_pin,
            new_pin=new_pin,
            merge_target_branch=merge_target,
            merge_target_tip=merge_target_tip,
            merge_target_action=merge_target_action,
            reasons=tuple(reasons),
            errors=tuple(errors),
        )

    return MergeBaseHousekeepPlan(
        action=action,
        safe=True,
        base_ref=base_ref,
        old_pin=old_pin,
        new_pin=new_pin,
        merge_target_branch=merge_target,
        merge_target_tip=merge_target_tip,
        merge_target_action=merge_target_action,
        reasons=tuple(reasons),
    )


def update_pinned_base_constant(path: Path, new_sha: str) -> bool:
    """Rewrite ``PINNED_BASE_COMMIT = \"...\"`` in a Python module. Returns True if changed."""

    if not _OID_RE.fullmatch(new_sha):
        raise HousekeepError(f"invalid commit oid for constant update: {new_sha!r}")
    text = path.read_text(encoding="utf-8")
    match = _PINNED_CONST_RE.search(text)
    if match is None:
        raise HousekeepError(
            f"{path} does not contain a PINNED_BASE_COMMIT = \"...\" assignment"
        )
    if match.group("sha") == new_sha:
        return False
    updated, count = _PINNED_CONST_RE.subn(
        rf'\g<prefix>{new_sha}\g<suffix>',
        text,
        count=1,
    )
    if count != 1:
        raise HousekeepError(f"failed to rewrite PINNED_BASE_COMMIT in {path}")
    path.write_text(updated, encoding="utf-8")
    return True


_PROFILE_PIN_RE = re.compile(
    r'("expected_base_commit"\s*:\s*")([0-9a-fA-F]{7,40})(")',
)


def _write_profile_pin(
    profile_path: Path,
    profile: MutableMapping[str, Any],
    new_pin: str,
) -> None:
    """Update expected_base_commit in the profile JSON with a surgical rewrite.

    Prefer a single-field text replacement so surrounding key order and
    formatting stay stable for review. Fall back to a full JSON rewrite only
    when the field cannot be matched safely.
    """

    creation = _creation_block(profile)
    creation["expected_base_commit"] = new_pin
    profile["merge_target_creation"] = creation

    text = profile_path.read_text(encoding="utf-8")
    matches = list(_PROFILE_PIN_RE.finditer(text))
    if len(matches) == 1:
        updated, count = _PROFILE_PIN_RE.subn(
            rf"\g<1>{new_pin}\3",
            text,
            count=1,
        )
        if count == 1:
            profile_path.write_text(updated, encoding="utf-8")
            # Verify the loaded object now matches.
            reloaded = _load_profile(profile_path)
            reloaded_creation = _creation_block(reloaded)
            if reloaded_creation.get("expected_base_commit") == new_pin:
                return
    # Fallback: full rewrite.
    profile_path.write_text(
        json.dumps(profile, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _fast_forward_branch(
    repo_root: Path,
    branch: str,
    new_tip: str,
    *,
    expected_old_tip: str,
) -> str:
    """Compare-and-swap fast-forward of a local branch tip."""

    if not _is_ancestor(repo_root, expected_old_tip, new_tip):
        raise HousekeepError(
            f"refusing non-fast-forward update of {branch}: "
            f"{expected_old_tip[:12]} -> {new_tip[:12]}"
        )
    # update-ref A B C: set A to B only if current value is C
    result = _git(
        repo_root,
        "update-ref",
        f"refs/heads/{branch}",
        new_tip,
        expected_old_tip,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise HousekeepError(
            f"failed to fast-forward {branch} to {new_tip[:12]}: {detail}"
        )
    live = _rev_parse(repo_root, branch)
    if live != new_tip:
        raise HousekeepError(
            f"merge target tip mismatch after update: got {live}, expected {new_tip}"
        )
    return live


def apply_merge_base_housekeeping(
    repo_root: Path | str,
    profile_path: Path | str,
    *,
    write: bool = True,
    dry_run: bool = False,
    companion_pin_paths: Sequence[Path | str] = (),
    receipt_path: Path | str | None = None,
    update_merge_target: bool = True,
    require_clean_tree: bool = False,
    fail_on_unsafe: bool = True,
) -> dict[str, Any]:
    """Plan and optionally apply merge-base housekeeping for a launch profile.

    Parameters
    ----------
    write:
        When True (and not dry_run), persist profile / companion / branch updates.
    dry_run:
        Analyze and report without mutating anything.
    companion_pin_paths:
        Python files containing ``PINNED_BASE_COMMIT = \"<sha>\"`` to keep in sync.
    receipt_path:
        Optional JSON receipt destination.
    update_merge_target:
        When True, FF lagging merge targets that are pure ancestors of the new pin.
    require_clean_tree:
        When True, refuse writes if the recursive worktree is dirty.
    fail_on_unsafe:
        When True, raise :class:`HousekeepError` if the plan is not safe.
    """

    root = Path(repo_root).resolve()
    profile_path = Path(profile_path)
    if not profile_path.is_absolute():
        profile_path = (root / profile_path).resolve()
    else:
        profile_path = profile_path.resolve()

    profile = _load_profile(profile_path)
    plan = plan_merge_base_housekeeping(
        root,
        profile,
        update_merge_target=update_merge_target,
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    result = MergeBaseHousekeepResult(
        applied=False,
        dry_run=dry_run or not write,
        action=plan.action,
        safe=plan.safe,
        base_ref=plan.base_ref,
        old_pin=plan.old_pin,
        new_pin=plan.new_pin,
        merge_target_branch=plan.merge_target_branch,
        merge_target_action=plan.merge_target_action,
        merge_target_tip_before=plan.merge_target_tip,
        merge_target_tip_after=plan.merge_target_tip,
        profile_path=str(profile_path),
        reasons=list(plan.reasons),
        errors=list(plan.errors),
        generated_at=generated_at,
    )

    if not plan.safe:
        if fail_on_unsafe:
            raise HousekeepError(
                "merge-base housekeeping unsafe: " + "; ".join(plan.errors)
            )
        return result.to_dict()

    if plan.action == "unchanged":
        if receipt_path and write and not dry_run:
            receipt = Path(receipt_path)
            if not receipt.is_absolute():
                receipt = (root / receipt).resolve()
            result.receipt_path = str(receipt)
            _write_receipt(receipt, root, result)
        return result.to_dict()

    if dry_run or not write:
        result.reasons.append("dry_run_or_write_false: no mutations applied")
        return result.to_dict()

    if require_clean_tree:
        dirty = _git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            check=True,
        )
        if dirty.stdout.strip():
            raise HousekeepError(
                "refusing merge-base housekeeping on a dirty recursive tree"
            )

    # 1) Re-pin profile when needed.
    if plan.old_pin != plan.new_pin:
        _write_profile_pin(profile_path, profile, plan.new_pin)
        result.reasons.append(f"updated profile expected_base_commit -> {plan.new_pin}")

    # 2) Companion PINNED_BASE_COMMIT constants.
    for raw in companion_pin_paths:
        path = Path(raw)
        if not path.is_absolute():
            path = (root / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise HousekeepError(f"companion pin path missing: {path}")
        changed = update_pinned_base_constant(path, plan.new_pin)
        if changed:
            result.companion_pin_paths_updated.append(str(path))
            result.reasons.append(f"updated PINNED_BASE_COMMIT in {path}")

    # 3) Fast-forward lagging merge target.
    if plan.merge_target_action == "fast_forward" and plan.merge_target_branch:
        new_tip = _fast_forward_branch(
            root,
            plan.merge_target_branch,
            plan.new_pin,
            expected_old_tip=plan.merge_target_tip,
        )
        result.merge_target_tip_after = new_tip
        result.reasons.append(
            f"fast-forwarded {plan.merge_target_branch} -> {new_tip[:12]}"
        )

    result.applied = True
    if receipt_path:
        receipt = Path(receipt_path)
        if not receipt.is_absolute():
            receipt = (root / receipt).resolve()
        _write_receipt(receipt, root, result)
        result.receipt_path = str(receipt)

    return result.to_dict()


def _write_receipt(
    receipt_path: Path,
    repo_root: Path,
    result: MergeBaseHousekeepResult,
) -> None:
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["repo_root"] = str(repo_root)
    receipt_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def housekeep_launch_profile_if_needed(
    repo_root: Path | str,
    profile_path: Path | str,
    *,
    companion_pin_paths: Sequence[Path | str] = (),
    receipt_path: Path | str | None = None,
    update_merge_target: bool = True,
    enabled: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Convenience entry for preflight: auto-housekeep when enabled.

    When ``enabled`` is False, returns a skipped receipt without mutation.
    """

    if not enabled:
        return {
            "schema": HOUSEKEEP_SCHEMA,
            "applied": False,
            "action": "skipped",
            "safe": True,
            "reasons": ["housekeeping disabled by caller"],
            "errors": [],
        }
    return apply_merge_base_housekeeping(
        repo_root,
        profile_path,
        write=True,
        dry_run=dry_run,
        companion_pin_paths=companion_pin_paths,
        receipt_path=receipt_path,
        update_merge_target=update_merge_target,
        fail_on_unsafe=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically re-pin launch profile expected_base_commit and "
            "fast-forward lagging merge targets when safe."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Git repository root (default: cwd)",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Path to supervisor launch profile JSON",
    )
    parser.add_argument(
        "--companion-pin",
        action="append",
        default=[],
        dest="companion_pins",
        help="Python file with PINNED_BASE_COMMIT to rewrite (repeatable)",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Optional receipt JSON path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze only; do not write",
    )
    parser.add_argument(
        "--no-merge-target",
        action="store_true",
        help="Do not fast-forward the merge target branch",
    )
    parser.add_argument(
        "--require-clean-tree",
        action="store_true",
        help="Refuse writes when the worktree is dirty",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit receipt JSON to stdout",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        receipt = apply_merge_base_housekeeping(
            args.repo_root,
            args.profile,
            write=not args.dry_run,
            dry_run=args.dry_run,
            companion_pin_paths=args.companion_pins,
            receipt_path=args.receipt,
            update_merge_target=not args.no_merge_target,
            require_clean_tree=args.require_clean_tree,
            fail_on_unsafe=True,
        )
    except HousekeepError as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "schema": HOUSEKEEP_SCHEMA,
                        "applied": False,
                        "safe": False,
                        "errors": [str(exc)],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(f"launch-profile housekeep FAILED: {exc}", file=__import__("sys").stderr)
        return 1

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        action = receipt.get("action")
        applied = receipt.get("applied")
        print(
            f"launch-profile housekeep OK: action={action} applied={applied} "
            f"pin={str(receipt.get('new_pin') or '')[:12]}"
        )
        for reason in receipt.get("reasons") or []:
            print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
