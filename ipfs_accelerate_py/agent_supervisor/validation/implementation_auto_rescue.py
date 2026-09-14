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
  apply the failure-review addendum without discarding the candidate;
* a rejected credential-shaped literal occurs only in task-owned Python test
  source — one changed-strategy pass may replace it, but the hard rejection is
  preserved and the replacement must pass the full gates afresh.

This module is pure planning. The implementation daemon owns workspace
mutations, provider invocation, and revalidation.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shlex
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

AUTO_RESCUE_PLAN_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/implementation-auto-rescue-plan@3"
)
AUTO_RESCUE_POLICY_VERSION = "deterministic-auto-rescue-v3"

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

# One hard-deny has a bounded *repair* route.  The failed proposal remains
# rejected: a provider may make one changed-strategy pass only when the live
# proposal gate proved that every offending value is a credential-shaped
# assignment in task-owned Python test source.  The replacement proposal must
# then traverse the ordinary proposal and validation gates from the beginning.
SCOPED_TEST_SECRET_FINDING_CODES = frozenset({"secret_change_forbidden"})
SCOPED_TEST_SECRET_REASON_CODES = frozenset(
    {"hard_deny_findings", "proposal_gate_failed"}
)
SCOPED_TEST_SEMANTIC_INVENTORY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "scoped-test-semantic-inventory@1"
)

_CREDENTIAL_BINDING_NAMES = frozenset(
    {
        "apikey",
        "accesstoken",
        "authtoken",
        "refreshtoken",
        "clientsecret",
        "password",
        "passwd",
    }
)
_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_VALIDATE_TOKEN_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])validate(?![A-Za-z0-9_])")
_MATERIALIZE_ALIASES = ("materialize", "write", "generate")


class AutoRescueAction(str, Enum):
    """Bounded automatic rescue actions the daemon may execute."""

    NONE = "none"
    MATERIALIZE_AND_STAGE = "materialize_and_stage"
    STAGE_AND_REVALIDATE = "stage_and_revalidate"
    STRIP_DENIED_HELPERS = "strip_denied_helpers"
    INLINE_PROVIDER_RESCUE = "inline_provider_rescue"
    REMEDIATE_SCOPED_TEST_SECRET = "remediate_scoped_test_secret"


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
    remediation_paths: tuple[str, ...] = ()
    prior_proposal_id: str = ""
    prior_receipt_id: str = ""
    prior_test_semantic_inventory: Mapping[str, Any] | None = None
    accepted_effect_count: int | None = None
    merge_effect_count: int | None = None
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
            "remediation_paths": list(self.remediation_paths),
            "prior_proposal_id": self.prior_proposal_id,
            "prior_receipt_id": self.prior_receipt_id,
            "prior_test_semantic_inventory": (
                json.loads(
                    json.dumps(
                        self.prior_test_semantic_inventory,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
                if self.prior_test_semantic_inventory is not None
                else None
            ),
            "accepted_effect_count": self.accepted_effect_count,
            "merge_effect_count": self.merge_effect_count,
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


def _record_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _credential_binding_name(node: ast.AST) -> str:
    raw = ""
    if isinstance(node, ast.Name):
        raw = node.id
    elif isinstance(node, ast.Attribute):
        raw = node.attr
    elif isinstance(node, ast.Subscript):
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            raw = key.value
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        raw = node.value
    return re.sub(r"[^a-z0-9]", "", raw.casefold())


def _ast_children_with_paths(
    node: ast.AST,
    path: str = "root",
) -> Sequence[tuple[str, ast.AST]]:
    children: list[tuple[str, ast.AST]] = []
    for field_name, value in ast.iter_fields(node):
        field_path = f"{path}.{field_name}"
        if isinstance(value, ast.AST):
            children.append((field_path, value))
        elif isinstance(value, list):
            children.extend(
                (f"{field_path}[{index}]", item)
                for index, item in enumerate(value)
                if isinstance(item, ast.AST)
            )
    return children


def _walk_ast_with_paths(
    node: ast.AST,
    path: str = "root",
) -> Sequence[tuple[str, ast.AST]]:
    result: list[tuple[str, ast.AST]] = [(path, node)]
    for child_path, child in _ast_children_with_paths(node, path):
        result.extend(_walk_ast_with_paths(child, child_path))
    return result


def _introduced_after_lines(before: str | None, after: str) -> frozenset[int]:
    if before is None:
        return frozenset(range(1, len(after.splitlines()) + 1))
    baseline_lines = Counter(before.splitlines(keepends=True))
    introduced: set[int] = set()
    for line_number, line in enumerate(after.splitlines(keepends=True), 1):
        if baseline_lines[line]:
            baseline_lines[line] -= 1
        else:
            introduced.add(line_number)
    return frozenset(introduced)


def _value_overlaps_introduced_lines(
    value: ast.AST,
    introduced_lines: frozenset[int],
) -> bool:
    start = int(getattr(value, "lineno", 0) or 0)
    end = int(getattr(value, "end_lineno", start) or start)
    return start > 0 and any(line in introduced_lines for line in range(start, end + 1))


def _credential_value_slots(
    tree: ast.AST,
    *,
    source: str,
    introduced_lines: frozenset[int],
    concrete_secret_value: Any,
) -> tuple[str, ...]:
    """Locate only introduced, concrete credential-value AST slots.

    The returned structural locators contain no source values.  They allow a
    replacement proposal to vary only the rejected value expression while a
    normalized whole-module fingerprint protects all surrounding semantics.
    """

    slots: set[str] = set()

    def eligible(value: ast.AST) -> bool:
        if not _value_overlaps_introduced_lines(value, introduced_lines):
            return False
        segment = ast.get_source_segment(source, value)
        return bool(segment and concrete_secret_value(segment))

    for path, node in _walk_ast_with_paths(tree):
        if isinstance(node, ast.Assign):
            if any(
                _credential_binding_name(target) in _CREDENTIAL_BINDING_NAMES
                for target in node.targets
            ) and eligible(node.value):
                slots.add(f"{path}.value")
        elif isinstance(node, ast.AnnAssign):
            if (
                node.value is not None
                and _credential_binding_name(node.target)
                in _CREDENTIAL_BINDING_NAMES
                and eligible(node.value)
            ):
                slots.add(f"{path}.value")
        elif isinstance(node, ast.NamedExpr):
            if (
                _credential_binding_name(node.target)
                in _CREDENTIAL_BINDING_NAMES
                and eligible(node.value)
            ):
                slots.add(f"{path}.value")
        elif isinstance(node, ast.keyword):
            normalized = re.sub(
                r"[^a-z0-9]", "", str(node.arg or "").casefold()
            )
            if normalized in _CREDENTIAL_BINDING_NAMES and eligible(node.value):
                slots.add(f"{path}.value")
        elif isinstance(node, ast.Dict):
            for index, (key, value) in enumerate(
                zip(node.keys, node.values, strict=True)
            ):
                if (
                    key is not None
                    and _credential_binding_name(key)
                    in _CREDENTIAL_BINDING_NAMES
                    and eligible(value)
                ):
                    slots.add(f"{path}.values[{index}]")
    return tuple(sorted(slots))


def _normalize_ast_value_slots(
    tree: ast.AST,
    value_slots: Sequence[str],
) -> bool:
    remaining = set(value_slots)

    def visit(node: ast.AST, path: str = "root") -> None:
        for field_name, value in ast.iter_fields(node):
            field_path = f"{path}.{field_name}"
            if isinstance(value, ast.AST):
                if field_path in remaining:
                    setattr(
                        node,
                        field_name,
                        ast.Name(id="__SCOPED_SECRET_VALUE__", ctx=ast.Load()),
                    )
                    remaining.remove(field_path)
                else:
                    visit(value, field_path)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if not isinstance(item, ast.AST):
                        continue
                    item_path = f"{field_path}[{index}]"
                    if item_path in remaining:
                        value[index] = ast.Name(
                            id="__SCOPED_SECRET_VALUE__", ctx=ast.Load()
                        )
                        remaining.remove(item_path)
                    else:
                        visit(item, item_path)

    visit(tree)
    return not remaining


class _TestSemanticVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self._owners: list[str] = []
        self.tests: list[dict[str, str]] = []
        self.assertions: list[dict[str, str]] = []

    def _owner(self) -> str:
        return ".".join(self._owners) or "<module>"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._owners.append(node.name)
        self.generic_visit(node)
        self._owners.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._owners.append(node.name)
        if node.name.startswith("test"):
            self.tests.append(
                {
                    "qualified_name": self._owner(),
                    "kind": (
                        "async_function"
                        if isinstance(node, ast.AsyncFunctionDef)
                        else "function"
                    ),
                    "semantic_id": _record_digest(ast.dump(node, include_attributes=False)),
                }
            )
        self.generic_visit(node)
        self._owners.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(  # noqa: N802
        self, node: ast.AsyncFunctionDef
    ) -> None:
        self._visit_function(node)

    def visit_Assert(self, node: ast.Assert) -> None:  # noqa: N802
        self.assertions.append(
            {
                "owner": self._owner(),
                "kind": "assert",
                "semantic_id": _record_digest(ast.dump(node, include_attributes=False)),
            }
        )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        kind = ""
        function = node.func
        if isinstance(function, ast.Attribute):
            if function.attr.startswith("assert"):
                kind = "assert_call"
            elif (
                isinstance(function.value, ast.Name)
                and function.value.id == "pytest"
                and function.attr in {"raises", "warns"}
            ):
                kind = f"pytest_{function.attr}"
        if kind:
            self.assertions.append(
                {
                    "owner": self._owner(),
                    "kind": kind,
                    "semantic_id": _record_digest(
                        ast.dump(node, include_attributes=False)
                    ),
                }
            )
        self.generic_visit(node)


def _test_semantic_path_record(
    *,
    path: str,
    source: str,
    value_slots: Sequence[str],
) -> dict[str, Any] | None:
    try:
        tree = ast.parse(source, filename=path)
    except (SyntaxError, ValueError, TypeError):
        return None
    if not _normalize_ast_value_slots(tree, value_slots):
        return None
    visitor = _TestSemanticVisitor()
    visitor.visit(tree)
    tests = visitor.tests
    assertions = visitor.assertions
    return {
        "path": path,
        "secret_value_slots": list(value_slots),
        "module_semantic_id": _record_digest(
            ast.dump(tree, include_attributes=False)
        ),
        "test_count": len(tests),
        "tests": tests,
        "assertion_count": len(assertions),
        "assertions": assertions,
    }


def build_scoped_test_semantic_inventory(
    proposal: Any,
    remediation_paths: Sequence[str],
    *,
    concrete_secret_value: Any,
) -> dict[str, Any] | None:
    """Build a secret-free semantic inventory from a live rejected proposal."""

    paths = _as_str_tuple(remediation_paths)
    entries = tuple(getattr(proposal, "candidate_diff", ()) or ())
    records: list[dict[str, Any]] = []
    for path in paths:
        matching = [
            entry
            for entry in entries
            if str(
                getattr(entry, "new_path", "")
                or getattr(entry, "old_path", "")
                or ""
            ).strip()
            == path
        ]
        if len(matching) != 1:
            return None
        entry = matching[0]
        source = getattr(entry, "after_source", None)
        if not isinstance(source, str):
            return None
        before = getattr(entry, "before_source", None)
        if before is not None and not isinstance(before, str):
            return None
        try:
            tree = ast.parse(source, filename=path)
        except (SyntaxError, ValueError, TypeError):
            return None
        value_slots = _credential_value_slots(
            tree,
            source=source,
            introduced_lines=_introduced_after_lines(before, source),
            concrete_secret_value=concrete_secret_value,
        )
        if not value_slots:
            return None
        record = _test_semantic_path_record(
            path=path,
            source=source,
            value_slots=value_slots,
        )
        if record is None:
            return None
        records.append(record)
    if len(records) != len(paths) or not records:
        return None
    payload = {
        "schema": SCOPED_TEST_SEMANTIC_INVENTORY_SCHEMA,
        "paths": records,
    }
    return {**payload, "inventory_id": _record_digest(payload)}


def valid_scoped_test_semantic_inventory(
    inventory: Any,
    remediation_paths: Sequence[str],
) -> bool:
    """Validate the closed, content-free inventory projection."""

    if not isinstance(inventory, Mapping) or set(inventory) != {
        "schema",
        "paths",
        "inventory_id",
    }:
        return False
    if inventory.get("schema") != SCOPED_TEST_SEMANTIC_INVENTORY_SCHEMA:
        return False
    inventory_id = inventory.get("inventory_id")
    if not isinstance(inventory_id, str) or _HEX_SHA256_RE.fullmatch(inventory_id) is None:
        return False
    raw_records = inventory.get("paths")
    if not isinstance(raw_records, list):
        return False
    records: list[Mapping[str, Any]] = []
    expected_paths = _as_str_tuple(remediation_paths)
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "path",
            "secret_value_slots",
            "module_semantic_id",
            "test_count",
            "tests",
            "assertion_count",
            "assertions",
        }:
            return False
        path = raw_record.get("path")
        slots = raw_record.get("secret_value_slots")
        if not isinstance(path, str) or not path:
            return False
        if (
            not isinstance(slots, list)
            or not slots
            or slots != sorted(set(slots))
            or any(not isinstance(slot, str) or not slot.startswith("root.") for slot in slots)
        ):
            return False
        if _HEX_SHA256_RE.fullmatch(str(raw_record.get("module_semantic_id") or "")) is None:
            return False
        for count_name, item_name, allowed_fields in (
            ("test_count", "tests", {"qualified_name", "kind", "semantic_id"}),
            ("assertion_count", "assertions", {"owner", "kind", "semantic_id"}),
        ):
            items = raw_record.get(item_name)
            count = raw_record.get(count_name)
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
                or not isinstance(items, list)
                or count != len(items)
            ):
                return False
            for item in items:
                if not isinstance(item, Mapping) or set(item) != allowed_fields:
                    return False
                if any(not isinstance(value, str) or not value for value in item.values()):
                    return False
                if _HEX_SHA256_RE.fullmatch(str(item.get("semantic_id") or "")) is None:
                    return False
        records.append(raw_record)
    if tuple(record.get("path") for record in records) != expected_paths:
        return False
    payload = {
        "schema": inventory["schema"],
        "paths": raw_records,
    }
    return _record_digest(payload) == inventory_id


def scoped_test_semantics_preserved(
    inventory: Any,
    proposal: Any,
    remediation_paths: Sequence[str],
) -> tuple[bool, str]:
    """Compare a replacement's live source with the rejected inventory."""

    if not valid_scoped_test_semantic_inventory(inventory, remediation_paths):
        return False, "prior_inventory_invalid"
    entries = tuple(getattr(proposal, "candidate_diff", ()) or ())
    prior_by_path = {
        str(record["path"]): record for record in inventory["paths"]
    }
    for path in _as_str_tuple(remediation_paths):
        matching = [
            entry
            for entry in entries
            if str(
                getattr(entry, "new_path", "")
                or getattr(entry, "old_path", "")
                or ""
            ).strip()
            == path
        ]
        if len(matching) != 1:
            return False, "replacement_path_binding_invalid"
        source = getattr(matching[0], "after_source", None)
        if not isinstance(source, str):
            return False, "replacement_source_unavailable"
        prior = prior_by_path[path]
        replacement = _test_semantic_path_record(
            path=path,
            source=source,
            value_slots=prior["secret_value_slots"],
        )
        if replacement is None:
            return False, "replacement_inventory_unavailable"
        if replacement != prior:
            return False, "replacement_test_semantics_changed"
    return True, "preserved"


def _failure_review_projection(
    validation_result: Mapping[str, Any],
) -> Mapping[str, Any]:
    review = _mapping(validation_result.get("failure_review"))
    if review:
        return review
    nested = _mapping(validation_result.get("validation"))
    return _mapping(nested.get("failure_review"))


def _scoped_test_secret_remediation_evidence(
    result: Mapping[str, Any],
    *,
    decision: str,
    reason_codes: Sequence[str],
    finding_codes: Sequence[str],
    provider_rescue_passes_used: int,
    accepted_effect_count: int | None,
    merge_effect_count: int | None,
) -> tuple[tuple[str, ...], str, str, Mapping[str, Any]] | None:
    """Return content-free evidence for one hard-deny remediation pass.

    This does not override or accept the secret finding.  It only recognizes
    an exact, independently produced scope examination so the existing
    provider can replace a credential-shaped *test literal*.  Missing or
    malformed evidence fails closed.
    """

    if (
        decision != "reject"
        or provider_rescue_passes_used != 0
        or accepted_effect_count != 0
        or merge_effect_count != 0
        or set(finding_codes) != SCOPED_TEST_SECRET_FINDING_CODES
        or "hard_deny_findings" not in reason_codes
        or not set(reason_codes).issubset(SCOPED_TEST_SECRET_REASON_CODES)
    ):
        return None

    examination = _mapping(result.get("secret_change_scope_examination"))
    examined_paths = _as_str_tuple(examination.get("examined_paths"))
    in_scope_paths = _as_str_tuple(examination.get("in_scope_paths"))
    out_of_scope_paths = _as_str_tuple(examination.get("out_of_scope_paths"))
    scoped_test_paths = _as_str_tuple(
        examination.get("scoped_python_test_source_paths")
    )
    if (
        examination.get("finding_code") != "secret_change_forbidden"
        or examination.get("scope_classification") != "in_scope"
        or examination.get("secret_policy_overridden") is not False
        or examination.get("candidate_diff_path_coverage_complete") is not True
        or examination.get("private_key_material_absence_verified") is not True
        or examination.get("credential_assignment_only") is not True
        or not examined_paths
        or out_of_scope_paths
        or set(in_scope_paths) != set(examined_paths)
        or set(scoped_test_paths) != set(examined_paths)
    ):
        return None

    semantic_inventory = examination.get("test_semantic_inventory")
    if not valid_scoped_test_semantic_inventory(
        semantic_inventory,
        examined_paths,
    ):
        return None

    proposal_gate = _mapping(result.get("proposal_gate"))
    proposal_id = str(
        examination.get("proposal_id") or proposal_gate.get("proposal_id") or ""
    ).strip()
    receipt_id = str(proposal_gate.get("receipt_id") or "").strip()
    if (
        not proposal_id
        or not receipt_id
        or proposal_gate.get("accepted") is not False
        or (
            proposal_gate.get("proposal_id")
            and str(proposal_gate.get("proposal_id")) != proposal_id
        )
    ):
        return None
    return examined_paths, proposal_id, receipt_id, semantic_inventory


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
    accepted_effect_count: int | None = None,
    merge_effect_count: int | None = None,
) -> AutoRescuePlan:
    """Plan the next automatic rescue action for a failed attempt.

    Fail-closed defaults:

    * hard-deny / contract-gap reviews → no acceptance; only the exact scoped
      test-secret remediation predicate may receive one repair pass
    * already exhausted automatic steps → none
    * materialize only when a validate→materialize rewrite exists
    * stage rescue only when staging can plausibly change the candidate
    * provider rescue only once per attempt, for ``guide_rescue`` or the exact
      scoped test-secret remediation predicate
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
    scoped_secret_evidence = _scoped_test_secret_remediation_evidence(
        result,
        decision=decision,
        reason_codes=reason_codes,
        finding_codes=finding_codes,
        provider_rescue_passes_used=provider_rescue_passes_used,
        accepted_effect_count=accepted_effect_count,
        merge_effect_count=merge_effect_count,
    )
    if allow_provider_rescue and scoped_secret_evidence is not None:
        (
            remediation_paths,
            prior_proposal_id,
            prior_receipt_id,
            prior_test_semantic_inventory,
        ) = scoped_secret_evidence
        return AutoRescuePlan(
            action=AutoRescueAction.REMEDIATE_SCOPED_TEST_SECRET,
            reason="remediate_rejected_scoped_test_secret_literal",
            finding_codes=finding_codes,
            reason_codes=reason_codes,
            failed_commands=failed_commands,
            expected_outputs=expected,
            missing_expected_outputs=missing,
            remediation_paths=remediation_paths,
            prior_proposal_id=prior_proposal_id,
            prior_receipt_id=prior_receipt_id,
            prior_test_semantic_inventory=prior_test_semantic_inventory,
            accepted_effect_count=accepted_effect_count,
            merge_effect_count=merge_effect_count,
            max_provider_rescue_passes=1,
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
    if (
        auto_rescue_plan is not None
        and auto_rescue_plan.action
        is AutoRescueAction.REMEDIATE_SCOPED_TEST_SECRET
    ):
        sections.extend(
            (
                "### Scoped test-secret remediation",
                "The prior proposal remains rejected. Remove every "
                "credential-shaped literal from the identified task-owned "
                "Python test source. Preserve the test's security assertion; "
                "do not weaken the test, scanner, proposal policy, or "
                "validation. When a redaction canary is necessary, use the "
                "proposal gate's existing explicit synthetic vocabulary (for "
                "example `test-only-password-value` or "
                "`should-not-appear`). Do not introduce production secrets, "
                "private-key material, or new allowlist exceptions.",
                "The replacement must be a different proposal and must pass "
                "the complete proposal gate and declared test plan afresh.",
            )
        )
        if auto_rescue_plan.remediation_paths:
            sections.append(
                "### Remediation paths\n"
                + "\n".join(
                    f"- `{path}`"
                    for path in auto_rescue_plan.remediation_paths[:12]
                )
            )
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
    "SCOPED_TEST_SECRET_FINDING_CODES",
    "SCOPED_TEST_SECRET_REASON_CODES",
    "STAGE_AND_REVALIDATE_FINDING_CODES",
    "build_inline_provider_rescue_prompt",
    "derive_materialize_commands",
    "plan_automatic_implementation_rescue",
]
