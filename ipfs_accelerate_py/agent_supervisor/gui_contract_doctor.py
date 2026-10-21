"""Deterministic click, DOM, and backend contract repair.

The repair uses three existing authorities and does not call a model:

* ``ipfs_datasets_py.logic.ui_ux_ir`` adapts the rendered census into a
  DOM/ARIA document. The adapter does not execute markup.
* ``apply_program_world_repair_operator`` may sketch one unique byte
  replacement. A sketch is proposal-only. This module writes the alias
  table and does not write application source.
* The deterministic doctor stays in its report-only status. A failed or
  missing doctor receipt does not authorize a repair.

A matrix selector becomes a DOM alias only when one census control has that
accessible name, or when window chrome is the matching ``.window-control``
selector. A posted tool id is recorded as a backend match only when it is
exactly one preferred tool id on one binding. ``mcp++`` path strings are not
rewritten onto ``ipfs_cat`` or any other tool.

The click work list is one row per rendered-census control. Template selectors
stay loss receipts and are not work-list rows. A binding attribute that is
absent from the desktop source and from the live catalog, and a ``#id`` or
``data-action`` selector whose token is absent from that source, are the same
kind of loss. Apps on the local live-click target never receive ``invoke`` or
``unlisted``: a catalog gateway binding there is ``unsupported`` with reason
``local_family_zero_call``, and a carried ``external-provider`` row stays
``external-provider``. The only declared read alias is
``mcp++:ipfs_kit_py/artifacts/get`` to ``ipfs_cat`` on a single
``.retrieve_content`` binding. ``finite_adapter`` sketches that span, or one
``callMcpTool`` function when the literal sits outside ``JSON.stringify``,
and does not write application source.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

CONTRACT_SCHEMA = "swissknife.click-dom-backend-contract.v1"
CONTRACT_REL = (
    "swissknife/test-results/virtual-desktop-ipfs-mcp-orb/vog/"
    "click-dom-backend-contract.json"
)
WORKLIST_SCHEMA = "swissknife.click-worklist.v1"
WORKLIST_REL = (
    "swissknife/test-results/virtual-desktop-ipfs-mcp-orb/vog/"
    "click-worklist.json"
)
MATRIX_REL = "swissknife/test-results/virtual-desktop-ipfs-mcp-orb/control-tool-matrix.json"
_CHROME_SELECTORS = {
    ".window-control.minimize",
    ".window-control.maximize",
    ".window-control.close",
}
_CARRIED_DISPOSITIONS = {
    "invoke",
    "confirm-then-invoke",
    "deny",
    "unlisted",
    "local-only",
    "external-provider",
    "unsupported",
}
# VOG-046 rejects every invoke and unlisted row. These ids match
# LIVE_CLICK_TARGETS.local.apps. A catalog gateway binding on this set is
# unsupported. A carried external-provider row is left as it is.
_LOCAL_ZERO_CALL_APPS = frozenset({
    "calculator",
    "clock",
    "terminal",
    "glasses-preview",
    "settings",
    "todo",
    "calendar",
    "cron",
    "task-manager",
    "music-studio",
    "music-studio-unified",
    "strudel",
    "strudel-ai-daw",
    "oauth-login",
    "openrouter",
})
_LOCAL_ZERO_CALL_REASON = "local_family_zero_call"
# Closed read alias. artifacts/put and the other posted path strings are absent
# on purpose: they are not mapped onto ipfs_add or dag_put.
_READ_ALIAS_TOOL = "mcp++:ipfs_kit_py/artifacts/get"
_READ_ALIAS_SUFFIX = ".retrieve_content"
_READ_ALIAS_ENVELOPE_TOOL = "ipfs_cat"
_GATEWAY_PROTOCOL = "swissknife.all-app-tool-gateway.v1"
_STALE_BINDING_ATTR = 'data-live-gateway-binding="mcp++:ipfs_kit_py/artifacts/get"'
_CATALOG_BINDING_ATTR = (
    'data-live-gateway-binding="image-viewer.ipfs_kit_py.retrieve_content"'
)
_IMAGE_VIEWER_REL = "swissknife/web/js/apps/image-viewer.js"
_THIN_CALL = re.compile(
    r"""JSON\.stringify\(\s*\{[^{}]*\btool:\s*(['"])(?P<tool>[^'"]+)\1[^{}]*\}\s*\)""",
    re.S,
)
_CALL_MCP_FUNCTION = re.compile(
    r"async function callMcpTool\(tool, toolArguments\) \{.*?\n\}",
    re.S,
)
_GET_CALL_SITE = re.compile(
    r"""callMcpTool\(\s*(['"])mcp\+\+:ipfs_kit_py/artifacts/get\1"""
)
_ACTION_SELECTOR = re.compile(r"""^\[data-action=(['"])([^'"]+)\1\]$""")
_ID_SELECTOR = re.compile(r"^#[A-Za-z][\w-]*$")
_ARIA = re.compile(r"""aria-label=(['"])(.*?)\1""")
_CHROME = re.compile(r"""data-action=(['"])(minimize|maximize|close)\1""")
_BINDING = re.compile(r"""data-live-gateway-binding=(['"])([^'"]+)\1""")
_TOOL = re.compile(r"""['"](mcp\+\+:[^'"]+)['"]""")
_ROLE = {
    "button": "button",
    "link": "link",
    "input": "textbox",
    "textbox": "textbox",
    "select": "combobox",
    "checkbox": "checkbox",
    "radio": "radio",
    "slider": "slider",
}


def _rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    rows = payload.get("rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    bindings = payload.get("bindings")
    if isinstance(bindings, list):
        return [row for row in bindings if isinstance(row, dict)]
    return []


def _aria_label(selector: str) -> str:
    match = _ARIA.search(selector or "")
    return match.group(2) if match else ""


def _chrome_target(selector: str) -> str:
    text = selector or ""
    inner = text[1:-1] if text.startswith("[") and text.endswith("]") else text
    match = _CHROME.fullmatch(inner)
    if not match:
        return ""
    return f".window-control.{match.group(2)}"


def _binding_id(selector: str) -> str:
    match = _BINDING.search(selector or "")
    return match.group(2) if match else ""


def _dom_document(app_id: str, controls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    children = []
    for index, row in enumerate(controls):
        role = _ROLE.get(str(row.get("role") or "").lower(), "group")
        children.append(
            {
                "node_id": f"component:{app_id}:{index}",
                "role": role,
                "name": str(row.get("accessible_name") or ""),
                "tag_name": "button" if role == "button" else "div",
                "actions": ["activate"] if role == "button" else [],
            }
        )
    return {
        "document_id": f"desktop-{app_id}",
        "title": f"{app_id} rendered census",
        "source_uri": "dom-aria://rendered-census",
        "source_id": "rendered-control-census",
        "root": {
            "node_id": f"component:{app_id}",
            "role": "group",
            "name": app_id,
            "children": children,
        },
    }


def _project_dom(census_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Adapt each app through the DOM/ARIA UI/UX IR adapter."""

    from ipfs_datasets_py.logic.ui_ux_ir.source_adapters.dom_aria import (
        DOMARIA_UIIR_ADAPTER,
        adapt_dom_aria_to_uiir,
    )

    by_app: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in census_rows:
        by_app[str(row.get("app_id") or "")].append(row)
    apps = []
    failures = []
    for app_id in sorted(key for key in by_app if key):
        try:
            result = adapt_dom_aria_to_uiir(_dom_document(app_id, by_app[app_id]))
        except Exception as exc:  # adapter errors stay typed in the receipt
            failures.append({"app_id": app_id, "error": type(exc).__name__})
            continue
        apps.append(
            {
                "app_id": app_id,
                "adapter": result.adapter,
                "components": len(result.components),
                "losses": len(result.losses),
                "execution_performed": result.execution_performed,
            }
        )
    return {
        "adapter": DOMARIA_UIIR_ADAPTER,
        "app_count": len(apps),
        "component_count": sum(item["components"] for item in apps),
        "loss_count": sum(item["losses"] for item in apps),
        "execution_performed": False,
        "failures": failures,
        "apps": apps,
    }


def _operator_gate(before: str, after: str) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.autonomous_repair.program_world_operators import (
        apply_program_world_repair_operator,
    )

    application = apply_program_world_repair_operator(
        operator_kind="replace_exact_bytes",
        path=CONTRACT_REL,
        source_text=before + "\n",
        before_span=before,
        after_span=after,
        admitted_scope=(CONTRACT_REL,),
    )
    return {
        "disposition": str(application.disposition.value),
        "reason_codes": [str(item) for item in application.reason_codes],
    }


def _candidate_aliases(
    matrix_rows: Sequence[Mapping[str, Any]],
    census_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    census_by_app: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in census_rows:
        census_by_app[str(row.get("app_id") or "")].append(row)
    admitted: list[dict[str, Any]] = []
    abstained: list[dict[str, Any]] = []
    for row in matrix_rows:
        app_id = str(row.get("app_id") or "")
        selector = str(row.get("selector") or "")
        if not app_id or not selector:
            continue
        live = census_by_app.get(app_id, [])
        live_selectors = {str(item.get("selector") or "") for item in live}
        if selector in live_selectors:
            continue
        if "${" in selector:
            abstained.append(
                {
                    "app_id": app_id,
                    "selector": selector,
                    "reason": "template_selector",
                }
            )
            continue
        target = ""
        reason = ""
        chrome = _chrome_target(selector)
        if chrome and sum(1 for item in live if item.get("selector") == chrome) == 1:
            target = chrome
            reason = "window_chrome"
        else:
            label = _aria_label(selector)
            if not label:
                abstained.append(
                    {
                        "app_id": app_id,
                        "selector": selector,
                        "reason": "no_dom_name",
                    }
                )
                continue
            matches = [
                str(item.get("selector") or "")
                for item in live
                if str(item.get("accessible_name") or "") == label
                and str(item.get("selector") or "")
            ]
            unique = sorted(set(matches))
            if len(unique) != 1:
                abstained.append(
                    {
                        "app_id": app_id,
                        "selector": selector,
                        "reason": "ambiguous_accessible_name"
                        if len(unique) > 1
                        else "accessible_name_absent",
                    }
                )
                continue
            target = unique[0]
            reason = "unique_accessible_name"
        gate = _operator_gate(selector, target)
        record = {
            "app_id": app_id,
            "from_selector": selector,
            "to_selector": target,
            "reason": reason,
            "resolution": "unique_dom_alias",
            "operator_disposition": gate["disposition"],
            "operator_reason_codes": gate["reason_codes"],
        }
        if gate["disposition"] == "unique":
            admitted.append(record)
        else:
            record["resolution"] = "operator_rejected"
            abstained.append(record)
    return admitted, abstained


def _backend_contracts(
    *,
    census_rows: Sequence[Mapping[str, Any]],
    live_bindings: Sequence[Mapping[str, Any]],
    executable_bindings: Sequence[Mapping[str, Any]],
    posted_calls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    catalog = {
        str(row.get("binding_id") or ""): row
        for row in live_bindings
        if row.get("binding_id")
    }
    preferred: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in executable_bindings:
        app_id = str(row.get("app_id") or "")
        binding_id = str(row.get("binding_id") or "")
        for tool_id in row.get("preferred_tool_ids") or []:
            text = str(tool_id or "")
            if app_id and binding_id and text:
                preferred[(app_id, text)].append(binding_id)
    gateway = []
    for row in census_rows:
        selector = str(row.get("selector") or "")
        binding_id = _binding_id(selector)
        if not binding_id:
            continue
        known = catalog.get(binding_id)
        gateway.append(
            {
                "app_id": str(row.get("app_id") or ""),
                "selector": selector,
                "binding_id": binding_id,
                "owner": str((known or {}).get("owner") or ""),
                "catalog_match": known is not None,
            }
        )
    matches = []
    abstained = []
    for call in posted_calls:
        app_id = str(call.get("app_id") or "")
        tool_id = str(call.get("tool_id") or "")
        bindings = sorted(set(preferred.get((app_id, tool_id), [])))
        record = {
            "app_id": app_id,
            "tool_id": tool_id,
            "source": str(call.get("source") or ""),
            "envelope": str(call.get("envelope") or ""),
        }
        if len(bindings) == 1 and not tool_id.startswith("mcp++:"):
            record["binding_id"] = bindings[0]
            record["repair"] = "already_preferred_tool"
            matches.append(record)
            continue
        record["repair"] = "abstain"
        record["reason"] = (
            "ambiguous_preferred_tool" if len(bindings) > 1 else "tool_not_preferred"
        )
        abstained.append(record)
    return {
        "gateway_controls": gateway,
        "preferred_tool_matches": matches,
        "backend_abstentions": abstained,
        "app_source_rewritten": False,
    }


def build_click_dom_backend_contract(
    *,
    census: Mapping[str, Any],
    matrix: Mapping[str, Any],
    live_bindings: Mapping[str, Any] | None = None,
    executable_bindings: Sequence[Mapping[str, Any]] = (),
    posted_calls: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return the contract. Callers write it. Application source stays put."""

    from ipfs_accelerate_py.agent_supervisor.procedure_compiler.world_model import (
        WORLD_PROJECTION_ROLE,
    )

    census_rows = _rows(census)
    matrix_rows = _rows(matrix)
    aliases, dom_abstentions = _candidate_aliases(matrix_rows, census_rows)
    backend = _backend_contracts(
        census_rows=census_rows,
        live_bindings=_rows(live_bindings),
        executable_bindings=executable_bindings,
        posted_calls=posted_calls,
    )
    return {
        "schema": CONTRACT_SCHEMA,
        "world_projection_role": WORLD_PROJECTION_ROLE,
        "mutation_authorized": False,
        "completion_authoritative": False,
        "app_source_rewritten": False,
        "ui_ux_ir": _project_dom(census_rows),
        "dom_aliases": aliases,
        "dom_abstentions": dom_abstentions,
        **backend,
    }


def doctor_status_receipt() -> dict[str, Any]:
    """Read the report-only doctor. Never promotes it to a mutating mode."""

    from ipfs_accelerate_py.agent_supervisor.control.deterministic_doctor_rollout import (
        status,
    )

    report = status()
    return {
        "mode": report.get("mode"),
        "effective_mode": report.get("effective_mode"),
        "mutation_authorized": report.get("mutation_authorized"),
        "completion_authoritative": report.get("completion_authoritative"),
        "valid": report.get("valid"),
    }


def posted_calls_from_sources(
    sources: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    """Read tool ids already present in app source. Does not choose new ids."""

    calls = []
    for app_id, text in sources:
        gateway = "swissknife.all-app-tool-gateway.v1" in text
        thin = (
            "JSON.stringify" in text
            and "method:" in text
            and "arguments:" in text
            and "/mcp/tools/call" in text
        )
        if gateway and thin:
            envelope = "mixed"
        elif gateway:
            envelope = "gateway"
        elif thin:
            envelope = "thin_idl"
        else:
            envelope = "no_tools_call"
        seen: set[str] = set()
        for tool_id in _TOOL.findall(text):
            if tool_id in seen:
                continue
            seen.add(tool_id)
            calls.append(
                {
                    "app_id": app_id,
                    "tool_id": tool_id,
                    "source": app_id,
                    "envelope": envelope,
                }
            )
    return calls


def attach_doctor(contract: dict[str, Any]) -> dict[str, Any]:
    try:
        receipt = doctor_status_receipt()
    except Exception as exc:
        receipt = {
            "mode": "unavailable",
            "mutation_authorized": False,
            "completion_authoritative": False,
            "error": type(exc).__name__,
        }
    contract["doctor"] = receipt
    contract["mutation_authorized"] = False
    contract["completion_authoritative"] = False
    return contract


def write_contract(path: Any, contract: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(target)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _preferred_ids(
    binding_id: str,
    executable_bindings: Sequence[Mapping[str, Any]],
) -> list[str]:
    found: list[str] = []
    for row in executable_bindings:
        if str(row.get("binding_id") or "") != binding_id:
            continue
        for tool in row.get("preferred_tool_ids") or []:
            text = str(tool or "")
            if text and text not in found:
                found.append(text)
    return found


def _carried_disposition(
    *,
    app_id: str,
    selector: str,
    matrix_rows: Sequence[Mapping[str, Any]],
    aliases: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str]:
    """Copy a matrix disposition only for an exact selector or one unique alias."""

    exact = [
        row
        for row in matrix_rows
        if str(row.get("app_id") or "") == app_id
        and str(row.get("selector") or "") == selector
    ]
    chosen = exact
    carried_from = "matrix_selector"
    if not exact:
        matched = [
            row
            for row in aliases
            if str(row.get("app_id") or "") == app_id
            and str(row.get("to_selector") or "") == selector
            and row.get("resolution") == "unique_dom_alias"
            and row.get("operator_disposition") == "unique"
        ]
        if len(matched) != 1:
            return "", "", "none"
        source = str(matched[0].get("from_selector") or "")
        chosen = [
            row
            for row in matrix_rows
            if str(row.get("app_id") or "") == app_id
            and str(row.get("selector") or "") == source
        ]
        carried_from = "unique_dom_alias"
    dispositions = {str(row.get("disposition") or "") for row in chosen}
    if len(dispositions) != 1:
        return "", "", "none"
    disposition = next(iter(dispositions))
    if disposition not in _CARRIED_DISPOSITIONS:
        return "", "", "none"
    tool_ids = {str(row.get("tool_id") or "") for row in chosen}
    tool_id = next(iter(tool_ids)) if len(tool_ids) == 1 else ""
    return disposition, tool_id, carried_from


def worklist_scope_gate() -> dict[str, Any]:
    """Admit the work-list path only. The matrix path stays out of scope."""

    from ipfs_accelerate_py.agent_supervisor.autonomous_repair.program_world_operators import (
        apply_program_world_repair_operator,
    )

    application = apply_program_world_repair_operator(
        operator_kind="replace_exact_bytes",
        path=MATRIX_REL,
        source_text="matrix\n",
        before_span="matrix",
        after_span="census",
        admitted_scope=(WORKLIST_REL,),
    )
    return {
        "admitted_scope": [WORKLIST_REL],
        "matrix_path": MATRIX_REL,
        "matrix_disposition": str(application.disposition.value),
        "matrix_reason_codes": [str(item) for item in application.reason_codes],
        "matrix_rewritten": False,
    }


def _selector_loss(
    selector: str,
    app_id: str,
    catalog: set[str],
    app_sources: Mapping[str, str] | None,
    shared_source: str,
) -> str:
    """Return a loss reason when current source cannot render the selector."""

    if app_sources is None:
        return ""
    text = str(app_sources.get(app_id) or "")
    binding_id = _binding_id(selector)
    if binding_id and binding_id not in catalog:
        needle = f'data-live-gateway-binding="{binding_id}"'
        if needle not in text and needle not in shared_source:
            return "binding_attribute_absent"
    if _ID_SELECTOR.fullmatch(selector):
        token = selector[1:]
        if token not in text and token not in shared_source:
            return "dom_id_absent"
    action = _ACTION_SELECTOR.fullmatch(selector)
    if action and f'data-action="{action.group(2)}"' not in text and (
        f'data-action="{action.group(2)}"' not in shared_source
    ):
        return "action_attribute_absent"
    return ""


def build_click_worklist(
    *,
    census: Mapping[str, Any],
    matrix: Mapping[str, Any],
    live_bindings: Mapping[str, Any] | None = None,
    executable_bindings: Sequence[Mapping[str, Any]] = (),
    dom_aliases: Sequence[Mapping[str, Any]] = (),
    app_sources: Mapping[str, str] | None = None,
    shared_source: str = "",
) -> dict[str, Any]:
    """One row per rendered control. Loss receipts are not rows."""

    gate = worklist_scope_gate()
    rows: list[dict[str, Any]] = []
    losses: list[dict[str, str]] = []
    if gate["matrix_disposition"] == "out_of_scope":
        catalog = {
            str(row.get("binding_id") or "")
            for row in _rows(live_bindings)
            if row.get("binding_id")
        }
        for row in _rows(census):
            app_id = str(row.get("app_id") or "")
            selector = str(row.get("selector") or "")
            if not app_id or not selector or "${" in selector:
                continue
            loss = _selector_loss(selector, app_id, catalog, app_sources, shared_source)
            if loss:
                losses.append({"app_id": app_id, "selector": selector, "reason": loss})
                continue
            carried, carried_tool, carried_from = _carried_disposition(
                app_id=app_id,
                selector=selector,
                matrix_rows=_rows(matrix),
                aliases=dom_aliases,
            )
            disposition = ""
            tool_id = ""
            reason = ""
            binding_id = _binding_id(selector)
            local_gateway = (
                app_id in _LOCAL_ZERO_CALL_APPS
                and bool(binding_id)
                and binding_id in catalog
            )
            if app_id in _LOCAL_ZERO_CALL_APPS and carried == "external-provider":
                disposition = "external-provider"
                tool_id = carried_tool
                reason = carried_from
            elif local_gateway:
                disposition = "unsupported"
                tool_id = ""
                reason = _LOCAL_ZERO_CALL_REASON
            elif binding_id and binding_id in catalog:
                preferred = _preferred_ids(binding_id, executable_bindings)
                if len(preferred) == 1:
                    disposition = "invoke"
                    tool_id = preferred[0]
                    reason = "single_preferred_tool"
                elif len(preferred) > 1:
                    disposition = "unlisted"
                    tool_id = ""
                    reason = "multiple_preferred_tools"
            if not disposition and selector in _CHROME_SELECTORS:
                disposition = "local-only"
                tool_id = ""
                reason = "window_chrome"
            if not disposition and carried:
                disposition = carried
                tool_id = carried_tool
                reason = carried_from
            if not disposition:
                disposition = "unsupported"
                tool_id = ""
                reason = "no_declared_binding"
            rows.append(
                {
                    "app_id": app_id,
                    "launch_id": str(row.get("launch_id") or app_id),
                    "selector": selector,
                    "role": str(row.get("role") or ""),
                    "accessible_name": str(row.get("accessible_name") or ""),
                    "disposition": disposition,
                    "tool_id": tool_id,
                    "reason": reason,
                    "carried_from": carried_from,
                }
            )
    rows.sort(key=lambda item: (item["app_id"], item["launch_id"], item["selector"]))
    return {
        "schema": WORKLIST_SCHEMA,
        "doctor_operation": "plan",
        "mutation_authorized": False,
        "completion_authoritative": False,
        "app_source_rewritten": False,
        "operator": gate,
        "row_count": len(rows),
        "unrendered_loss_count": len(losses),
        "unrendered_losses": losses,
        "rows": rows,
    }


def _finite_adapter(path: str, source: str, before: str, after: str):
    from ipfs_accelerate_py.agent_supervisor.autonomous_repair.program_world_operators import (
        apply_program_world_repair_operator,
    )

    return apply_program_world_repair_operator(
        operator_kind="finite_adapter",
        path=path,
        source_text=source if source else "\n",
        before_span=before,
        after_span=after,
        admitted_scope=(path,),
    )


def _sketch_record(application: Any, source: str) -> dict[str, Any]:
    disposition = str(application.disposition.value)
    rewritten = application.after_source if disposition == "unique" else source
    return {
        "disposition": disposition,
        "reason_codes": [str(item) for item in application.reason_codes],
        "source_text": source,
        "rewritten_text": rewritten,
        "applied": False,
    }


def _call_mcp_get_branch(before: str, app_id: str, binding_id: str) -> str:
    """Keep every non-get call on the thin body. Get posts the catalog envelope."""

    indent = "    "
    for line in before.splitlines()[1:]:
        if line.strip():
            indent = line[: len(line) - len(line.lstrip(" "))]
            break
    envelope = (
        "JSON.stringify({ protocol: 'swissknife.all-app-tool-gateway.v1',"
        " receipt_required: true,"
        f" binding_id: '{binding_id}',"
        f" app_id: '{app_id}',"
        f" intent_id: '{binding_id}',"
        " owner: 'ipfs_kit_py',"
        " tool_id: 'ipfs_cat',"
        " transport: 'http',"
        " route: '/mcp/tools/call',"
        " correlation_id: correlationId,"
        " input: { payload: toolArguments,"
        " policy: { outcome: 'require_confirmation',"
        " dry_run: globalThis.__SWISSKNIFE_GATEWAY_FORCE_DRY_RUN__ === true } } })"
    )
    step = indent
    return "\n".join(
        (
            "async function callMcpTool(tool, toolArguments) {",
            f"{step}const correlationId = '{app_id}:' + Date.now();",
            f"{step}const body = tool === '{_READ_ALIAS_TOOL}'",
            f"{step}    ? {envelope}",
            f"{step}    : JSON.stringify({{",
            f"{step}        method: 'request_action',",
            f"{step}        tool,",
            f"{step}        arguments: toolArguments",
            f"{step}    }});",
            f"{step}const response = await fetch('/mcp/tools/call', {{",
            f"{step}    method: 'POST',",
            f"{step}    headers: {{ 'Content-Type': 'application/json' }},",
            f"{step}    body",
            f"{step}}});",
            f"{step}return parseToolResponse(response);",
            "}",
        )
    )


def _ipfs_cat_envelope(app_id: str, binding_id: str) -> str:
    return (
        "JSON.stringify({ protocol: 'swissknife.all-app-tool-gateway.v1',"
        " receipt_required: true,"
        f" binding_id: '{binding_id}',"
        f" app_id: '{app_id}',"
        f" intent_id: '{binding_id}',"
        " owner: 'ipfs_kit_py',"
        " tool_id: 'ipfs_cat',"
        " transport: 'http',"
        " route: '/mcp/tools/call',"
        " input: { policy: { dry_run: globalThis.__SWISSKNIFE_GATEWAY_FORCE_DRY_RUN__ === true } } })"
    )


def _read_alias_binding_ids(
    app_id: str,
    executable_bindings: Sequence[Mapping[str, Any]],
) -> list[str]:
    found: list[str] = []
    for row in executable_bindings:
        if str(row.get("app_id") or "") != app_id:
            continue
        binding_id = str(row.get("binding_id") or "")
        if not binding_id.endswith(_READ_ALIAS_SUFFIX):
            continue
        tools = [str(item) for item in (row.get("preferred_tool_ids") or [])]
        if _READ_ALIAS_ENVELOPE_TOOL not in tools or binding_id in found:
            continue
        found.append(binding_id)
    return found


def sketch_finite_read_alias(
    source: str,
    *,
    app_id: str,
    path: str,
    executable_bindings: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Sketch one artifacts/get envelope. Never writes the file."""

    text = str(source)
    spans = list(_THIN_CALL.finditer(text))
    get_spans = [match for match in spans if match.group("tool") == _READ_ALIAS_TOOL]
    bindings = _read_alias_binding_ids(app_id, executable_bindings)

    def unchanged(before: str, after: str) -> dict[str, Any]:
        application = _finite_adapter(path, text, before, after)
        record = _sketch_record(application, text)
        record["rewritten_text"] = text
        record["applied"] = False
        return record

    if len(get_spans) == 1 and len(bindings) == 1:
        before = get_spans[0].group(0)
        after = _ipfs_cat_envelope(app_id, bindings[0])
        application = _finite_adapter(path, text, before, after)
        record = _sketch_record(application, text)
        record["applied"] = False
        if record["disposition"] != "unique":
            record["rewritten_text"] = text
        return record
    functions = list(_CALL_MCP_FUNCTION.finditer(text))
    get_sites = list(_GET_CALL_SITE.finditer(text))
    if get_sites and len(bindings) == 1 and functions:
        before = functions[0].group(0)
        after = _call_mcp_get_branch(before, app_id, bindings[0])
        if len(functions) != 1:
            record = unchanged(before, after)
            record["disposition"] = "ambiguous"
            if "adapter_span_not_unique" not in record["reason_codes"]:
                record["reason_codes"] = ["adapter_span_not_unique", *record["reason_codes"]]
            return record
        application = _finite_adapter(path, text, before, after)
        record = _sketch_record(application, text)
        record["applied"] = False
        if record["disposition"] != "unique":
            record["rewritten_text"] = text
        return record
    if len(get_spans) > 1:
        before = get_spans[0].group(0)
        after = _ipfs_cat_envelope(app_id, bindings[0] if bindings else "abstain")
        return unchanged(before, after)
    if spans:
        return unchanged(spans[0].group(0), "")
    return unchanged("", "")


def sketch_gateway_binding_attribute(
    source: str,
    *,
    path: str = _IMAGE_VIEWER_REL,
) -> dict[str, Any]:
    """Sketch the image-viewer attribute span. Never writes the file."""

    text = str(source)
    if path != _IMAGE_VIEWER_REL:
        from ipfs_accelerate_py.agent_supervisor.autonomous_repair.program_world_operators import (
            apply_program_world_repair_operator,
        )

        application = apply_program_world_repair_operator(
            operator_kind="finite_adapter",
            path=path,
            source_text=text or "\n",
            before_span=_STALE_BINDING_ATTR,
            after_span=_CATALOG_BINDING_ATTR,
            admitted_scope=(_IMAGE_VIEWER_REL,),
        )
        record = _sketch_record(application, text)
        record["rewritten_text"] = text
        record["applied"] = False
        return record
    application = _finite_adapter(path, text, _STALE_BINDING_ATTR, _CATALOG_BINDING_ATTR)
    record = _sketch_record(application, text)
    record["applied"] = False
    if record["disposition"] != "unique":
        record["rewritten_text"] = text
    return record


def _compact_sketch(app_id: str, path: str, sketch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "app_id": app_id,
        "path": path,
        "disposition": sketch.get("disposition"),
        "reason_codes": list(sketch.get("reason_codes") or []),
        "applied": False,
    }


def _shared_desktop_source(repo_root: Path) -> str:
    web = repo_root / "swissknife/web"
    if not web.is_dir():
        return ""
    parts: list[str] = []
    for path in sorted(web.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".js", ".ts", ".html", ".css"}:
            continue
        if "node_modules" in path.parts or path.stat().st_size > 2_000_000:
            continue
        if path.parent.name == "apps" and path.suffix == ".js":
            continue
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def build_from_repo(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = repo_root / "swissknife/test-results/virtual-desktop-ipfs-mcp-orb"
    census = _load_json(evidence / "vog/rendered-control-census.json")
    matrix = _load_json(evidence / "control-tool-matrix.json")
    bindings_path = evidence / "all-app-live-tool-bindings.json"
    live_bindings = _load_json(bindings_path) if bindings_path.is_file() else {}
    executable_path = evidence / "vog/executable-backend-bindings.json"
    executable = _rows(_load_json(executable_path)) if executable_path.is_file() else []
    sources = []
    apps = repo_root / "swissknife/web/js/apps"
    if apps.is_dir():
        for path in sorted(apps.glob("*.js")):
            sources.append((path.stem, path.read_text(encoding="utf-8", errors="replace")))
    contract = attach_doctor(
        build_click_dom_backend_contract(
            census=census,
            matrix=matrix,
            live_bindings=live_bindings,
            executable_bindings=executable,
            posted_calls=posted_calls_from_sources(sources),
        )
    )
    app_sources = {app_id: text for app_id, text in sources}
    shared_source = _shared_desktop_source(repo_root)
    worklist = build_click_worklist(
        census=census,
        matrix=matrix,
        live_bindings=live_bindings,
        executable_bindings=executable,
        dom_aliases=contract["dom_aliases"],
        app_sources=app_sources,
        shared_source=shared_source,
    )
    sketches = []
    image_text = ""
    for app_id, text in sources:
        rel = f"swissknife/web/js/apps/{app_id}.js"
        if app_id == "image-viewer":
            image_text = text
        sketch = sketch_finite_read_alias(
            text,
            app_id=app_id,
            path=rel,
            executable_bindings=executable,
        )
        sketches.append(_compact_sketch(app_id, rel, sketch))
    attribute = (
        sketch_gateway_binding_attribute(image_text, path=_IMAGE_VIEWER_REL)
        if image_text
        else {"disposition": "abstained", "reason_codes": ["adapter_span_absent"]}
    )
    contract["read_alias_sketches"] = sketches
    contract["gateway_attribute_sketch"] = {
        "path": _IMAGE_VIEWER_REL,
        "disposition": attribute.get("disposition"),
        "reason_codes": list(attribute.get("reason_codes") or []),
        "applied": False,
    }
    contract["app_source_rewritten"] = False
    contract["mutation_authorized"] = False
    contract["completion_authoritative"] = False
    return contract, worklist


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    contract, worklist = build_from_repo(repo_root)
    write_contract(repo_root / CONTRACT_REL, contract)
    doctor = contract.get("doctor") or {}
    gate_ok = worklist["operator"]["matrix_disposition"] == "out_of_scope"
    doctor_ok = (
        doctor.get("mode") == "report_only"
        and doctor.get("effective_mode") in {None, "report_only"}
        and doctor.get("mutation_authorized") is False
        and contract.get("mutation_authorized") is False
    )
    wrote_worklist = False
    if gate_ok and doctor_ok:
        write_contract(repo_root / WORKLIST_REL, worklist)
        wrote_worklist = True
    unique = sum(
        1
        for row in contract["read_alias_sketches"]
        if row.get("disposition") == "unique"
    )
    print(
        "click dom backend contract"
        f" aliases={len(contract['dom_aliases'])}"
        f" dom_abstentions={len(contract['dom_abstentions'])}"
        f" gateway={len(contract['gateway_controls'])}"
        f" backend_abstentions={len(contract['backend_abstentions'])}"
        f" doctor={doctor.get('mode')}"
        f" worklist_rows={worklist['row_count']}"
        f" unrendered_losses={worklist.get('unrendered_loss_count', 0)}"
        f" worklist_wrote={wrote_worklist}"
        f" read_alias_unique={unique}"
        f" attribute={contract['gateway_attribute_sketch'].get('disposition')}"
    )
    return 0 if wrote_worklist else 1


if __name__ == "__main__":
    raise SystemExit(main())
