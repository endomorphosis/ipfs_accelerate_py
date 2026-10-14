"""Task-scoped source views for the existing isolated coding session.

The verified semantic producer admits the public population. This read-only
adapter shares the numerical reader's socket/child lifecycle, creates no
database owner, and does not impersonate the typed owner's source.get grant.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from . import worker_source_client as protocol
from .worker_source_request_observation import AUDIT_SCHEMA, request_details
from .numerical_obligation_broker import (
    MAX_SESSION_SECONDS, MAX_TOTAL_RESPONSE_BYTES, NumericalObligationBroker, _argument,
)


class WorkerSourceBroker(NumericalObligationBroker):
    protocol = protocol
    audit_schema = AUDIT_SCHEMA
    file_prefix = "source-view"

    def __init__(self, *, repository, workspace, boundary, prompt, timeout_seconds,
                 profile="python-whitespace@1", view_policy=None, source_edit_mode="disabled",
                 semantic_evidence_mode="disabled"):
        from . import source_formalization_context as guard
        from .semantic_router_translation import encode_semantic_router_prompt
        from .worker_source_views import WorkerSourceViews
        from .source_view_policy import SourceViewPolicy

        if view_policy is not None and (type(view_policy) is not SourceViewPolicy
                or type(profile) is not str or profile != "python-whitespace@1"):
            raise ValueError("shared policy cannot be combined with legacy profile")
        selected_profile = profile if view_policy is None else view_policy.profile
        if type(source_edit_mode) is not str or source_edit_mode not in ("disabled", *protocol.EDIT_MODES):
            raise ValueError("unsupported source edit mode")
        self.source_edit_mode = source_edit_mode
        if type(semantic_evidence_mode) is not str or semantic_evidence_mode not in (
                "disabled", "selected-capsules@1"):
            raise ValueError("unsupported semantic evidence mode")
        self.semantic_evidence_mode = semantic_evidence_mode
        self.semantic_evidence = None
        self._edit_confirmation_blocked = False
        self._edit_confirmation_request = None
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= MAX_SESSION_SECONDS:
            raise ValueError("bounded source-view session required")
        self.repository, self.workspace = guard._workspace(repository, workspace)
        self.boundary = Path(boundary).absolute()
        boundary_value, boundary_sha = guard._owner_boundary(self.boundary)
        self.owner_uid, self.worker_uid = boundary_value["owner_uid"], boundary_value["worker_uid"]
        if not (self.owner_uid > 0 and self.worker_uid > 0 and os.getgid() == self.owner_uid):
            raise ValueError("source-view broker requires the isolated nonroot owner group")
        self.prompt = prompt
        # Only the sealed-argv factory can establish whether a separately
        # admitted public instruction was selected for this original launch.
        # Direct constructors deliberately leave that fact unknown.
        self.original_requirements = None
        self.original_requirements_unavailable_reason = "original_launch_selection_not_captured"
        self._doctor_request = None
        encoded = encode_semantic_router_prompt(prompt=prompt, repository=self.repository)
        table = encoded.table.to_dict()
        self._table_bytes = protocol.canonical_json(table)
        self._binding = protocol.semantic_binding(encoded, selected_profile)
        view_args = dict(self._binding)
        if view_policy is not None:
            view_args.pop("profile")
            view_args["view_policy"] = view_policy
        self.views = WorkerSourceViews(repository=self.repository, workspace=self.workspace,
            source_hashes={name: binding["sha256"] for name, binding in table["source_manifest"].items()},
            **view_args)
        admission = self.views.admission()
        self._view_admission = protocol.canonical_json(admission)
        self._session_binding = dict(self._binding)
        self._descriptor_schema = protocol.DESCRIPTOR_SCHEMA
        if view_policy is not None:
            self._session_binding.update(source_scope_sha256=admission["scope_sha256"],
                view_policy_binding_sha256=hashlib.sha256(
                    protocol.canonical_json(admission["view_policy_binding"])).hexdigest())
            self._descriptor_schema = protocol.DESCRIPTOR_SCHEMA_V2
        # Read the exact artifact already sealed into the verified translation.
        # Its versioned program selection excludes support/instruction files.
        from .semantic_router_translation import _read, _sha, _parse
        from .worker_candidate_context import WorkerCandidateContext
        raw = _read(self.repository, table["semantic_artifact"])
        if _sha(raw) != table["semantic_sha256"]:
            raise ValueError("candidate program selection changed")
        payload = _parse(raw.decode("utf-8"))
        program_paths = payload.get("program_paths", sorted(table["source_manifest"]))
        self.candidate_context = WorkerCandidateContext(self.views, tuple(program_paths))
        self.source_edits = None
        if source_edit_mode != "disabled":
            if type(payload.get("program_paths")) is not list or not payload["program_paths"]:
                raise ValueError("source edits require an explicit verified program population")
            from .worker_source_edit import WorkerSourceEditService
            self.source_edits = WorkerSourceEditService(views=self.views, allowed_paths=tuple(program_paths),
                **({"mode": source_edit_mode} if source_edit_mode != "mapped-span@1" else {}))
            self._session_binding.update(source_edit_mode=source_edit_mode,
                source_scope_sha256=admission["scope_sha256"],
                view_policy_binding_sha256=self._session_binding.get("view_policy_binding_sha256"))
            self._descriptor_schema = protocol.DESCRIPTOR_SCHEMA_V3
        if semantic_evidence_mode != "disabled":
            from .worker_semantic_evidence import WorkerSemanticEvidence
            self.semantic_evidence = WorkerSemanticEvidence(capsules=payload["capsules"], binding=self._binding)
            self._session_binding.update(source_edit_mode=source_edit_mode,
                source_scope_sha256=admission["scope_sha256"],
                view_policy_binding_sha256=self._session_binding.get("view_policy_binding_sha256"),
                semantic_evidence_mode=semantic_evidence_mode)
            self._descriptor_schema = protocol.DESCRIPTOR_SCHEMA_V4
            self.audit_schema = "supervisor-worker-source-session@6"
        self._start_session(identity=self._session_binding, boundary_sha=boundary_sha,
                            timeout_seconds=timeout_seconds, descriptor_schema=self._descriptor_schema)

    def _current_context(self, *, working=False, generation=None):
        from . import source_formalization_context as guard
        from .semantic_router_translation import encode_semantic_router_prompt

        if not self._live():
            raise ValueError("source-view session expired")
        if guard._owner_boundary(self.boundary)[1] != self.descriptor["boundary_manifest_sha256"]:
            raise ValueError("source-view boundary changed")
        if (self.descriptor.get("schema") != self._descriptor_schema
                or any(self.descriptor.get(key) != value for key, value in self._session_binding.items())):
            raise ValueError("source-view descriptor binding changed")
        encoded = encode_semantic_router_prompt(prompt=self.prompt, repository=self.repository)
        admission = self.views.baseline_admission() if working else self.views.admission()
        if (protocol.canonical_json(encoded.table.to_dict()) != self._table_bytes
                or protocol.canonical_json(admission) != self._view_admission):
            raise ValueError("source-view semantic admission changed")
        if generation is not None:
            self.views.working_admission(expected_workspace_generation=generation)

    def _answer(self, request, channel):
        if not self._live():
            return self._response(request, reason="session_closed")
        if request["session_id"] != self.descriptor["session_id"] or request["request_id"] in self._seen:
            return self._response(request, reason="invalid_request")
        self._seen.add(request["request_id"])
        if self._response_bytes + request["query"]["max_bytes"] > MAX_TOTAL_RESPONSE_BYTES:
            return self._response(request, reason="budget_exhausted")
        try:
            self._peer(channel)
        except (OSError, ValueError):
            return self._response(request, reason="peer_refused")
        if request["schema"] == "supervisor-worker-source-edit-request@1":
            return self._answer_edit(request, channel)
        if request["schema"] == "supervisor-worker-semantic-evidence-request@1":
            return self._answer_semantic_evidence(request, channel)
        try:
            candidate = request["schema"] in {"supervisor-worker-candidate-context-request@1",
                                               "supervisor-worker-candidate-context-request@2"}
            modern = request["schema"] == protocol.REQUEST_SCHEMA_V3
            discovery = modern and request["query"]["operation"] == "discover"
            if modern:
                if request["query"]["view_policy_binding_sha256"] != self._session_binding.get(
                        "view_policy_binding_sha256"):
                    return self._response(request, reason="invalid_request")
            elif not candidate and self._session_binding.get("view_policy_binding_sha256") is not None:
                return self._response(request, reason="invalid_request")
            working = candidate or (not discovery and request["query"].get("source_state") == "working-candidate")
            self._current_context(working=working,
                generation=request["query"]["expected_workspace_generation"] if candidate else None)
        except (OSError, ValueError, RuntimeError):
            self.candidate_context.clear()
            return self._response(request, reason="stale_context")
        overhead = len(protocol.canonical_json(self._response(request, page={}))) - 2
        query = dict(request["query"], max_bytes=request["query"]["max_bytes"] - overhead)
        if modern:
            query.pop("operation")
            query.pop("view_policy_binding_sha256")
        try:
            if candidate:
                page = self.candidate_context.page(**query)
            elif discovery:
                page = self.views.discovery(**query)
            elif working:
                query.pop("source_state")
                page = self.views.working_page(**query)
            else:
                if modern:
                    query.pop("source_state")
                    query.pop("expected_workspace_generation")
                page = self.views.page(**query)
        except (OSError, ValueError, RuntimeError):
            self.candidate_context.clear()
            return self._response(request, reason="retrieval_refused")
        try:
            self._current_context(working=working, generation=page.get("workspace_generation"))
            self._peer(channel)
        except (OSError, ValueError, RuntimeError):
            self.candidate_context.clear()
            return self._response(request, reason="stale_context")
        response = self._response(request, page=page)
        if len(protocol.canonical_json(response)) > request["query"]["max_bytes"]:
            return self._response(request, reason="retrieval_refused")
        return response

    def _answer_semantic_evidence(self, request, channel):
        if (self.semantic_evidence is None or self._descriptor_schema != protocol.DESCRIPTOR_SCHEMA_V4
                or self._session_binding.get("semantic_evidence_mode") != "selected-capsules@1"):
            return self._response(request, reason="invalid_request")
        try:
            self._current_context(working=False)
        except (OSError, ValueError, RuntimeError):
            self.semantic_evidence.clear()
            return self._response(request, reason="stale_context")
        overhead = len(protocol.canonical_json(self._response(request, page={}))) - 2
        try:
            page = self.semantic_evidence.page(**dict(request["query"],
                max_bytes=request["query"]["max_bytes"] - overhead))
        except (OSError, ValueError, RuntimeError):
            self.semantic_evidence.clear()
            return self._response(request, reason="retrieval_refused")
        try:
            self._current_context(working=False)
            self._peer(channel)
        except (OSError, ValueError, RuntimeError):
            self.semantic_evidence.clear()
            return self._response(request, reason="stale_context")
        response = self._response(request, page=page)
        if len(protocol.canonical_json(response)) > request["query"]["max_bytes"]:
            self.semantic_evidence.clear()
            return self._response(request, reason="retrieval_refused")
        return response

    def _answer_edit(self, request, channel):
        """Prepare/confirm a mapping; all filesystem writes stay in the worker."""
        query = request["query"]
        if (self.source_edits is None or self._descriptor_schema not in (
                protocol.DESCRIPTOR_SCHEMA_V3, protocol.DESCRIPTOR_SCHEMA_V4)
                or query["view_policy_binding_sha256"] != self._session_binding.get("view_policy_binding_sha256")):
            return self._response(request, reason="invalid_request")
        if self.source_edit_mode == "mapped-span@2":
            if self._edit_confirmation_blocked:
                return self._response(request, reason="retrieval_refused")
            if query["operation"] == "confirm" and self.source_edits.pending_confirmation_matches(
                    query["proposal_sha256"]):
                # The worker may already have written. Surrounding owner checks
                # and response delivery must also succeed before another edit.
                self._edit_confirmation_blocked = True
                self._edit_confirmation_request = (request["request_id"], query["proposal_sha256"])
        try:
            self._current_context(working=True,
                generation=query.get("expected_workspace_generation"))
            page = self.source_edits.handle(query,
                request_sha256=hashlib.sha256(protocol.canonical_json(request)).hexdigest())
            if query["operation"] == "confirm":
                self.candidate_context.clear()
            self._current_context(working=True,
                generation=page.get("workspace_generation", query.get("expected_workspace_generation")))
            self._peer(channel)
            response = self._response(request, page=page)
            if len(protocol.canonical_json(response)) > query["max_bytes"]:
                raise ValueError("source edit response exceeds bound")
            return response
        except (OSError, ValueError, RuntimeError):
            self.candidate_context.clear()
            return self._response(request, reason="retrieval_refused")

    def _response_finished(self, request, response, *, delivered):
        if self.source_edit_mode != "mapped-span@2" or self._edit_confirmation_request is None:
            return
        request_id, proposal = self._edit_confirmation_request
        if request is None or request.get("request_id") != request_id:
            return
        self._edit_confirmation_request = None
        page = response.get("page") if type(response) is dict and response.get("ok") is True else None
        if (delivered is True and type(page) is dict
                and page.get("schema") == "supervisor-worker-source-edit-confirmed@1"
                and page.get("proposal_sha256") == proposal):
            self._edit_confirmation_blocked = False
        # A successful send is not a client acknowledgment. Client-side loss
        # still requires inspection and must never trigger an automatic retry.

    def _request_details(self, request=None):
        return request_details(request)

    def _record_details(self, response=None):
        page = response["page"] if response is not None and response["ok"] else None
        edit = page is not None and page["schema"].startswith("supervisor-worker-source-edit-")
        candidate = page is not None and page["schema"] in {
            "supervisor-worker-candidate-context-page@1", "supervisor-worker-candidate-context-page@2"}
        discovery = page is not None and page["schema"] in {
            "supervisor-worker-source-view-discovery@1", "supervisor-worker-source-view-discovery@2"}
        semantic = page is not None and page["schema"] == "supervisor-worker-semantic-evidence-page@1"
        count = page["end_offset"] - page["offset"] if page is not None and not discovery and not edit else 0
        result = dict(returned_source_bytes=0 if candidate or discovery or edit or semantic else count,
            returned_candidate_context_bytes=count if candidate else 0,
            candidate_context_page=candidate, discovery_page=discovery,
            returned_discovery_rows=page["returned_count"] if discovery else 0)
        if self.semantic_evidence_mode != "disabled":
            result.update(semantic_evidence_page=semantic,
                returned_semantic_evidence_bytes=count if semantic else 0)
        if self.source_edit_mode != "disabled":
            result.update(source_edit_page=edit,
                source_edit_operation=("prepare" if page["schema"].endswith("prepared@1") else "confirm") if edit else None,
                owner_writes_performed=False)
        return result



def worker_source_argv(argv):
    """Remove owner-only policy configuration before starting an isolated worker."""
    from .source_view_policy import SOURCE_VIEW_POLICY_FLAG, SourceViewPolicy

    if type(argv) is not list or any(type(arg) is not str for arg in argv):
        raise ValueError("literal source-view owner argv required")
    raw = _argument(argv, SOURCE_VIEW_POLICY_FLAG)
    if raw is not None:
        SourceViewPolicy.from_json(raw)
    result, index = [], 0
    while index < len(argv):
        arg = argv[index]
        if arg == SOURCE_VIEW_POLICY_FLAG:
            index += 2
        elif arg.startswith(SOURCE_VIEW_POLICY_FLAG + "="):
            index += 1
        else:
            result.append(arg)
            index += 1
    return result


def prepare_worker_source_broker(*, argv, boundary, workspace, input_fd=0, view_policy=None):
    """Opt in using an owner-sealed semantic prompt, without numerical context."""
    if (type(argv) is not list or any(type(arg) is not str for arg in argv)
            or any(arg == flag or arg.startswith(flag + "=") for arg in argv
                   for flag in (protocol.DESCRIPTOR_FLAG, protocol.DIGEST_FLAG))):
        raise ValueError("owner creates source-view descriptors independently")
    from .source_view_policy import SOURCE_VIEW_POLICY_FLAG, SourceViewPolicy

    profile = _argument(argv, "--source-view-profile", "disabled")
    working_set = _argument(argv, "--semantic-working-set", "legacy")
    if working_set not in ("legacy", "capsule-references@1"):
        raise ValueError("unsupported semantic working set")
    if working_set != "legacy" and profile == "disabled":
        raise ValueError("capsule references require an enabled source view profile")
    edit_mode = _argument(argv, "--source-edit-mode", "disabled")
    if edit_mode not in ("disabled", *protocol.EDIT_MODES) or edit_mode != "disabled" and profile == "disabled":
        raise ValueError("source edits require an enabled source view profile")
    serialized_policy = _argument(argv, SOURCE_VIEW_POLICY_FLAG)
    if serialized_policy is not None:
        if view_policy is not None:
            raise ValueError("owner policy has more than one selection")
        view_policy = SourceViewPolicy.from_json(serialized_policy)
    if view_policy is not None:
        if type(view_policy) is not SourceViewPolicy:
            raise ValueError("exact owner source-view policy required")
        view_policy.to_dict()
        if profile != view_policy.profile:
            raise ValueError("owner policy differs from selected source-view profile")
    if profile == "disabled":
        return None
    if profile not in protocol.PROFILES:
        raise ValueError("unsupported source-view profile")
    if _argument(argv, "--purpose", "coding") != "coding" or "--preflight" in argv:
        raise ValueError("source-view broker requires an isolated coding invocation")
    repository = _argument(argv, "--semantic-repository")
    if repository is None:
        raise ValueError("source-view broker requires the selected semantic repository")
    import fcntl

    required = sum(getattr(fcntl, name, value) for name, value in (
        ("F_SEAL_SEAL", 1), ("F_SEAL_SHRINK", 2), ("F_SEAL_GROW", 4), ("F_SEAL_WRITE", 8)))
    if fcntl.fcntl(input_fd, getattr(fcntl, "F_GET_SEALS", 1034)) & required != required:
        raise ValueError("source-view broker requires the sealed owner prompt")
    size = os.fstat(input_fd).st_size
    if size > 2 * 1024 * 1024:
        raise ValueError("source-view prompt exceeds bound")
    prompt = os.pread(input_fd, size + 1, 0).decode("utf-8")
    instruction_args = tuple(_argument(argv, flag) for flag in (
        "--public-instruction-artifact", "--public-instruction-sha256", "--public-instruction-task-cid"))
    if any(value is not None for value in instruction_args) and not all(instruction_args):
        raise ValueError("complete original public instruction selection required")
    doctor_args = tuple(_argument(argv, flag) for flag in (
        "--doctor-residual-artifact", "--doctor-residual-sha256", "--doctor-residual-task-cid"))
    if any(value is not None for value in doctor_args) and not all(doctor_args):
        raise ValueError("complete original Doctor residual selection required")
    value = WorkerSourceBroker(repository=repository, workspace=workspace, boundary=boundary,
        prompt=prompt, timeout_seconds=int(_argument(argv, "--timeout", "90")),
        **({"semantic_evidence_mode": "selected-capsules@1"} if working_set != "legacy" else {}),
        **({"source_edit_mode": edit_mode} if edit_mode != "disabled" else {}),
        **(dict(profile=profile) if view_policy is None else dict(view_policy=view_policy)))
    try:
        from .owner_candidate_round_prompt import capture_public_requirements, capture_native_only_requirements
        doctor_options = ({"doctor_request": dict(artifact=Path(doctor_args[0]),
            expected_sha256=doctor_args[1], task_cid=doctor_args[2])} if all(doctor_args) else {})
        if doctor_options:
            value._doctor_request = (str(Path(doctor_args[0]).absolute()), doctor_args[1], doctor_args[2])
        value.original_requirements = (capture_public_requirements(source_broker=value,
            artifact=Path(instruction_args[0]), expected_sha256=instruction_args[1],
            task_cid=instruction_args[2], **doctor_options) if all(instruction_args)
            else capture_native_only_requirements(source_broker=value, **doctor_options))
        value.original_requirements_unavailable_reason = None
    except BaseException:
        value.close()
        raise
    return value
