"""Fresh repository observations and residual planning for one native task.

This additive owner boundary composes the finite proposal service with an
independently signed local task. A finite fact is never a general source theorem,
and the proposal service's review-only result never supplies execution capacity.
The native daemon retains validation, publication and completion ownership.
"""
from __future__ import annotations

import ast
import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time

from ..planning import finite_integer_codebase as matcher
from ..planning.finite_integer_plan_preview import preview_finite_integer_plan
from ..planning.structural_codebase_context import structural_codebase_context
from ..proof.formal_verification_contracts import content_identity
from ..task_sources.intent_repository import IntentRepository
from . import local_planning_admission as local
from .scalar_candidate_handoff import _source, _persist
from .doctor_contract_candidate_runner import _path

SCHEMA = "supervisor-repository-finite-handoff@1"
RESULT_SCHEMA = "supervisor-repository-finite-preparation@1"
SCOPE = "exact integer observations over the complete explicitly enumerated input domain"
FALSE = {name: False for name in ("proof_authority", "source_semantics_verified",
    "whole_program_semantics_verified", "execution_authority", "completion_authority",
    "publication_authority", "omission_authority", "model_inference_used")}


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def propose_offset_source(source: bytes, contract) -> bytes:
    """Change only the closed source profile's returned arithmetic expression."""
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import _guard
    function, old_offset = _guard(source, contract)
    if old_offset == contract.offset:
        raise ValueError("the selected offset is already present")
    value = function.body[0].value
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:value.lineno - 1])) + value.col_offset
    end = sum(map(len, lines[:value.end_lineno - 1])) + value.end_col_offset
    expression = (contract.parameter if contract.offset == 0 else
        f"{contract.parameter} {'+' if contract.offset > 0 else '-'} {abs(contract.offset)}")
    candidate = source[:start] + expression.encode("ascii") + source[end:]
    after, found = _guard(candidate, contract)
    expected = ast.parse(source.decode("ascii"), type_comments=True)
    expected.body[0].body[0].value = ast.parse(expression, mode="eval").body
    if (found != contract.offset or ast.dump(expected, include_attributes=False) !=
            ast.dump(ast.parse(candidate.decode("ascii"), type_comments=True), include_attributes=False)
            or after.name != function.name):
        raise ValueError("candidate changed syntax outside the selected return expression")
    return candidate


def _public_artifact(repository, envelope):
    raw = _wire(envelope)
    if len(raw) > 2_000_000:
        raise ValueError("finite handoff exceeds worker byte bound")
    current = repository
    for name in (".runtime", "repository-finite-handoffs"):
        current = current / name
        try:
            current.mkdir(mode=0o755)
        except FileExistsError:
            pass
        info = current.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o022
                or stat.S_IMODE(info.st_mode) & 0o005 != 0o005):
            raise ValueError("finite handoff parent must be owner-controlled and worker-readable")
    artifact = current / (_sha(raw) + ".json")
    _persist(artifact, raw)
    return artifact, _sha(raw)


def _candidate_observation(*, root, source, contract, inputs, tool_policy, owner, remaining):
    """Capture and observe the proposed source through another native owner."""
    import duckdb
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
    from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
    from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import (
        observe_finite_integer_source, validate_finite_integer_observation,
    )
    repository = root / "repository"
    target = repository / contract.path
    target.parent.mkdir(parents=True, mode=0o700)
    target.write_bytes(source)
    for args in (("init", "-q"), ("add", "--", contract.path),
            ("-c", "user.name=Finite candidate owner", "-c", "user.email=qualification@example.invalid",
             "commit", "-qm", "Private candidate observation")):
        subprocess.run(["/usr/bin/git", "-C", str(repository), *args], check=True,
                       capture_output=True, timeout=remaining())
    with duckdb.connect(str(root / "candidate.duckdb"), config={"threads": 1, "memory_limit": "64MB"}) as connection:
        store = DuckDBASTStore(connection=connection)
        artifacts = ImmutableCAS(root / "artifacts")
        index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
                                       catalog=CodebaseCatalog(store, artifacts))
        head = index.prepare_current(repository, repository_id="finite-candidate:" + _sha(source),
            operation_id="candidate:initial", expected_head=None, scheduler=owner.scheduler,
            parent_lease=owner.parent_lease, cancel_event=owner.cancel_event,
            timeout_seconds=remaining(), memory_mb=owner.memory_mb).head
        observed = observe_finite_integer_source(index=index, repository=repository, expected_head=head,
            contract=contract, inputs=inputs, output=root / "observation", tool_policy=tool_policy,
            scheduler=owner.scheduler, parent_lease=owner.parent_lease, cancel_event=owner.cancel_event,
            timeout_seconds=remaining(), memory_mb=owner.memory_mb)
        validate_finite_integer_observation(observed, expected_head=head, contract=contract,
                                          inputs=inputs, tool_policy=tool_policy)
    return observed


def prepare_finite_repository_handoff(*, owner, request, intent_document, source_text,
        operation_catalog, tool_policy, policy_observer, admission, intent: IntentRepository,
        task_cid: str, state: Path, instruction_path: str | None = None) -> dict:
    """Freshly match, plan and check a candidate within independent native scope.

    No caller receipt, match, fact or candidate source can enter this boundary.
    The complete instruction is the signed task objective or immutable signed source.
    Only a nonempty offset residual is eligible; all-satisfied/no-work remains a
    proposal and cannot omit, complete or mutate an existing native task.
    """
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        LeaseCancelledError, LeaseTimeoutError,
    )
    if not isinstance(intent, IntentRepository):
        raise ValueError("native IntentRepository required")
    repository, state = owner.repository, Path(state).absolute()
    if state.exists() or state.resolve() != state or state.is_relative_to(repository):
        raise ValueError("fresh exact external finite preparation state required")
    declared = local.verify_local_benchmark_admission(admission, initial=True)
    manifest = declared["manifest"]
    task = intent.get_task(task_cid)
    native = [row for row in declared["graph"].tasks if row.task_cid == task_cid]
    if (manifest["repository"] != str(repository) or task is None or len(native) != 1
            or task["status"] != "ready"):
        raise ValueError("exact independently signed ready task and original instruction required")
    task_contract, _, _, _ = local._contract(task["body"], task_cid)
    outputs = task_contract["task_spec"]["outputs"]
    query = matcher.prepare_finite_integer_query(intent_document=intent_document, source_text=source_text)
    if not query["supported"]:
        raise ValueError("complete supported finite instruction required")
    contract = IntegerOffsetContract.from_dict(query["contract"])
    if (not query["supported"] or len(outputs) != 1 or outputs[0]["effect"] != "modify"
            or outputs[0]["path"] != contract.path or contract.path not in manifest["sources"]
            or task_contract["manifest_cid"] != declared["receipt"]["manifest_cid"]
            or task["task_alias"] != native[0].task_key):
        raise ValueError("complete finite prompt and single independently declared modification required")
    if instruction_path is None:
        if source_text != native[0].objective:
            raise ValueError("original finite instruction differs from signed task objective")
    else:
        _path(instruction_path)
        if (instruction_path == contract.path or instruction_path not in manifest["sources"]
                or _source(repository, instruction_path) != source_text.encode()
                or _sha(source_text.encode()) != manifest["sources"][instruction_path]["sha256"]):
            raise ValueError("original finite instruction differs from immutable signed input")
    source = _source(repository, contract.path)
    if _sha(source) != manifest["sources"][contract.path]["sha256"]:
        raise ValueError("finite source differs from signed baseline")
    inputs = _wire(dict(query=query, tool_policy=tool_policy,
        request=request.to_dict(), operations=operation_catalog.to_dict()))
    policy = deepcopy(tool_policy)
    deadline = time.monotonic() + min(90, owner.timeout_seconds, request.budget.max_latency_ms / 1000)

    def remaining():
        if owner.cancel_event is not None and owner.cancel_event.is_set():
            raise LeaseCancelledError("finite handoff cancelled")
        duration = deadline - time.monotonic()
        if duration <= 0:
            raise LeaseTimeoutError("finite handoff deadline exceeded")
        return duration

    def current():
        remaining()
        check = local.verify_local_benchmark_admission(admission, initial=True)
        if (check["receipt"] != declared["receipt"] or intent.get_task(task_cid) != task
                or _source(repository, contract.path) != source
                or inputs != _wire(dict(query=matcher.prepare_finite_integer_query(
                    intent_document=intent_document, source_text=source_text), tool_policy=tool_policy,
                    request=request.to_dict(), operations=operation_catalog.to_dict()))):
            raise ValueError("finite owner, task or selected input changed")
        if instruction_path is not None and _source(repository, instruction_path) != source_text.encode():
            raise ValueError("signed instruction changed during finite preparation")
        with structural_codebase_context(owner.index, repository,
                repository_id=owner.expected_head.repository_id, expected_head=owner.expected_head,
                scheduler=owner.scheduler, parent_lease=owner.parent_lease, cancel_event=owner.cancel_event,
                timeout_seconds=remaining(), memory_mb=owner.memory_mb):
            observed = policy_observer(request)
            observed.require_current(request.roots)
        remaining()

    current()
    state.mkdir(parents=True, mode=0o700)
    preview = preview_finite_integer_plan(owner=owner, request=request, intent_document=intent_document,
        source_text=source_text, operation_catalog=operation_catalog, output=state / "initial-observation",
        tool_policy=policy, policy_observer=policy_observer)
    current()
    result = dict(schema=RESULT_SCHEMA, status="residual", task_cid=task_cid,
        task_revision=task["revision"], preview=preview, candidate_observation=None,
        handoff_path=None, handoff_sha256=None, signed_evidence=None, reason_codes=[],
        source_unchanged=True, native_task_population_unchanged=True, scope=SCOPE, **FALSE)
    selected = [row for row in operation_catalog.operations if row.requirement_id == matcher.OFFSET_STATEMENT_ID]
    match = preview["match"]
    if (len(selected) != 1 or preview["selected_task_ids"] != [selected[0].task_id]
            or match["eligible_clause_ids"] != [matcher.TYPE_STATEMENT_ID]
            or match["residual_clause_ids"] != [matcher.OFFSET_STATEMENT_ID]
            or not match["finite_counterexamples"]):
        result["reason_codes"] = ["requires_one_nonempty_offset_residual_and_exact_type_fact"]
    else:
        candidate = propose_offset_source(source, contract)
        observation = _candidate_observation(root=state / "candidate", source=candidate, contract=contract,
            inputs=query["domain_inputs"], tool_policy=policy, owner=owner, remaining=remaining)
        result["candidate_observation"] = observation
        current()
        if (observation["status"] != "observed" or observation["source_sha256"] != _sha(candidate)
                or not all(observation[name] is True for name in ("type_clause_satisfied",
                    "offset_clause_satisfied", "runtime_observation_coverage_complete", "kernel_checked_model_table"))):
            result["reason_codes"] = ["candidate_did_not_pass_complete_finite_observation"]
        else:
            # Revalidate the owner's original receipt after all candidate work.
            matcher._check_observation(observation=match["observation"], index=owner.index,
                head=owner.expected_head, contract=contract, inputs=query["domain_inputs"],
                tool_policy=policy, output=state / "initial-observation")
            payload = dict(schema=SCHEMA, repository=str(repository), baseline_commit=manifest["baseline_commit"],
                task_cid=task_cid, task_id=task["task_alias"], task_revision=task["revision"],
                manifest_cid=declared["receipt"]["manifest_cid"],
                instruction_sha256=_sha(source_text.encode()), instruction_path=instruction_path,
                permitted_output=deepcopy(outputs[0]),
                edit=dict(path=contract.path, before_sha256=_sha(source), after_sha256=_sha(candidate),
                          after_bytes_base64=base64.b64encode(candidate).decode()),
                evidence=dict(query_cid=query["query_cid"], source_head=owner.expected_head.to_dict(),
                    preview_cid=preview["result_cid"], selected_operation=selected[0].to_dict(),
                    complete_requirements=query["requirement_ids"], initial_observation_cid=match["observation_cid"],
                    candidate_observation_cid=observation["result_cid"], input_sha256=_sha(inputs),
                    scope=SCOPE, inputs=query["domain_inputs"]), scope=SCOPE, **FALSE)
            payload["artifact_cid"] = content_identity(payload)
            current()
            signed = local._signed(payload, manifest)
            if local._verify_signature(signed, declared["profile"]) != payload:
                raise ValueError("finite evidence owner signature failed")
            current()
            artifact, digest = _public_artifact(repository, signed)
            result.update(status="candidate_ready", signed_evidence=signed,
                          handoff_path=str(artifact), handoff_sha256=digest)
    current()
    result["result_cid"] = cid_for_structured(result)
    _persist(state / "result.json", _wire(result))
    return result


__all__ = ["prepare_finite_repository_handoff", "propose_offset_source", "SCHEMA", "RESULT_SCHEMA"]
