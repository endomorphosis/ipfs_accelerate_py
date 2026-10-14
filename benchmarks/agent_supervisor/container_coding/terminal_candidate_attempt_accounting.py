"""Join native owner child observations to distinct observed router receipts.

Private lifecycle evidence determines possible attempts. Log rows can nominate
or describe a receipt, but cannot remove a possible child from accounting.
This is a bounded single-owner qualification, not billing verification.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

from .terminal_context_audit import _open_directory, _identity, _digest
from .terminal_output_observation import _read


def _need(condition):
    if not condition:
        raise ValueError("owner round accounting binding differs")


def _base():
    return dict(schema="terminal-candidate-attempt-accounting@1", status="not_selected",
        reason="no_candidate_round_selection", rounds=[], potential_unreceipted_attempt=False,
        complete_usage_established=False, observation_only=True,
        proof_authority=False, execution_authority=False, completion_authority=False)


def _unavailable(result):
    result.update(status="unavailable", reason="owner_round_evidence_unavailable",
        potential_unreceipted_attempt=True)


def _private_lifecycle(parent):
    from ipfs_accelerate_py.agent_supervisor.runtime.candidate_round_lifecycle import MAX_BYTES, parse_retention, lifecycle_name
    from ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context import _protected_directory
    _need(os.getuid() == os.geteuid() and os.getuid() > 0)
    parent = Path(parent)
    try:
        parent.lstat()
    except FileNotFoundError:
        return None
    _protected_directory(parent, owner=os.geteuid(), group=os.geteuid())
    fd = _open_directory(parent)
    try:
        names, archives = [], []
        with os.scandir(fd) as entries:
            for number, entry in enumerate(entries):
                _need(number < 4096)
                if entry.name.startswith("candidate-rounds-"):
                    names.append(entry.name)
                    _need(len(names) <= 2)
                elif entry.name.startswith("candidate-audit-"):
                    archives.append(entry.name)
                    _need(len(archives) <= 2)
        if not names:
            # Preparation publishes this archive before a possible second
            # child. Owner death may leave no final lifecycle or log marker.
            # Even partial publication cannot establish that usage was zero.
            _need(not archives)
            return None
        # A sibling owner cannot be selected away by a forged log nomination.
        _need(len(names) == 1)
        name = names[0]
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o400
            and info.st_uid == info.st_gid == os.geteuid() and info.st_nlink == 1)
        raw = _read(fd, name, limit=MAX_BYTES, private=True)
        current = os.stat(name, dir_fd=fd, follow_symlinks=False)
        _need((info.st_dev, info.st_ino, info.st_mode, info.st_gid) ==
              (current.st_dev, current.st_ino, current.st_mode, current.st_gid))
        retained = parse_retention(raw)
        _need(name == lifecycle_name(retained["prior_session_id"]))
        if archives:
            rounds = retained["observation"]["rounds"]
            _need(len(rounds) == 2)
            archive = "candidate-audit-" + rounds[1]["session_id"]
            # Archives may only reduce confidence here. Their contents are
            # checked by the separate input audit, never counted as receipts.
            _need(set(archives) <= {archive, archive + ".json"})
        _protected_directory(parent, owner=os.geteuid(), group=os.geteuid())
        return raw, retained
    finally:
        os.close(fd)


def collect(*, parent: Path, report: dict):
    from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_views import canonical_json
    result = _base()
    selected = False
    try:
        _need(type(report) is dict)
        rows = report.get("provider_invocations", [])
        markers = report.get("owner_candidate_rounds", [])
        selected = report.get("candidate_rounds") == 2 or bool(markers)
        _need(type(rows) is list and len(rows) <= 128 and all(type(row) is dict for row in rows))
        candidates = [row for row in rows if "candidate_round_dispatch" in row]
        selected = selected or bool(candidates)
        try:
            private = _private_lifecycle(parent)
        except Exception:
            # Private evidence cannot be made invisible by omitting log markers.
            _unavailable(result)
            return result
        if private is None:
            if selected:
                _unavailable(result)
            return result
        selected = True
        _unavailable(result)
        raw, retained = private
        _need(len(canonical_json(rows)) <= 2*1024*1024)
        identities = [_identity(row["invocation_id"]) for row in rows]
        _need(len(identities) == len(set(identities)))
        _need(type(markers) is list and len(markers) <= 8 and all(type(marker) is dict for marker in markers))
        prior = retained["prior_session_id"]
        for marker in markers:
            _need(marker.get("schema") == "owner-candidate-rounds-observation@1"
                and type(marker.get("rounds")) is list and 1 <= len(marker["rounds"]) <= 2
                and marker["rounds"][0].get("session_id") == prior)
        coding = [row for row in rows if row.get("phase") == "coding"]
        _need(all(row.get("phase") == "coding" for row in candidates))
        used, matched = set(), []
        for attempt in retained["observation"]["rounds"]:
            matches = []
            for row in coding:
                binding = row.get("source_view_retrieval") if attempt["round_number"] == 1 else row.get("candidate_round_dispatch")
                if type(binding) is not dict or binding.get("session_id") != attempt["session_id"]:
                    continue
                _need(row.get("schema") == "router-implementation-invocation@1" and row.get("purpose") == "coding"
                    and binding["descriptor_sha256"] == attempt["descriptor_sha256"]
                    and binding["task_id"] == retained["task_id"]
                    and binding["native_prompt_sha256"] == row["native_prompt_sha256"] == retained["native_prompt_sha256"]
                    and type(row["workspace"]) is str and 0 < len(row["workspace"].encode()) <= 4096
                    and hashlib.sha256(row["workspace"].encode()).hexdigest() == retained["workspace_sha256"]
                    and row["invocation_id"] not in used
                    and all(binding.get(key) is False for key in ("proof_authority", "execution_authority", "completion_authority")))
                _digest(row["native_prompt_sha256"])
                if attempt["round_number"] == 1:
                    _need(binding.get("schema") in {"worker-source-dispatch-observation@1",
                        "worker-source-dispatch-observation@2", "worker-source-dispatch-observation@3",
                        "worker-source-dispatch-observation@4"} and "candidate_round_dispatch" not in row)
                else:
                    _need(binding.get("schema") == "supervisor-candidate-dispatch-observation@1"
                        and binding.get("prior_session_id") == prior
                        and binding.get("prior_descriptor_sha256") == retained["prior_descriptor_sha256"]
                        and type(binding.get("round_number")) is int and binding["round_number"] == 2
                        and "source_view_retrieval" not in row)
                matches.append(row)
            _need(len(matches) <= 1 and (not matches or attempt["process_started"] is True))
            if matches:
                used.add(matches[0]["invocation_id"])
            matched.append(dict(round_number=attempt["round_number"], process_started=attempt["process_started"],
                receipt_observed=bool(matches), possible_unreceipted_attempt=attempt["process_started"] is not False and not matches))
        _need(len(used) == len(coding))
        missing = any(row["possible_unreceipted_attempt"] for row in matched)
        result.update(status="incomplete" if missing else "matched_round_receipts",
            reason="potential_worker_attempt_without_router_receipt" if missing else "each_started_child_has_distinct_router_receipt",
            rounds=matched, potential_unreceipted_attempt=missing,
            owner_lifecycle_sha256=hashlib.sha256(raw).hexdigest())
    except Exception:
        # Malformed receipt sets cannot establish either zero usage or complete
        # candidate accounting. Preserve a conservative unknown observation.
        _unavailable(result)
    return result


def apply(report, *, parent):
    result = collect(parent=parent, report=report)
    report["candidate_attempt_accounting"] = result
    if result["potential_unreceipted_attempt"] and not report.get("unreceipted_provider_attempt"):
        report["unreceipted_provider_attempt"] = dict(phase="coding", usage=None,
            reason="candidate_round_accounting_incomplete")
    return result
