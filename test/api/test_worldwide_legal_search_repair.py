"""WWL-102: isolated supervisor repair and validated promotion.

A reproduced native defect opens one repair task. Grok is bound through
llm_router into a campaign-owned worktree, and only the new reproducer plus
the reproduced source frames may change. Regressions and the canary gate a
native merge, then the lane is reloaded and fenced. Rollback evidence stays.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search import CANONICAL_ROUTING
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_progress import (
    DEFAULT_LIMITS,
    ProgressLedger,
)
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_repair import (
    ALLOWED_PATTERNS,
    CAMPAIGN_REPAIR_CAP,
    CANARY_SCHEMA,
    CANONICAL_PYTHON,
    DISPOSITIONS,
    HOLD_MARKERS,
    PRIMARY_MODEL,
    PRIMARY_PROVIDER,
    REGRESSION_SCHEMA,
    REPAIR_LIMIT,
    ROUTE_POLICY,
    ROUTER_NAME,
    RepairContractError,
    RepairSession,
    admitted_repair_task_id,
    bind_repair_route,
    path_is_protected,
    repair_task_identity,
    reproducer_path_admitted,
    source_path_admitted,
)

RUN = "wwl-run-102"
TASK = "WWL-102"
SOURCE = "ipfs_accelerate_py/llm_router.py"
REPRO = "test/api/test_worldwide_legal_search_repair_repro.py"
BASELINE_REV = "a" * 40
FAILURE_KEY = "native_supervisor_defect"
BASELINE_TEXT = "def route():\n    return 'broken'\n"
FIXED_TEXT = "def route():\n    return 'healthy'\n"
REPRO_TEXT = "def test_reproduced_native_defect():\n    assert 'native_supervisor_defect'\n"
MERGE_COMMIT = "b" * 40
MERGE_RECEIPT = "sha256:" + "c" * 64
MERGE_TREE = "sha256:" + "d" * 64
ARGV = ["python3", "-m", "pytest", REPRO, "-q"]


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(text: str) -> str:
    return "sha256:" + sha_text(text)


def campaign_plan(model: str = "grok-4.7") -> dict:
    routing = json.loads(json.dumps(CANONICAL_ROUTING))
    routing["primary_model"] = model
    return {"schema": "worldwide-legal-search.plan@1", "routing": routing}


def provider() -> dict:
    return {
        "grok_bin": "/usr/bin/grok",
        "codex_bin": "/usr/bin/codex",
        "codex_model": "deployment-configured-model",
        "codex_reasoning_effort": "high",
    }


def failure(**extra) -> dict:
    body = {
        "schema": "wwl.native-failure@1",
        "run_id": RUN,
        "task_id": TASK,
        "stage": "task",
        "defect": {"failure_key": FAILURE_KEY, "message": "pid=4 attempt=1"},
        "effects_unknown": False,
        "source_revision": BASELINE_REV,
        "reproducer": {"path": REPRO, "argv": list(ARGV)},
        "source_frames": [{"path": SOURCE, "source_sha256": sha_text(BASELINE_TEXT)}],
        "baseline_files": {SOURCE: BASELINE_TEXT},
        "receipt_id": digest("failure"),
        "pid": 4,
        "attempt": 1,
    }
    body.update(extra)
    return body


def edits(source: str = FIXED_TEXT, repro: str = REPRO_TEXT, *, source_path: str = SOURCE, repro_path: str = REPRO) -> list:
    return [
        {"path": repro_path, "role": "reproducer", "text": repro},
        {"path": source_path, "role": "source", "text": source},
    ]


def named_defect(key: str) -> dict:
    """Failure plus a reproducer that names that exact defect."""
    return {
        "failure": failure(defect={"failure_key": key, "message": "pid=4 attempt=1"}),
        "edits": edits(repro=REPRO_TEXT.replace(FAILURE_KEY, key)),
    }


def request(tmp_path: Path, action: str = "action-1", **extra) -> dict:
    campaign = tmp_path / "campaign"
    live = tmp_path / "live"
    campaign.mkdir(parents=True, exist_ok=True)
    live.mkdir(parents=True, exist_ok=True)
    sentinel = live / "sentinel.txt"
    if not sentinel.exists():
        sentinel.write_text("live\n", encoding="utf-8")
    body = {
        "action_id": action,
        "observed_at": 100,
        "deadline_at": 500,
        "failure": failure(),
        "campaign_root": str(campaign),
        "live_checkout": str(live),
        "plan": campaign_plan(),
        "provider": provider(),
        "python_executable": CANONICAL_PYTHON,
        "edits": edits(),
        "completion_authority": False,
    }
    body.update(extra)
    return body


def flag(argv: list, name: str) -> str:
    return argv[argv.index(name) + 1]


class Gates:
    def __init__(self) -> None:
        self.regressions = 0
        self.canaries = 0
        self.merges = 0
        self.reloads = 0
        self.fail_regression = False
        self.fail_canary = False
        self.canary_promotes = False
        self.merge_unknown = False
        self.fail_reload = False
        self.drift = False
        self.contexts: list[dict] = []
        self.session: RepairSession | None = None

    def regression(self, context: dict) -> dict:
        self.regressions += 1
        self.contexts.append(context)
        if self.session is not None and self.regressions == 1 and context.get("repair_task_id"):
            nested = getattr(self, "nested_request", None)
            if nested is not None and "inner" not in self.__dict__:
                self.inner = self.session.repair(nested)
        if self.drift:
            Path(context["worktree"], "notes.txt").write_text("drift\n", encoding="utf-8")
        passed = not self.fail_regression
        return {
            "schema": REGRESSION_SCHEMA,
            "baseline_failure_verified": passed,
            "candidate_passed": passed,
            "focused_passed": passed,
            "protected_hashes_unchanged": True,
            "exit_status": 0 if passed else 1,
            "argv": list(context["reproducer_argv"]),
            "completion_authority": False,
        }

    def canary(self, context: dict) -> dict:
        self.canaries += 1
        verified = not self.fail_canary
        return {
            "schema": CANARY_SCHEMA,
            "baseline_failure_verified": verified,
            "candidate_regressions_verified": verified,
            "repair_canary_verified": verified,
            "validated_promotion": self.canary_promotes,
            "paid_provider_calls": 0,
            "hosted_model_invocations": 0,
            "completion_authority": False,
        }

    def merge(self, context: dict) -> dict:
        self.merges += 1
        if self.merge_unknown:
            return {"merged": False, "effects_unknown": True, "completion_authority": False}
        return {
            "merged": True,
            "merge_commit": MERGE_COMMIT,
            "merge_receipt": MERGE_RECEIPT,
            "tree": MERGE_TREE,
            "effects_unknown": False,
            "completion_authority": False,
        }

    def reload(self, context: dict) -> dict:
        self.reloads += 1
        if self.fail_reload:
            return {"reloaded": False, "fenced": False, "lane_revision": "", "completion_authority": False}
        return {
            "reloaded": True,
            "fenced": True,
            "lane_revision": context["merge"]["merge_commit"],
            "completion_authority": False,
        }


def open_session(gates: Gates | None = None, **kwargs) -> tuple[RepairSession, Gates]:
    gates = gates or Gates()
    owner = RepairSession(
        RUN,
        regression_runner=gates.regression,
        canary_runner=gates.canary,
        merger=gates.merge,
        reloader=gates.reload,
        **kwargs,
    )
    gates.session = owner
    return owner, gates


def test_contract_constants_and_incident_dedup():
    assert REPAIR_LIMIT == DEFAULT_LIMITS["repair"] == 2
    assert CAMPAIGN_REPAIR_CAP == 8
    assert DEFAULT_LIMITS["provider"] == 4
    assert ROUTER_NAME == "ipfs_accelerate_py.llm_router"
    assert PRIMARY_PROVIDER == "grok_cli"
    assert PRIMARY_MODEL == "grok-4.7"
    assert ROUTE_POLICY == "grok_quota_auth_or_unavailable"
    assert CANONICAL_PYTHON == "/usr/bin/python3.12"
    assert HOLD_MARKERS == ("HOLD", "STOP", "OPERATOR_STOP")
    for name in (
        "reloaded", "duplicate_writer", "already_promoted", "effects_unknown", "held",
        "deadline_exceeded", "repair_budget_exhausted", "campaign_repair_cap", "scope_refused",
        "worktree_refused", "route_refused", "reproducer_required", "regressions_failed",
        "canary_failed", "promotion_refused", "reload_failed",
    ):
        assert name in DISPOSITIONS
    assert "ipfs_accelerate_py/llm_router.py" in ALLOWED_PATTERNS
    assert source_path_admitted(SOURCE) is True
    assert source_path_admitted("test/api/test_worldwide_legal_search_repair.py") is False
    assert reproducer_path_admitted(REPRO) is True
    assert reproducer_path_admitted("test/api/test_other.py") is False
    assert path_is_protected("ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_completion.py") is True
    assert path_is_protected("watchdog.policy.json") is True
    assert path_is_protected("docs/worldwide/product-acceptance.json") is True
    first = repair_task_identity(
        run_id=RUN, task_id=TASK, stage="implementation",
        defect={"failure_key": FAILURE_KEY, "message": "pid=1 attempt=1 heartbeat_at=9"},
    )
    second = repair_task_identity(
        run_id=RUN, task_id=TASK, stage="task",
        defect={"failure_key": FAILURE_KEY, "message": "pid=99 attempt=7"},
    )
    assert first == second
    assert admitted_repair_task_id(first) == "WWL-REPAIR-" + first.split(":", 1)[1][:16]


def test_grok_route_edits_only_the_campaign_worktree_then_promotes(tmp_path: Path):
    owner, gates = open_session()
    decision = owner.repair(request(tmp_path))
    assert decision["disposition"] == "reloaded"
    assert decision["promoted"] is True
    assert decision["reloaded"] is True
    assert decision["fenced"] is True
    assert decision["deduplicated"] is False
    assert decision["completion_authority"] is False
    assert decision["edit_order"] == [REPRO, SOURCE]
    assert decision["budget"]["repair"]["spent"] == 1
    assert decision["budget"]["provider"]["spent"] == 1
    assert decision["budget"]["validation"]["spent"] == 1
    assert decision["budget"]["merge"]["spent"] == 1
    assert decision["budget"]["repair"]["charged"] is True
    assert decision["budget"]["merge"]["charged"] is True
    route = decision["route"]
    assert route["router"] == ROUTER_NAME
    assert route["primary_provider"] == PRIMARY_PROVIDER
    assert route["primary_model"] == PRIMARY_MODEL
    assert route["fallback_policy"] == ROUTE_POLICY
    assert route["launched"] is False
    assert route["paid_provider_calls"] == 0
    assert route["hosted_model_invocations"] == 0
    assert route["attempt"] == 1
    argv = route["argv"]
    primary = json.loads(flag(argv, "--primary-command-json"))
    fallback = json.loads(flag(argv, "--fallback-command-json"))
    worktree = Path(decision["worktree"]).resolve()
    live = (tmp_path / "live").resolve()
    assert Path(flag(argv, "--workspace")).resolve() == worktree
    assert Path(flag(primary, "--workspace")).resolve() == worktree
    assert Path(flag(fallback, "-C")).resolve() == worktree
    assert live != worktree
    assert primary[primary.index("--model") + 1] == "grok-4.7"
    assert primary[primary.index("--mode") + 1] == "agent"
    assert Path(argv[1]).name == "provider_fallback_runner.py"
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert (worktree / REPRO).read_text(encoding="utf-8") == REPRO_TEXT
    assert (worktree / SOURCE).read_text(encoding="utf-8") == FIXED_TEXT
    assert (live / "sentinel.txt").read_text(encoding="utf-8") == "live\n"
    assert not (live / REPRO).exists()
    assert not (live / SOURCE).exists()
    marker = json.loads((worktree / ".wwl-repair-owner.json").read_text(encoding="utf-8"))
    assert marker["schema"] == "wwl.repair.worktree@1"
    assert marker["run_id"] == RUN
    assert marker["fenced"] is True
    assert marker["completion_authority"] is False
    rollback = decision["rollback"]
    assert rollback["retained"] is True
    assert rollback["baseline_revision"] == BASELINE_REV
    assert rollback["promoted_revision"] == MERGE_COMMIT
    assert rollback["merge_receipt"] == MERGE_RECEIPT
    assert rollback["invalidated_evidence"] == ["baseline:" + BASELINE_REV, digest("failure")]
    assert json.loads((worktree / "rollback.json").read_text(encoding="utf-8"))["retained"] is True
    evidence = json.loads((worktree / "artifacts" / "worldwide-legal-search" / "repairs" / f"{decision['repair_task_id']}.json").read_text(encoding="utf-8"))
    assert evidence["completion_authority"] is False
    assert evidence["rollback_retained"] is True
    assert gates.merges == 1 and gates.reloads == 1
    replay = owner.repair(request(tmp_path))
    assert replay["deduplicated"] is True
    assert replay["disposition"] == "reloaded"
    assert gates.merges == 1
    again = owner.repair(request(tmp_path, action="action-2", failure=failure(pid=99, attempt=7, defect={"failure_key": FAILURE_KEY, "message": "pid=99 attempt=7"})))
    assert again["disposition"] == "already_promoted"
    assert again["repair_task_id"] == decision["repair_task_id"]
    assert again["budget"]["repair"]["spent"] == 1
    assert gates.merges == 1
    assert owner.rollback_evidence(decision["repair_task_id"])[0]["baseline_revision"] == BASELINE_REV


def test_refusals_do_not_open_a_writer_or_touch_the_live_tree(tmp_path: Path):
    owner, gates = open_session()
    unknown = owner.repair(request(tmp_path, action="unknown", failure=failure(effects_unknown=True)))
    assert unknown["disposition"] == "effects_unknown"
    assert unknown["applied"] is False
    assert unknown["budget"]["repair"]["spent"] == 0
    assert unknown["fenced"] is False
    late = owner.repair(request(tmp_path, action="late", observed_at=900, deadline_at=500))
    assert late["disposition"] == "deadline_exceeded"
    assert late["budget"]["repair"]["spent"] == 0
    owner.set_hold(True, marker="STOP")
    held = owner.repair(request(tmp_path, action="held"))
    assert held["disposition"] == "held"
    assert held["budget"]["repair"]["charged"] is False
    owner.set_hold(False)
    unordered = owner.repair(request(tmp_path, action="order", edits=list(reversed(edits()))))
    assert unordered["disposition"] == "reproducer_required"
    assert unordered["budget"]["repair"]["spent"] == 0
    existing = failure(baseline_files={SOURCE: BASELINE_TEXT, REPRO: "def test_old():\n    assert False\n"})
    reused = owner.repair(request(tmp_path, action="existing", failure=existing, edits=edits()))
    assert reused["disposition"] == "reproducer_required"
    mismatched = failure(source_frames=[{"path": SOURCE, "source_sha256": "0" * 64}])
    scope = owner.repair(request(tmp_path, action="hash", failure=mismatched))
    assert scope["disposition"] == "scope_refused"
    assert scope["budget"]["repair"]["spent"] == 0
    drifted = owner.repair(request(tmp_path, action="drift", plan=campaign_plan("grok-4")))
    assert drifted["disposition"] == "route_refused"
    assert drifted["budget"]["repair"]["spent"] == 0
    assert drifted["fenced"] is False
    bare = RepairSession(RUN)
    unbound = bare.repair(request(tmp_path, action="unbound"))
    assert unbound["disposition"] == "validation_unbound"
    assert not (tmp_path / "live" / REPRO).exists()
    assert (tmp_path / "live" / "sentinel.txt").read_text(encoding="utf-8") == "live\n"
    assert gates.merges == 0
    with pytest.raises(RepairContractError, match="protected"):
        owner.repair(request(
            tmp_path, action="protected",
            failure=failure(
                source_frames=[{"path": "ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_completion.py", "source_sha256": sha_text("x")}],
                baseline_files={"ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_completion.py": "x"},
            ),
            edits=edits(source="y", source_path="ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_completion.py"),
        ))
    with pytest.raises(RepairContractError, match="escapes"):
        owner.repair(request(tmp_path, action="escape", edits=edits(source_path="../llm_router.py")))
    with pytest.raises(RepairContractError, match="outside"):
        owner.repair(request(
            tmp_path, action="readme",
            failure=failure(source_frames=[{"path": "README.md", "source_sha256": sha_text("x")}], baseline_files={"README.md": "x"}),
            edits=edits(source="y", source_path="README.md"),
        ))
    with pytest.raises(RepairContractError, match="completion authority"):
        owner.repair(request(tmp_path, action="grant", completion_authority=True))


def test_duplicate_writer_and_foreign_worktree_are_refused(tmp_path: Path):
    gates = Gates()
    owner, gates = open_session(gates)
    gates.nested_request = request(tmp_path, action="action-nested")
    decision = owner.repair(request(tmp_path))
    assert decision["disposition"] == "reloaded"
    assert gates.inner["disposition"] == "duplicate_writer"
    assert gates.inner["duplicate_writer"] is True
    assert gates.inner["budget"]["repair"]["charged"] is False
    assert gates.merges == 1
    replay = owner.repair(request(tmp_path, action="action-nested"))
    assert replay["deduplicated"] is True
    assert replay["disposition"] == "duplicate_writer"
    other = tmp_path / "other-campaign"
    other.mkdir()
    live = other / "live"
    live.mkdir()
    (live / "sentinel.txt").write_text("live\n", encoding="utf-8")
    incident = repair_task_identity(run_id=RUN, task_id=TASK, stage="task", defect={"failure_key": "foreign-defect"})
    task_id = admitted_repair_task_id(incident)
    occupied = other / "campaign" / "repair-worktrees" / task_id
    occupied.mkdir(parents=True)
    (occupied / ".wwl-repair-owner.json").write_text(json.dumps({
        "schema": "wwl.repair.worktree@1", "run_id": "other-run", "repair_task_id": task_id,
        "fenced": False, "fence_generation": 0, "completion_authority": False,
    }), encoding="utf-8")
    foreign = owner.repair(request(
        tmp_path, action="foreign",
        campaign_root=str(other / "campaign"), live_checkout=str(live),
        **named_defect("foreign-defect"),
    ))
    assert foreign["disposition"] == "worktree_refused"
    assert foreign["budget"]["repair"]["spent"] == 0
    linked = tmp_path / "linked-campaign"
    linked.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (linked / "repair-worktrees").symlink_to(outside, target_is_directory=True)
    link_live = tmp_path / "link-live"
    link_live.mkdir()
    (link_live / "sentinel.txt").write_text("live\n", encoding="utf-8")
    symlink = owner.repair(request(
        tmp_path, action="symlink",
        campaign_root=str(linked), live_checkout=str(link_live),
        **named_defect("symlink-defect"),
    ))
    assert symlink["disposition"] == "worktree_refused"
    assert not (live / REPRO).exists()


def test_regressions_canary_and_merge_effects_gate_promotion(tmp_path: Path):
    gates = Gates()
    gates.fail_regression = True
    owner, gates = open_session(gates)
    failed = owner.repair(request(tmp_path, action="red"))
    assert failed["disposition"] == "regressions_failed"
    assert failed["fenced"] is True
    assert failed["promoted"] is False
    assert failed["rollback"]["retained"] is True
    assert failed["rollback"]["baseline_revision"] == BASELINE_REV
    assert failed["budget"]["repair"]["spent"] == 1
    assert failed["budget"]["validation"]["charged"] is True
    assert failed["budget"]["merge"]["spent"] == 0
    assert gates.canaries == 0 and gates.merges == 0
    gates.fail_regression = False
    gates.canary_promotes = True
    claimed = owner.repair(request(tmp_path, action="self-promote"))
    assert claimed["disposition"] == "canary_failed"
    assert "cannot promote" in claimed["reason"]
    assert gates.merges == 0
    gates.canary_promotes = False
    gates.drift = True
    drifted = owner.repair(request(tmp_path, action="drift-file", **named_defect("drift")))
    assert drifted["disposition"] == "scope_refused"
    assert gates.merges == 0
    assert (tmp_path / "live" / "sentinel.txt").read_text(encoding="utf-8") == "live\n"
    fresh, fresh_gates = open_session()
    fresh_gates.merge_unknown = True
    unknown = fresh.repair(request(tmp_path / "merge", action="unknown-merge"))
    assert unknown["disposition"] == "effects_unknown"
    assert unknown["promoted"] is False
    assert unknown["reloaded"] is False
    assert unknown["fenced"] is True
    assert unknown["rollback"]["retained"] is True
    assert unknown["rollback"]["promoted_revision"] == ""
    assert fresh_gates.reloads == 0
    assert unknown["budget"]["merge"]["charged"] is True


def test_reload_failure_keeps_rollback_and_a_later_reload_fences(tmp_path: Path):
    gates = Gates()
    gates.fail_reload = True
    owner, gates = open_session(gates)
    failed = owner.repair(request(tmp_path))
    assert failed["disposition"] == "reload_failed"
    assert failed["promoted"] is True
    assert failed["reloaded"] is False
    assert failed["fenced"] is True
    assert failed["rollback"]["promoted_revision"] == MERGE_COMMIT
    assert failed["rollback"]["baseline_revision"] == BASELINE_REV
    assert failed["budget"]["repair"]["spent"] == 1
    blocked = owner.repair(request(tmp_path, action="second-writer"))
    assert blocked["disposition"] == "reload_failed"
    assert blocked["budget"]["repair"]["charged"] is False
    assert blocked["budget"]["repair"]["spent"] == 1
    assert gates.merges == 1
    gates.fail_reload = False
    missing = owner.repair(request(tmp_path, action="not-pending", phase="reload", failure=failure(defect={"failure_key": "other"})))
    assert missing["disposition"] == "reload_not_pending"
    reloaded = owner.repair(request(tmp_path, action="reload", phase="reload"))
    assert reloaded["disposition"] == "reloaded"
    assert reloaded["reloaded"] is True
    assert reloaded["promoted"] is True
    assert reloaded["budget"]["repair"]["spent"] == 1
    assert reloaded["budget"]["repair"]["charged"] is False
    assert gates.reloads == 2
    assert gates.merges == 1
    marker = json.loads((Path(reloaded["worktree"]) / ".wwl-repair-owner.json").read_text(encoding="utf-8"))
    assert marker["fenced"] is True
    records = owner.rollback_evidence(reloaded["repair_task_id"])
    assert len(records) >= 2
    assert all(item["retained"] is True and item["baseline_revision"] == BASELINE_REV for item in records)


def test_budgets_checkpoint_and_ledger_do_not_refund(tmp_path: Path):
    gates = Gates()
    gates.fail_regression = True
    owner, gates = open_session(gates)
    first = owner.repair(request(tmp_path, action="spend-1"))
    second = owner.repair(request(tmp_path, action="spend-2"))
    assert first["disposition"] == second["disposition"] == "regressions_failed"
    blocked = owner.repair(request(tmp_path, action="spend-3"))
    assert blocked["disposition"] == "repair_budget_exhausted"
    assert blocked["budget"]["repair"]["spent"] == 2
    assert blocked["budget"]["repair"]["remaining"] == 0
    assert gates.regressions == 2
    checkpoint = tmp_path / "repair-state.json"
    owner.save(checkpoint)
    restored = RepairSession.load(
        checkpoint,
        regression_runner=gates.regression,
        canary_runner=gates.canary,
        merger=gates.merge,
        reloader=gates.reload,
    )
    assert restored.spent["repair"][first["incident_id"]] == 2
    assert restored.campaign_repairs == 2
    assert restored.rollback_evidence(first["repair_task_id"])
    raw = json.loads(checkpoint.read_text(encoding="utf-8"))
    raw["spent"]["repair"][first["incident_id"]] = 0
    checkpoint.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RepairContractError, match="rewritten"):
        RepairSession.load(checkpoint)
    with pytest.raises(RepairContractError, match="durable repair bound"):
        RepairSession(RUN, ledger=ProgressLedger(RUN, limits={"repair": 3}))
    ledger = ProgressLedger(RUN)
    ledger.observe({
        "run_id": RUN, "task_id": TASK, "stage": "task", "observed_at": 1,
        "defect": {"failure_key": FAILURE_KEY, "message": "pid=4 attempt=1"},
    })
    bound_gates = Gates()
    bound_gates.fail_canary = True
    bound, bound_gates = open_session(bound_gates, ledger=ledger)
    canary_failed = bound.repair(request(tmp_path / "ledger", action="ledger-1"))
    incident = canary_failed["incident_id"]
    assert ledger.incidents[incident]["spent"]["repair"] == 1
    assert ledger.incidents[incident]["spent"]["provider"] == 1
    assert ledger.incidents[incident]["spent"]["validation"] == 1
    assert ledger.incidents[incident]["spent"]["merge"] == 0
    assert canary_failed["disposition"] == "canary_failed"
    bound_gates.fail_canary = False
    promoted = bound.repair(request(tmp_path / "ledger", action="ledger-2"))
    assert promoted["disposition"] == "reloaded"
    assert ledger.incidents[incident]["spent"] == {
        "task": 0, "validation": 2, "merge": 1, "provider": 2, "repair": 2, "recover": 0,
    }
    exhausted = bound.repair(request(tmp_path / "ledger", action="ledger-3"))
    assert exhausted["disposition"] == "already_promoted"
    assert exhausted["budget"]["repair"]["spent"] == 2
    assert exhausted["budget"]["repair"]["charged"] is False
    assert bound_gates.merges == 1
    empty = RepairSession(RUN, ledger=ProgressLedger(RUN), regression_runner=bound_gates.regression, canary_runner=bound_gates.canary, merger=bound_gates.merge, reloader=bound_gates.reload)
    with pytest.raises(RepairContractError, match="no such incident"):
        empty.repair(request(tmp_path / "empty", action="missing-incident"))


def test_campaign_cap_and_redirected_route_do_not_edit_the_live_tree(tmp_path: Path):
    gates = Gates()
    gates.fail_regression = True
    owner, gates = open_session(gates)
    for index in range(CAMPAIGN_REPAIR_CAP):
        decision = owner.repair(request(
            tmp_path, action=f"cap-{index}",
            **named_defect(f"defect-{index}"),
        ))
        assert decision["disposition"] == "regressions_failed"
    blocked = owner.repair(request(
        tmp_path, action="cap-over",
        **named_defect("defect-over"),
    ))
    assert blocked["disposition"] == "campaign_repair_cap"
    assert blocked["budget"]["repair"]["spent"] == 0
    assert gates.regressions == CAMPAIGN_REPAIR_CAP
    live = tmp_path / "redirect-live"
    campaign = tmp_path / "redirect-campaign"
    live.mkdir()
    campaign.mkdir()
    (live / "sentinel.txt").write_text("live\n", encoding="utf-8")

    def redirect(plan, config, *, workspace, task_id, attempt, python_executable):
        binding = bind_repair_route(
            plan, config, workspace=workspace, task_id=task_id, attempt=attempt, python_executable=python_executable,
        )
        argv = list(binding["argv"])
        argv[argv.index("--workspace") + 1] = str(live)
        binding["argv"] = argv
        return binding

    redirected, redirected_gates = open_session(Gates(), route_binder=redirect)
    refused = redirected.repair(request(
        tmp_path, action="redirect",
        campaign_root=str(campaign), live_checkout=str(live),
        **named_defect("redirect-defect"),
    ))
    assert refused["disposition"] == "route_refused"
    assert refused["budget"]["repair"]["spent"] == 1
    assert refused["budget"]["provider"]["spent"] == 0
    assert refused["fenced"] is True
    assert refused["rollback"]["retained"] is True
    assert (live / "sentinel.txt").read_text(encoding="utf-8") == "live\n"
    assert redirected_gates.merges == 0


def test_contract_document_and_receipt_match_the_binding():
    document = Path("docs/worldwide/repair-contract.md").read_text(encoding="utf-8")
    receipt = json.loads(Path("artifacts/worldwide-legal-search/receipts/WWL-102.json").read_text(encoding="utf-8"))
    for phrase in (
        "ipfs_accelerate_py.llm_router",
        "grok-4.7",
        "campaign-owned repair worktree",
        "reproducer",
        "native merge",
        "rollback",
        "fence",
        "completion authority",
    ):
        assert phrase in document
    assert receipt["schema"] == "worldwide-legal-search.task-receipt@1"
    assert receipt["task_id"] == "WWL-102"
    assert receipt["goal_id"] == "WWL-G090"
    assert receipt["subgoal_id"] == "WWL-G090.S2"
    assert receipt["depends_on"] == ["WWL-101", "WWL-003"]
    assert receipt["repair_limit"] == 2
    assert receipt["campaign_repair_cap"] == 8
    assert receipt["router"] == ROUTER_NAME
    assert receipt["primary_model"] == PRIMARY_MODEL
    assert receipt["completion_authority"] is False
    assert receipt["self_certified_completion"] is False
    assert receipt["execution_completed"] is False
    assert "docs/worldwide/repair-contract.md" in receipt["outputs"]
    assert "ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_repair.py" in receipt["outputs"]
    assert receipt["validation_argv"] == ["python3", "-m", "pytest", "test/api/test_worldwide_legal_search_repair.py", "-q"]
    assert "does not grant campaign completion" in receipt["limitations"][-1]
