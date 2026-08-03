"""Focused tests for lazy, fail-closed contract-repair capability admission."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from ipfs_accelerate_py.agent_supervisor.integrations.contract_repair_capabilities import (
    PINNED_TYPESCRIPT_VERSION,
    ContractRepairCapabilityStatus,
    ContractRepairDiagnosticCode,
    probe_contract_repair_capabilities,
)


def _module(path: str, **values):
    return SimpleNamespace(__file__=path, **values)


def _datasets_provider():
    class Kind:
        def __init__(self, value):
            self.value = value

    class Receipt:
        def __init__(self, module):
            self.module = module
            self.available = True

    class Probe:
        def __init__(self, kind):
            self.kind = Kind(kind)
            self.available = True
            self.symbol_receipts = (Receipt(f"datasets.{kind}"),)
            self.reconstruction_compatible = kind != "ir"
            self.capability_revision = f"revision:{kind}"
            self.package_version = "1.0.0"
            self.provider_id = kind

    return _module(
        "/fixture/logic_provider.py",
        DatasetsLogicBackendKind=object(),
        LOGIC_IR_INTERFACE="LogicIR@1",
        DATASETS_LOGIC_PROBE_SCHEMA="datasets-probe@1",
        probe_all_datasets_logic_backends=lambda *, importer: tuple(
            Probe(kind) for kind in ("ir", "tdfol", "cec", "smt", "hammer")
        ),
    )


def _importer(name: str):
    if name == "ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider":
        return _datasets_provider()
    if name.startswith("datasets."):
        return _module(f"/fixture/{name}.py")
    if name.endswith("program_contracts"):
        return _module(
            "/fixture/program_contracts.py",
            ExpectedProgramContract=object,
            ObservedProgramContract=object,
            ProgramContractBundle=object,
            PROGRAM_CONTRACT_VERSION=1,
            SCHEMA_VERSION=1,
        )
    raise ModuleNotFoundError(name)


def _runner(command, **_kwargs):
    executable = Path(command[0]).name
    if executable == "git":
        return SimpleNamespace(returncode=0, stdout="160000 commit d144be65ffe4c6423e4e1c30cd692812607343eb\tipfs_datasets_py\n", stderr="")
    output = {
        "node": "v18.19.1",
        "tsc": "Version 5.5.0",
        "cvc5": "cvc5 version 1.3.3",
    }.get(executable, "")
    return SimpleNamespace(returncode=0 if output else 1, stdout=output, stderr="")


def _which(executable: str):
    return {"node": "/bin/node", "tsc": "/bin/tsc", "cvc5": "/bin/cvc5"}.get(executable)


def test_probe_is_fail_closed_and_keeps_solver_routes_non_authoritative(tmp_path):
    report = probe_contract_repair_capabilities(
        importer=_importer, which=_which, runner=_runner, repository_root=tmp_path
    )

    assert report.datasets_gitlink_revision == "d144be65ffe4c6423e4e1c30cd692812607343eb"
    assert report.capability("datasets.logic_ir").available
    assert report.capability("datasets.hammer").candidate_authoritative is False
    assert report.capability("datasets.hammer").reconstruction_compatible is True
    assert report.capability("vfs.program_contract").available
    assert report.capability("vfs.program_graph").status is ContractRepairCapabilityStatus.UNAVAILABLE
    assert report.capability("toolchain.cvc5").available
    assert report.capability("toolchain.z3").status is ContractRepairCapabilityStatus.UNAVAILABLE
    assert report.capability("toolchain.mypy").status is ContractRepairCapabilityStatus.UNAVAILABLE
    assert report.capability("toolchain.typescript").status is ContractRepairCapabilityStatus.INCOMPATIBLE
    assert report.capability("toolchain.typescript").details["expected_version"] == PINNED_TYPESCRIPT_VERSION
    assert report.to_dict()["network_access"] is False
    assert report.to_dict()["auto_install"] is False


def test_missing_symbol_and_timeout_are_typed_diagnostics(tmp_path):
    def slow_importer(name: str):
        if name.endswith("program_graph"):
            time.sleep(0.05)
        return _importer(name)

    report = probe_contract_repair_capabilities(
        importer=slow_importer,
        which=_which,
        runner=_runner,
        timeout_seconds=0.001,
        repository_root=tmp_path,
    )
    graph = report.capability("vfs.program_graph")
    assert graph.status is ContractRepairCapabilityStatus.TIMED_OUT
    assert graph.diagnostic.code is ContractRepairDiagnosticCode.PROBE_TIMED_OUT


def test_version_command_timeout_is_typed(tmp_path):
    def runner(command, **_kwargs):
        if Path(command[0]).name == "node":
            raise subprocess.TimeoutExpired(command, 1)
        return _runner(command)

    report = probe_contract_repair_capabilities(
        importer=_importer, which=_which, runner=runner, repository_root=tmp_path
    )
    node = report.capability("toolchain.node")
    assert node.status is ContractRepairCapabilityStatus.TIMED_OUT
    assert node.diagnostic.code is ContractRepairDiagnosticCode.PROBE_TIMED_OUT
