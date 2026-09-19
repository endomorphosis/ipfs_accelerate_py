"""Cross-supervisor isolation: one extra-gate cannot write another's store."""

from __future__ import annotations

from dataclasses import dataclass


class CrossSupervisorWriteError(PermissionError):
    """A supervisor attempted a direct write into another board store."""


@dataclass(frozen=True)
class SupervisorStore:
    name: str
    database_path: str


@dataclass
class IsolatedOwner:
    store: SupervisorStore

    def write(self, database_path: str) -> None:
        if database_path != self.store.database_path:
            raise CrossSupervisorWriteError(
                f"{self.store.name} cannot write {database_path}"
            )


SAWM = SupervisorStore("sawm", "/sawm/run-r2-m27/control.duckdb")
DOEP = SupervisorStore("doep", "/doep/r5/control.duckdb")
SPAR = SupervisorStore("spar", "/spar/control.duckdb")


def test_sawm_owner_cannot_write_doep_or_spar() -> None:
    owner = IsolatedOwner(SAWM)
    owner.write(SAWM.database_path)
    for foreign in (DOEP.database_path, SPAR.database_path):
        try:
            owner.write(foreign)
        except CrossSupervisorWriteError:
            continue
        raise AssertionError(f"SAWM wrote foreign store {foreign}")


def test_doep_owner_cannot_write_sawm() -> None:
    owner = IsolatedOwner(DOEP)
    try:
        owner.write(SAWM.database_path)
        raise AssertionError("DOEP wrote SAWM store")
    except CrossSupervisorWriteError:
        pass
