"""Partitioning, eligible work stealing, and logical acceptance."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence


class PartitionError(ValueError):
    """Work steal or partition was illegal."""


def partition_tasks(tasks: Sequence[str], supervisors: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
    if not supervisors:
        raise PartitionError("no supervisors")
    assigned: dict[str, list[str]] = {name: [] for name in supervisors}
    for index, task in enumerate(sorted(tasks)):
        assigned[supervisors[index % len(supervisors)]].append(task)
    return MappingProxyType({key: tuple(value) for key, value in assigned.items()})


def steal(source: Sequence[str], target: Sequence[str], *, eligible: bool) -> Mapping[str, Any]:
    if not eligible:
        raise PartitionError("steal is not eligible")
    if not source:
        return MappingProxyType({"stolen": None, "source": tuple(source), "target": tuple(target)})
    stolen = source[-1]
    return MappingProxyType(
        {
            "stolen": stolen,
            "source": tuple(source[:-1]),
            "target": tuple(target) + (stolen,),
        }
    )
