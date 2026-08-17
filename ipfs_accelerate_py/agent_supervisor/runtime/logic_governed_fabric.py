"""Bounded fixed-point convergence and terminals."""
from types import MappingProxyType
class ConvergenceError(ValueError): pass
def converge(steps, *, bound):
    if len(steps) > bound:
        raise ConvergenceError("exceeded fixed-point bound")
    terminal = steps[-1] if steps else "empty"
    return MappingProxyType({"terminal": terminal, "steps": len(steps), "bounded": True})
