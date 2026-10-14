"""Explicit owner selection for at most one continuation of ordinary coding."""

from __future__ import annotations


def validate_candidate_rounds(value):
    if type(value) is not int or value not in (1, 2):
        raise ValueError("candidate rounds must be exactly one or two")
    return value


def split_owner_candidate_rounds(argv):
    """Remove the owner-only control before preparing the original worker.

    Duplicates and incomplete selections refuse rather than change which
    interpretation the owner and the child see. An absent flag retains one
    round; this function does not admit a continuation.
    """
    if type(argv) is not list or any(type(arg) is not str for arg in argv):
        raise ValueError("literal owner argv required")
    result, selected = [], None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--candidate-rounds":
            index += 1
            if index == len(argv):
                raise ValueError("candidate rounds value required")
            value = argv[index]
        elif arg.startswith("--candidate-rounds="):
            value = arg.split("=", 1)[1]
        else:
            result.append(arg)
            index += 1
            continue
        if selected is not None or value not in ("1", "2"):
            raise ValueError("one exact candidate rounds selection required")
        selected = int(value)
        index += 1
    return 1 if selected is None else selected, result
