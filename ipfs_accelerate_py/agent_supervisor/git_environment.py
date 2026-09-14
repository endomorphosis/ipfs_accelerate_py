"""Keep observational Git work from refreshing repository index metadata."""

from __future__ import annotations

import os
from collections.abc import Mapping


def git_subprocess_environment(
    source: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Copy the selected environment and disable Git's optional lock work.

    Status and diff may otherwise rewrite cached index stat fields, breaking
    an independently retained raw-index pin. Required writes by explicitly
    mutating Git commands still work. This does not sanitize repository
    routing or relax any source/index identity check.
    """

    inherited = os.environ if source is None else source
    result = {str(key): str(value) for key, value in inherited.items()}
    result["GIT_OPTIONAL_LOCKS"] = "0"
    return result
