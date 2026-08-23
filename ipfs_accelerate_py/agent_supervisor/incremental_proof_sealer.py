"""Released public IncrementalProofSealer capability (PCCE-014).

Cold import does not probe unavailable kit/datasets stores or daemons.
The sealer class is imported lazily on first use.
"""

from __future__ import annotations

from typing import Any

INTERFACE = "IncrementalProofSealer@1"
V01_PERSISTENCE = "ipfs_kit_py.proof_context.incremental_seal_store"


def IncrementalProofSealer(*args: Any, **kwargs: Any) -> Any:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.sealer import (
        IncrementalProofSealer as _Sealer,
    )

    return _Sealer(*args, **kwargs)
