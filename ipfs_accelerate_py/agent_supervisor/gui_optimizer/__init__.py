"""VerifiedGuiOptimizer execution package (ipfs_accelerate agent_supervisor).

This package owns bounded execution for the Verified GUI Optimizer:

* security authority (patch roots, browser-host boundary, acceptance evidence);
* later tasks add patch-scope, proposal, worktree, checks, journal, and CLI.

The package is standalone.  It must not import semantic-index, semantic-capsule,
proof-cache, or model-routing modules.  Browser content never selects host
paths or process commands; UI state never synthesizes authorization.
"""

from __future__ import annotations

from typing import Final

from .authority import (
    ALWAYS_HUMAN_REVIEW_KINDS,
    AcceptanceAuthorityRequest,
    AuthorityDecision,
    AuthorityEvidence,
    AuthorityEvidenceKind,
    AuthorityReasonCode,
    AuthorityVerdict,
    BrowserHostInput,
    DEFAULT_ALLOWED_ROOTS,
    DEFAULT_FORBIDDEN_PATH_PARTS,
    FORBIDDEN_BROWSER_COMMAND_FIELDS,
    FORBIDDEN_BROWSER_PAYLOAD_KEYS,
    ForbiddenChangeKind,
    GUI_ACCEPTANCE_AUTHORITY_INTERFACE,
    GUI_ACCEPTANCE_AUTHORITY_SCHEMA,
    GUI_AUTHORITY_DECISION_SCHEMA,
    GUI_HOST_BOUNDARY_POLICY_INTERFACE,
    GUI_HOST_BOUNDARY_POLICY_SCHEMA,
    GUI_PATCH_AUTHORITY_INTERFACE,
    GUI_PATCH_AUTHORITY_SCHEMA,
    GuiAcceptanceAuthority,
    GuiAuthorityError,
    GuiHostBoundaryPolicy,
    GuiOptimizerSecurityAuthority,
    GuiPatchAuthority,
    PatchPathClaim,
    SENSITIVE_CHANGE_KINDS,
    default_security_authority,
    path_has_forbidden_segment,
    path_under_allowed_roots,
)

GUI_OPTIMIZER_PACKAGE_NAME: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.gui_optimizer"
)
GUI_OPTIMIZER_OWNED_MODULES: Final[tuple[str, ...]] = ("authority",)

__all__: Final[tuple[str, ...]] = (
    "ALWAYS_HUMAN_REVIEW_KINDS",
    "AcceptanceAuthorityRequest",
    "AuthorityDecision",
    "AuthorityEvidence",
    "AuthorityEvidenceKind",
    "AuthorityReasonCode",
    "AuthorityVerdict",
    "BrowserHostInput",
    "DEFAULT_ALLOWED_ROOTS",
    "DEFAULT_FORBIDDEN_PATH_PARTS",
    "FORBIDDEN_BROWSER_COMMAND_FIELDS",
    "FORBIDDEN_BROWSER_PAYLOAD_KEYS",
    "ForbiddenChangeKind",
    "GUI_ACCEPTANCE_AUTHORITY_INTERFACE",
    "GUI_ACCEPTANCE_AUTHORITY_SCHEMA",
    "GUI_AUTHORITY_DECISION_SCHEMA",
    "GUI_HOST_BOUNDARY_POLICY_INTERFACE",
    "GUI_HOST_BOUNDARY_POLICY_SCHEMA",
    "GUI_OPTIMIZER_OWNED_MODULES",
    "GUI_OPTIMIZER_PACKAGE_NAME",
    "GUI_PATCH_AUTHORITY_INTERFACE",
    "GUI_PATCH_AUTHORITY_SCHEMA",
    "GuiAcceptanceAuthority",
    "GuiAuthorityError",
    "GuiHostBoundaryPolicy",
    "GuiOptimizerSecurityAuthority",
    "GuiPatchAuthority",
    "PatchPathClaim",
    "SENSITIVE_CHANGE_KINDS",
    "default_security_authority",
    "path_has_forbidden_segment",
    "path_under_allowed_roots",
)
