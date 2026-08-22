"""Verified Residual Intelligence Foundry candidate contracts.

The package is provider-free at import time.  It deliberately exposes no
training, promotion, proof-acceptance, or mutation side effect.
"""

from .contracts import (
    ExpertDisposition,
    PrerequisiteFinding,
    PrerequisiteStatus,
    PrivacyClass,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    TrainingAvailability,
    TypedBlocker,
)
from .corpus import (
    CorpusSourceKind,
    LabelDisposition,
    ResidualDistillationCorpus,
    ResidualDistillationExample,
)
from .inventory import (
    ModelInvocationObservation,
    ResidualFamilyBoundary,
    ResidualReasoningInventory,
    TrajectoryOutcome,
)
from .residual_ir import ResidualIntelligenceIR, ResidualTaskInput, ResidualTaskOutput
from .rights import LeakageAudit, TrainingCorpusAdmission
from .splits import SemanticSplitManifest, SemanticSplitPolicy, SplitPartition
from .structured_decoding import (
    DecodeStatus,
    ExpertGrammar,
    PayloadFieldContract,
    PayloadFieldKind,
    StructuredDecodeResult,
    decode_structured_output,
    grammar_for,
)

__all__ = (
    "CorpusSourceKind",
    "DecodeStatus",
    "ExpertDisposition",
    "ExpertGrammar",
    "LabelDisposition",
    "LeakageAudit",
    "ModelInvocationObservation",
    "PayloadFieldContract",
    "PayloadFieldKind",
    "PrerequisiteFinding",
    "PrerequisiteStatus",
    "PrivacyClass",
    "ResidualDistillationCorpus",
    "ResidualDistillationExample",
    "ResidualFamilyBoundary",
    "ResidualIntelligenceError",
    "ResidualIntelligenceIR",
    "ResidualReasoningInventory",
    "ResidualTaskFamily",
    "ResidualTaskInput",
    "ResidualTaskOutput",
    "RiskClass",
    "SemanticSplitManifest",
    "SemanticSplitPolicy",
    "SplitPartition",
    "StructuredDecodeResult",
    "TrainingAvailability",
    "TrainingCorpusAdmission",
    "TrajectoryOutcome",
    "TypedBlocker",
    "decode_structured_output",
    "grammar_for",
)
