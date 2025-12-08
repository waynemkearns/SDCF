# SDCF Reference Implementation
# Synthetic Data Compliance Framework
# Version: 1.0
# Author: Wayne Kearns, Kaionix Labs
# License: MIT

"""
SDCF Reference Implementation Package

This package provides reference implementations of:
1. Metric computation for Privacy, Fidelity, and Fairness
2. SDCF decision logic for conformance determination
3. Certificate generation helpers

Basic Usage:
    from src.sdcf_assessor import SDCFAssessor
    
    assessor = SDCFAssessor(tier='gold')
    results = assessor.assess(
        source_data=df_source,
        synthetic_data=df_synthetic,
        categorical_columns=['gender', 'region'],
        protected_attributes=['gender'],
        sensitive_attributes=['income']
    )
    print(results.conformance)  # 'SDCF-A', 'SDCF-P', or 'SDCF-R'
"""

from .sdcf_reference_implementation import (
    AssessmentTier,
    Conformance,
    DimensionStatus,
    DimensionScore,
    SDCFResults,
    PrivacyMetrics,
    FidelityMetrics,
    FairnessMetrics
)

from .sdcf_assessor import SDCFAssessor

__version__ = "1.0.0"
__author__ = "Wayne Kearns"
__email__ = "wayne.kearns@nortesconsulting.com"
__license__ = "MIT"

__all__ = [
    "SDCFAssessor",
    "AssessmentTier",
    "Conformance",
    "DimensionStatus",
    "DimensionScore",
    "SDCFResults",
    "PrivacyMetrics",
    "FidelityMetrics",
    "FairnessMetrics"
]

