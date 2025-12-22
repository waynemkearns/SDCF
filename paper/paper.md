---
title: 'SDCF: A Python Implementation of the Synthetic Data Compliance Framework'
tags:
  - Python
  - synthetic data
  - privacy
  - fairness
  - data governance
  - GDPR
  - EU AI Act
authors:
  - name: Wayne Kearns
    orcid: 0009-0000-6196-483X
    affiliation: 1
affiliations:
 - name: Nortes Consulting Ltd., Ireland
   index: 1
date: 22 December 2025
bibliography: paper.bib
---

# Summary

Synthetic data generation has emerged as a privacy-preserving approach to data sharing, analytics, and machine learning. However, organizations struggle to answer a fundamental question: is this synthetic data fit for my specific purpose? Existing quality metrics focus on statistical fidelity or theoretical privacy guarantees but fail to provide integrated assessments that connect to regulatory compliance requirements.

SDCF is a Python package that implements the Synthetic Data Compliance Framework, a methodology for evaluating synthetic data across three dimensions: privacy preservation, statistical fidelity, and demographic fairness. The package provides a tiered assessment approach that produces clear conformance decisions (Acceptable, Provisional, or Rejected) tied to specific intended uses.

# Statement of Need

Synthetic data adoption is accelerating across healthcare, finance, and public sector applications [@el2020practical; @goncalves2020generation]. Organizations deploying synthetic data face regulatory obligations under GDPR Article 32 and the EU AI Act Article 10, yet lack standardized tooling to demonstrate compliance [@veale2018fairness].

Current synthetic data evaluation tools fall into three categories. Statistical quality libraries such as SDMetrics [@sdv2023] and Synthcity [@qian2023synthcity] measure distributional similarity but do not assess privacy risk or demographic fairness. Privacy auditing tools such as TAPAS [@stadler2022synthetic] focus exclusively on re-identification risk without considering data utility. Fairness toolkits such as AI Fairness 360 [@bellamy2019ai] evaluate bias but are not designed for synthetic data contexts.

No existing tool provides integrated, purpose-bounded assessment that connects technical metrics to regulatory obligations. SDCF fills this gap.

# Core Functionality

SDCF implements three assessment tiers based on data access:

**Gold Tier** provides the highest confidence assessments when full source data is available. The package calculates Distance to Closest Record (DCR) for privacy risk, runs Kolmogorov-Smirnov tests for distributional fidelity, and measures demographic parity across protected groups.

**Silver Tier** enables medium-confidence assessments when only summary statistics or samples of source data are available. Privacy assessment uses aggregate-level analysis rather than record-level comparison.

**Bronze Tier** supports limited assessments when no source data is available. The package evaluates intrinsic data quality through completeness, validity, and consistency checks. Privacy cannot be verified at this tier, which the framework explicitly acknowledges.

Each assessment produces a conformance level:

- **SDCF-A (Acceptable)**: Synthetic data meets all thresholds for the intended purpose
- **SDCF-P (Provisional)**: Borderline results require documented usage restrictions  
- **SDCF-R (Rejected)**: Synthetic data is not fit for the intended purpose

The package includes regulatory mapping to GDPR Articles 25 and 32, EU AI Act Article 10, and ISO/IEC 27001 controls. Assessment results link directly to specific compliance requirements.

## Example Usage

```python
from sdcf.sdcf_assessor import SDCFAssessor
import pandas as pd

# Load data
source_df = pd.read_csv("source_data.csv")
synthetic_df = pd.read_csv("synthetic_data.csv")

# Create assessor
assessor = SDCFAssessor(tier='gold')

# Run assessment
results = assessor.assess(
    source_data=source_df,
    synthetic_data=synthetic_df,
    categorical_columns=['gender', 'region'],
    protected_attributes=['gender', 'age_group'],
    sensitive_attributes=['income']
)

# Check results
print(f"Conformance: {results.conformance.value}")
print(f"Privacy Score: {results.privacy.score:.2f}")
print(f"Fidelity Score: {results.fidelity.score:.2f}")
print(f"Fairness Score: {results.fairness.score:.2f}")
```

# Comparison to Existing Tools

| Tool | Privacy | Fidelity | Fairness | Regulatory Mapping | Purpose-Bounded |
|------|---------|----------|----------|-------------------|-----------------|
| SDMetrics | No | Yes | No | No | No |
| TAPAS | Yes | No | No | No | No |
| AIF360 | No | No | Yes | No | No |
| Synthcity | Partial | Yes | No | No | No |
| **SDCF** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

SDCF is the only tool that integrates all three dimensions with explicit regulatory mapping and purpose-bounded assessment logic.

# Empirical Validation

The package includes Bronze Tier validation results from ten publicly available synthetic datasets. This retrospective study evaluated the framework's intrinsic quality checks and face validity assessments. Results demonstrated that Bronze Tier assessments correctly identified quality issues in datasets with known problems while passing datasets meeting quality standards.

Key findings from the validation study:

- TVAE-generated data consistently outperformed CTGAN on fidelity metrics
- Privacy scores showed expected sensitivity to generation parameters
- Fairness metrics detected demographic skew in biased synthetic outputs

Full validation methodology and results are available in the repository's `validation/` directory.

# Target Audience

SDCF serves three primary user groups:

**Data Protection Officers** need to document that synthetic data meets GDPR requirements. SDCF provides structured assessment reports that map directly to Article 32 technical measures.

**Researchers** publishing with synthetic data need to demonstrate fitness for their specific analysis. SDCF assessments provide evidence for ethics committee reviews and journal submissions.

**Public Sector Organizations** face NIS2 requirements for data governance. SDCF's regulatory mapping supports compliance documentation for national cybersecurity authorities.

# Implementation

SDCF is implemented in Python 3.8+ with minimal dependencies: pandas, numpy, scipy, and scikit-learn. The package is designed for integration into existing data pipelines and supports both programmatic use and command-line assessment.

Source code is available at https://github.com/waynemkearns/SDCF under MIT license. Documentation includes a quick-start guide, implementation reference, and assessment templates for each conformance level.

# Acknowledgements

The framework builds on research and standards from the EU Agency for Cybersecurity (ENISA), the National Institute of Standards and Technology (NIST), and the synthetic data research community.

# References

