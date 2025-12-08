# SDCF Implementation Guide
## Technical Reference for the Python Implementation

**Version:** 1.95  
**Author:** Wayne Kearns, Kaionix Labs

---

## Overview

This guide provides technical details for using the SDCF Python reference implementation.

---

## Installation

### Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- scikit-learn >= 0.24.0

### Install

```bash
pip install -r requirements.txt
```

---

## Core Classes

### SDCFAssessor

The main class for running SDCF assessments.

```python
from src.sdcf_assessor import SDCFAssessor

assessor = SDCFAssessor(tier='gold')  # 'gold', 'silver', or 'bronze'
```

### Assessment Method

```python
results = assessor.assess(
    source_data=df_source,           # Required for Gold/Silver
    synthetic_data=df_synthetic,     # Required
    categorical_columns=['col1'],    # Optional
    protected_attributes=['gender'], # Optional - for fairness
    sensitive_attributes=['income'], # Optional - for privacy
    quasi_identifiers=['age', 'zip'] # Optional - for k-anonymity
)
```

---

## Results Object

The `SDCFResults` object contains:

```python
results.conformance      # Conformance enum (SDCF-A, SDCF-P, SDCF-R)
results.overall_score    # Float 0.0-1.0
results.privacy          # DimensionScore object
results.fidelity         # DimensionScore object
results.fairness         # DimensionScore object
results.tier             # AssessmentTier enum
results.limitations      # List of limitation strings
results.restrictions     # List of restriction strings

# Export to JSON
results.to_dict()
```

### DimensionScore Object

```python
dimension.score      # Float 0.0-1.0
dimension.status     # DimensionStatus enum
dimension.metrics    # Dict of computed metrics
dimension.concerns   # List of concern strings
```

---

## Metrics Reference

### Privacy Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| DCR 90th percentile | Distance to Closest Record | > 0.05 |
| DCR 50th percentile | Median DCR | > 0.01 |
| K-anonymity | Minimum k value | ≥ 3 |
| Disclosure risk | Attribute disclosure % | < 5% |

### Fidelity Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| KS test pass rate | % variables passing KS test | > 85% |
| Correlation preservation | % correlations within ±0.15 | > 80% |
| Mean difference | Summary statistic difference | < 10% |

### Fairness Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| Demographic parity ratio | Group representation ratio | 0.80-1.25 |
| Group preservation | Within-group fidelity | > 75% |

---

## Customizing Thresholds

```python
# Access default thresholds
print(SDCFAssessor.THRESHOLDS)

# Override thresholds
assessor = SDCFAssessor(tier='gold')
assessor.THRESHOLDS['privacy']['dcr_90th_acceptable'] = 0.10  # More strict
```

---

## Example: Full Assessment

```python
import pandas as pd
import numpy as np
from src.sdcf_assessor import SDCFAssessor

# Generate sample data
np.random.seed(42)
n = 1000

source_df = pd.DataFrame({
    'age': np.random.normal(45, 15, n).clip(18, 90),
    'income': np.random.normal(50000, 20000, n).clip(20000, 150000),
    'gender': np.random.choice(['M', 'F'], n),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n)
})

# Synthetic data (with noise)
synthetic_df = source_df.copy()
synthetic_df['age'] += np.random.normal(0, 2, n)
synthetic_df['income'] += np.random.normal(0, 5000, n)

# Run assessment
assessor = SDCFAssessor(tier='gold')
results = assessor.assess(
    source_data=source_df,
    synthetic_data=synthetic_df,
    categorical_columns=['gender', 'region'],
    protected_attributes=['gender'],
    sensitive_attributes=['income'],
    quasi_identifiers=['age', 'region']
)

# Print results
print(f"Conformance: {results.conformance.value}")
print(f"Overall Score: {results.overall_score:.2f}")

# Export to JSON
import json
print(json.dumps(results.to_dict(), indent=2))
```

---

## Tier-Specific Behavior

### Gold Tier
- Full DCR computation
- KS tests on all numeric variables
- Complete fairness analysis

### Silver Tier
- Summary statistics comparison only
- No record-level privacy metrics
- Group-level fairness only

### Bronze Tier
- Intrinsic quality checks only
- Privacy cannot be verified
- Default provisional scores

---

## Error Handling

```python
# Missing synthetic data
try:
    assessor.assess(source_data=df)  # Missing synthetic_data
except ValueError as e:
    print(e)  # "synthetic_data is required"

# Wrong tier for available data
try:
    assessor = SDCFAssessor(tier='gold')
    assessor.assess(synthetic_data=df)  # Missing source_data for gold
except ValueError as e:
    print(e)  # "gold tier requires source_data"
```

---

## Integration with Other Tools

### With SDMetrics

```python
from sdmetrics.reports.single_table import QualityReport

# Use SDMetrics for detailed analysis
report = QualityReport()
report.generate(source_df, synthetic_df, metadata)

# Then use SDCF for conformance determination
assessor = SDCFAssessor(tier='gold')
results = assessor.assess(
    source_data=source_df,
    synthetic_data=synthetic_df
)
```

---

## License

MIT License - see LICENSE file.

---

**SDCF Version:** 1.0  
**Author:** Wayne Kearns, Kaionix Labs

© 2025 Wayne Kearns, Kaionix Labs.

