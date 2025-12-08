# Synthetic Data Compliance Framework (SDCF)
## Quick Start Guide

**Version:** 1.95  
**Date:** December 2025  
**Author:** Wayne Kearns, Kaionix Labs  
**Contact:** wayne.kearns@nortesconsulting.com

---

## What is SDCF?

SDCF is a **practical framework for assessing whether synthetic data is fit for a specific purpose** from privacy, fidelity, and fairness perspectives.

**Key Features:**
- ✅ **Purpose-bounded** - Assesses fitness for YOUR specific use case
- ✅ **Three tiers** - Works with full source data (Gold), partial (Silver), or none (Bronze)
- ✅ **Regulatory-mapped** - Connects to GDPR, EU AI Act, ISO, NIST requirements
- ✅ **Tool-agnostic** - Works with SDMetrics, mostlyai-qa, or any metric tool

**What you get:** A conformance determination (SDCF-A/P/R) with quantitative scores, documented limitations, and usage restrictions.

---

## Quick Start: 3 Steps to Your First Assessment

### Step 1: Choose Your Tier (2 minutes)

**Do you have access to the original source data?**

| Your Situation | Tier | Confidence Level |
|----------------|------|------------------|
| ✅ Full source dataset available | **Gold** | High confidence |
| ⚠️ Partial source data (sample/aggregate) | **Silver** | Medium confidence |
| ❌ No source data access | **Bronze** | Limited confidence |

---

### Step 2: Define Your Purpose (10 minutes)

SDCF requires you to specify **exactly what you want to use the synthetic data for**.

**Example purposes:**
- "Train ML model to predict customer churn (logistic regression)"
- "Generate test data for software QA (schema validation only)"
- "Create anonymized dataset for external research collaboration"

---

### Step 3: Run Assessment (30-90 minutes)

#### Install dependencies

```bash
pip install -r requirements.txt
```

#### Run assessment

```python
from src.sdcf_assessor import SDCFAssessor
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
    protected_attributes=['gender'],
    sensitive_attributes=['income']
)

# Check results
print(f"Conformance: {results.conformance.value}")
print(f"Privacy Score: {results.privacy.score:.2f}")
print(f"Fidelity Score: {results.fidelity.score:.2f}")
print(f"Fairness Score: {results.fairness.score:.2f}")
```

---

## Interpreting Results

### SDCF-A (Acceptable)
**Meaning:** Synthetic data meets all thresholds for intended purpose.
**Can you use it?** Yes, for the specified purpose with documented restrictions.

### SDCF-P (Provisional)
**Meaning:** Borderline or has gaps in one dimension.
**Can you use it?** Possibly, depending on risk tolerance and restrictions.

### SDCF-R (Rejected)
**Meaning:** Fails critical thresholds.
**Can you use it?** No, not for the assessed purpose.

---

## Common Pitfalls

### ❌ "Generic Quality" Mindset
**Wrong:** "Is this synthetic data good?"  
**Right:** "Is this synthetic data fit for training our customer churn model?"

### ❌ Ignoring Purpose Boundaries
**Wrong:** Assessing for "general analytics," then using for high-stakes decisions  
**Right:** Assess for specific purpose, restrict usage to that purpose

### ❌ Over-Trusting Bronze Tier
**Wrong:** "SDCF-A certificate means it's safe"  
**Right:** "SDCF-A Bronze means intrinsic quality is acceptable, but privacy is unverified"

---

## Next Steps

1. **Read the main paper** for complete methodology
2. **Use templates** in `/templates` for documentation
3. **Review worked examples** in the paper appendices
4. **Contact us** with questions: wayne.kearns@nortesconsulting.com

---

**SDCF Version:** 1.95  
**License:** CC BY-NC-SA 4.0 (documentation), MIT (code)

© 2025 Wayne Kearns, Kaionix Labs.

