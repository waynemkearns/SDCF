# Synthetic Data Compliance Framework (SDCF)

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.95-green.svg)](https://github.com/waynemkearns/SDCF)

**A Purpose-Bounded Methodology for Assessing Privacy, Fidelity, and Fairness in Synthetic Data**

> *"Is this synthetic data fit for my specific purpose?"*

---

## Overview

SDCF is a practical framework for evaluating whether synthetic data is acceptable for a **specific intended use**. Unlike generic quality metrics, SDCF provides **purpose-bounded assessments** that evaluate fitness across three dimensions:

| Dimension | What it Measures |
|-----------|------------------|
| **Privacy** | Re-identification risk, attribute disclosure, distance to closest record |
| **Fidelity** | Statistical similarity, correlation preservation, distribution matching |
| **Fairness** | Demographic parity, group representation, bias preservation |

### Key Features

- ✅ **Purpose-Bounded** — Assesses fitness for YOUR specific use case
- ✅ **Three Tiers** — Gold (full source data), Silver (partial), Bronze (no source data)
- ✅ **Regulatory-Mapped** — GDPR, EU AI Act, ISO/IEC 27001, NIST AI RMF
- ✅ **Tool-Agnostic** — Works with SDMetrics, mostlyai-qa, or custom tools
- ✅ **Transparent** — Honest about confidence levels and limitations

### Conformance Levels

| Level | Meaning |
|-------|---------|
| **SDCF-A** | Acceptable — meets all thresholds for intended purpose |
| **SDCF-P** | Provisional — borderline results, use with documented restrictions |
| **SDCF-R** | Rejected — not fit for intended purpose |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/waynemkearns/SDCF.git
cd sdcf

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.sdcf_assessor import SDCFAssessor
import pandas as pd

# Load your data
source_df = pd.read_csv("source_data.csv")
synthetic_df = pd.read_csv("synthetic_data.csv")

# Create assessor (gold tier = full source data access)
assessor = SDCFAssessor(tier='gold')

# Run assessment
results = assessor.assess(
    source_data=source_df,
    synthetic_data=synthetic_df,
    categorical_columns=['gender', 'region'],
    protected_attributes=['gender', 'age_group'],
    sensitive_attributes=['income']
)

# Check conformance
print(f"Conformance: {results.conformance.value}")  # SDCF-A, SDCF-P, or SDCF-R
print(f"Privacy Score: {results.privacy.score:.2f}")
print(f"Fidelity Score: {results.fidelity.score:.2f}")
print(f"Fairness Score: {results.fairness.score:.2f}")
```

---

## Repository Structure

```
sdcf/
├── README.md                          # This file
├── LICENSE                            # MIT License (code)
├── LICENSE-Framework.txt              # CC BY-NC-SA 4.0 (documentation)
├── requirements.txt                   # Python dependencies
│
├── paper/                             # Academic paper
│   ├── sdcf_framework.pdf             # Main paper (PDF)
│   ├── sdcf_framework.tex             # LaTeX source
│   └── references.bib                 # Bibliography
│
├── src/                               # Reference implementation
│   ├── __init__.py
│   ├── sdcf_reference_implementation.py   # Core metrics
│   └── sdcf_assessor.py               # Main assessor class
│
├── templates/                         # Assessment templates
│   ├── Template-Assessment-Checklist.md
│   ├── Template-Purpose-Definition.md
│   ├── Template-SDCF-A-Certificate.md
│   ├── Template-SDCF-P-Certificate.md
│   └── Template-SDCF-R-Statement.md
│
├── docs/                              # Documentation
│   ├── QUICK-START-GUIDE.md
│   └── IMPLEMENTATION-GUIDE.md
│
└── validation/                        # Bronze tier validation study
    ├── README.md
    ├── bronze_retrospective.py
    ├── requirements.txt
    └── results/
        ├── bronze_retrospective_results.json
        └── bronze_retrospective_summary.md
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [**Main Paper**](paper/sdcf_framework.pdf) | Complete 40+ page methodology with mathematical definitions |
| [**Quick Start Guide**](docs/QUICK-START-GUIDE.md) | Get started in 30 minutes |
| [**Implementation Guide**](docs/IMPLEMENTATION-GUIDE.md) | Technical reference for the Python implementation |
| [**Assessment Checklist**](templates/Template-Assessment-Checklist.md) | Comprehensive checklist for assessments |

---

## Assessment Tiers

### Gold Tier (Highest Confidence)
Full access to source data enables:
- Record-level DCR (Distance to Closest Record) analysis
- Comprehensive KS tests and correlation preservation
- Detailed fairness metrics across protected groups

### Silver Tier (Medium Confidence)
Partial source data (summaries, samples) allows:
- Aggregate-level privacy assessment
- Summary statistic comparison
- Group-level fairness evaluation

### Bronze Tier (Limited Confidence)
No source data access restricts assessment to:
- Intrinsic quality checks (completeness, validity, consistency)
- Expert face validity review
- **Privacy cannot be verified**

---

## Regulatory Mapping

SDCF connects directly to major regulatory frameworks:

| Regulation | SDCF Mapping |
|------------|--------------|
| **GDPR Article 32** | Privacy dimension, technical measures documentation |
| **GDPR Article 25** | Purpose-bounded approach, data protection by design |
| **EU AI Act Article 10** | Data governance, quality assessment |
| **ISO/IEC 27001** | Information security management alignment |
| **NIST AI RMF** | Risk management, trustworthiness characteristics |

---

## Citation

If you use SDCF in research or practice, please cite:

```bibtex
@techreport{kearns2025sdcf,
  author = {Kearns, Wayne},
  title = {Synthetic Data Compliance Framework (SDCF): A Purpose-Bounded 
           Methodology for Assessing Privacy, Fidelity, and Fairness 
           in Synthetic Data},
  institution = {Kaionix Labs},
  year = {2025},
  month = {December},
  url = {https://github.com/waynemkearns/SDCF},
  version = {1.95}
}
```

**APA:**
> Kearns, W. (2025). *Synthetic Data Compliance Framework (SDCF): A purpose-bounded methodology for assessing privacy, fidelity, and fairness in synthetic data* (Version 1.95). Kaionix Labs.

---

## License

This project uses a dual-license structure:

| Component | License |
|-----------|---------|
| **Framework Documentation** | [CC BY-NC-SA 4.0](LICENSE-Framework.txt) |
| **Reference Implementation (Code)** | [MIT License](LICENSE) |

**You are free to:**
- Use the code in commercial and non-commercial projects (MIT)
- Share and adapt the documentation for non-commercial purposes (CC BY-NC-SA)
- Build upon the framework with proper attribution

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ways to contribute:**
- 📊 **Empirical Validation** — Run SDCF on your datasets, share results
- 🔧 **Tool Integrations** — Build connectors for additional metric libraries
- 📚 **Case Studies** — Document real-world SDCF applications
- 🐛 **Bug Reports** — Found an issue? Open a GitHub issue

---

## Support

- **Email:** wayne.kearns@nortesconsulting.com
- **Website:** https://github.com/waynemkearns/SDCF
- **Issues:** [GitHub Issues](https://github.com/waynemkearns/SDCF/issues)

### Professional Services

Kaionix Labs offers:
- SDCF training workshops (1-2 days)
- Assessment implementation consulting
- Custom metric development
- Regulatory mapping for specific jurisdictions

---

## Acknowledgments

SDCF builds on research and best practices from:
- EU Agency for Cybersecurity (ENISA)
- National Institute of Standards and Technology (NIST)
- International Organization for Standardization (ISO)
- Synthetic data research community (SDMetrics, SDV, Gretel, MostlyAI)

---

**Version:** 1.95 | **Author:** Wayne Kearns | **Organization:** Kaionix Labs

© 2025 Wayne Kearns, Kaionix Labs. All rights reserved.

