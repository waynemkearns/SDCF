# SDCF Bronze Tier Validation

This directory contains code and data for reproducing the Bronze Tier retrospective validation study presented in the SDCF paper.

## Contents

### Core Scripts

- `bronze_retrospective.py` - Main validation script that assesses synthetic datasets using Bronze Tier metrics (B-PRS, B-FI, B-FV)

### Results

- `results/bronze_retrospective_results.json` - Complete validation results
- `results/bronze_retrospective_summary.md` - Human-readable summary

## Quick Start

### Install Dependencies

```bash
pip install pandas numpy scipy scikit-learn
```

### Run Assessment

```bash
python bronze_retrospective.py
```

## Understanding the Output

### Metrics

- **B-PRS (Bronze Privacy Risk Score)**: Range 0.00-1.00, lower is better
  - Excellent: <0.30
  - Acceptable: 0.30-0.50
  - Concerning: >0.50

- **B-FI (Bronze Fidelity Index)**: Range 0.00-1.00, higher is better
  - Excellent: >0.90
  - Acceptable: 0.75-0.90
  - Poor: <0.75

- **B-FV (Bronze Fairness Vulnerability)**: Range 0.00-1.00
  - High values indicate presence of demographic/sensitive attributes

### Conformance Levels

- **SDCF-A-Bronze**: Acceptable for general-purpose use
- **SDCF-P-Bronze**: Provisional - requires additional review
- **SDCF-R-Bronze**: Restricted - limited use cases only

## Citation

```bibtex
@techreport{kearns2025sdcf,
  author = {Kearns, Wayne},
  title = {Synthetic Data Compliance Framework (SDCF)},
  institution = {Kaionix Labs},
  year = {2025}
}
```

## License

Code: MIT License
Framework: CC BY-NC-SA 4.0

## Contact

Wayne Kearns
wayne.kearns@nortesconsulting.com

