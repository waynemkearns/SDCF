# Bronze Tier Validation Study

**SDCF v1.95** - Synthetic Data Compliance Framework

This directory contains the reproducibility package for the Bronze Tier retrospective validation study presented in:

> Kearns, W. (2025). "Empirical Validation of SDCF Bronze Tier Assessment: A Retrospective Study of Ten Synthetic Datasets." *Journal of Privacy and Confidentiality* (submitted).

## Reproducibility

### Quick Verification

To verify the pre-computed results match the paper:

```bash
# Results are in bronze_retrospective_results.json
# Compare with Paper 2 Table 2 values
python -c "import json; print(json.dumps(json.load(open('bronze_retrospective_results.json')), indent=2))"
```

### Full Reproduction

To reproduce the validation from scratch:

#### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 2: Download Datasets

```bash
python download_data.py
```

This will:
- Auto-download datasets available via direct URL (MostlyAI Census, MostlyAI CDNOW)
- Provide instructions for datasets requiring manual download
- Create `data/` directory with all required files

#### Step 3: Run Validation

```bash
python bronze_retrospective.py
```

This generates:
- `results/bronze_retrospective_results.json` - Full results
- `results/bronze_retrospective_summary.csv` - Summary table
- `results/bronze_retrospective_summary.md` - Markdown report

#### Step 4: Verify Results

```bash
python download_data.py --verify
```

## Dataset Portfolio

| ID | Dataset | Domain | Source | Records |
|----|---------|--------|--------|---------|
| D1 | PLEIAs SYNTH | AI Training | HuggingFace | 10,000 |
| D2 | SDV Adult - GaussianCopula | Demographic | SDV Generated | 32,561 |
| D3 | SDV Adult - CTGAN | Demographic | SDV Generated | 32,561 |
| D4 | SDV Adult - TVAE | Demographic | SDV Generated | 32,561 |
| D5 | Gretel Safety Alignment | AI Safety | HuggingFace | 8,361 |
| D6 | MostlyAI Census | Demographic | GitHub | 48,842 |
| D7 | MostlyAI CDNOW | E-commerce | GitHub | 69,659 |
| D8 | CMS DE-SynPUF | Healthcare | CMS.gov | 5,000 |
| D9 | US Census SynLBD | Business | Census.gov | 10,000 |
| D10 | Jupyter Agent | Code/Data | HuggingFace | 377 |

## Pre-Computed Results

The `bronze_retrospective_results.json` file contains the validated results used in the paper:

| ID | Dataset | B-PRS | B-FI | B-FV | Conformance |
|----|---------|-------|------|------|-------------|
| D1 | PLEIAs SYNTH | 0.777 | 0.927 | 0.909 | SDCF-R-Bronze |
| D2 | SDV Adult - GaussianCopula | 0.639 | 0.999 | 0.693 | SDCF-R-Bronze |
| D3 | SDV Adult - CTGAN | 0.637 | 0.998 | 0.548 | SDCF-R-Bronze |
| D4 | SDV Adult - TVAE | 0.534 | 0.994 | 0.724 | SDCF-R-Bronze |
| D5 | Gretel Safety Alignment | 0.789 | 0.999 | 0.100 | SDCF-R-Bronze |
| D6 | MostlyAI Census | 0.631 | 0.997 | 0.692 | SDCF-R-Bronze |
| D7 | MostlyAI CDNOW | 0.360 | 0.999 | 0.100 | SDCF-A-Bronze |
| D8 | CMS DE-SynPUF | 0.390 | 0.997 | 0.100 | SDCF-A-Bronze |
| D9 | US Census SynLBD | 0.360 | 1.000 | 0.100 | SDCF-A-Bronze |
| D10 | Jupyter Agent | 0.090 | 0.999 | 0.128 | SDCF-A-Bronze |

**Summary Statistics (from paper):**
- B-PRS: Mean = 0.521, SD = 0.219
- Classification: 1 Low, 4 Moderate, 5 High Risk

## File Structure

```
validation/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── download_data.py                   # Dataset download script
├── bronze_retrospective.py            # Main validation script
├── bronze_retrospective_results.json  # Pre-computed results
└── data/                              # Downloaded datasets (not in repo)
    ├── pleias_synth_sample.csv
    ├── sdv_adult_synthetic_*.csv
    ├── gretel_safety_all.csv
    ├── mostlyai_*.csv
    ├── cms_synpuf_sample_demo.csv
    ├── synlbd_demo.csv
    └── jupyter_agent_sample.csv
```

## License

- **Code**: MIT License
- **Results/Documentation**: CC BY-NC-SA 4.0

## Citation

```bibtex
@article{kearns2025bronze,
  title={Empirical Validation of SDCF Bronze Tier Assessment: A Retrospective Study of Ten Synthetic Datasets},
  author={Kearns, Wayne},
  journal={Journal of Privacy and Confidentiality},
  year={2025},
  note={Submitted}
}
```

## Contact

For questions about reproducibility:
- GitHub Issues: https://github.com/waynemkearns/SDCF/issues
- Email: wayne.kearns@nortesconsulting.com
