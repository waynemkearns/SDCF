#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Download Script for Bronze Tier Validation
SDCF v1.95 - Synthetic Data Compliance Framework

Downloads the 10 synthetic datasets used in the Bronze Tier retrospective study.
Some datasets require manual download due to licensing or size constraints.

Author: Wayne Kearns
License: MIT
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path

# Data directory
DATA_DIR = Path(__file__).parent / "data"

# Dataset sources and instructions
DATASETS = {
    "D1": {
        "name": "PLEIAs SYNTH",
        "file": "pleias_synth_sample.csv",
        "source": "https://huggingface.co/datasets/PleIAs/synthetic-data-for-reasoning-CoT-stage-1",
        "method": "manual",
        "instructions": """
        1. Visit: https://huggingface.co/datasets/PleIAs/synthetic-data-for-reasoning-CoT-stage-1
        2. Download the dataset (requires HuggingFace account)
        3. Extract a 10,000 record sample
        4. Save as: validation/data/pleias_synth_sample.csv
        """
    },
    "D2": {
        "name": "SDV Adult - GaussianCopula",
        "file": "sdv_adult_synthetic_gaussiancopula.csv",
        "source": "Generated using SDV library from UCI Adult dataset",
        "method": "generate",
        "instructions": """
        Generate using SDV:
        ```python
        from sdv.single_table import GaussianCopulaSynthesizer
        from sdv.datasets.demo import download_demo
        
        real_data, metadata = download_demo(modality='single_table', dataset_name='adult')
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(real_data)
        synthetic = synthesizer.sample(num_rows=32561)
        synthetic.to_csv('sdv_adult_synthetic_gaussiancopula.csv', index=False)
        ```
        """
    },
    "D3": {
        "name": "SDV Adult - CTGAN",
        "file": "sdv_adult_synthetic_ctgan.csv",
        "source": "Generated using SDV library from UCI Adult dataset",
        "method": "generate",
        "instructions": """
        Generate using SDV:
        ```python
        from sdv.single_table import CTGANSynthesizer
        from sdv.datasets.demo import download_demo
        
        real_data, metadata = download_demo(modality='single_table', dataset_name='adult')
        synthesizer = CTGANSynthesizer(metadata, epochs=300)
        synthesizer.fit(real_data)
        synthetic = synthesizer.sample(num_rows=32561)
        synthetic.to_csv('sdv_adult_synthetic_ctgan.csv', index=False)
        ```
        """
    },
    "D4": {
        "name": "SDV Adult - TVAE",
        "file": "sdv_adult_synthetic_tvae.csv",
        "source": "Generated using SDV library from UCI Adult dataset",
        "method": "generate",
        "instructions": """
        Generate using SDV:
        ```python
        from sdv.single_table import TVAESynthesizer
        from sdv.datasets.demo import download_demo
        
        real_data, metadata = download_demo(modality='single_table', dataset_name='adult')
        synthesizer = TVAESynthesizer(metadata, epochs=300)
        synthesizer.fit(real_data)
        synthetic = synthesizer.sample(num_rows=32561)
        synthetic.to_csv('sdv_adult_synthetic_tvae.csv', index=False)
        ```
        """
    },
    "D5": {
        "name": "Gretel Safety Alignment",
        "file": "gretel_safety_all.csv",
        "source": "https://huggingface.co/datasets/gretelai/synthetic-text-to-sql",
        "method": "manual",
        "instructions": """
        1. Visit: https://huggingface.co/datasets/gretelai/synthetic-gsm8k-reflection-405b
        2. Download the dataset
        3. Save as: validation/data/gretel_safety_all.csv
        """
    },
    "D6": {
        "name": "MostlyAI Census",
        "file": "mostlyai_census.csv",
        "source": "https://github.com/mostly-ai/public-demo-data",
        "method": "download",
        "url": "https://raw.githubusercontent.com/mostly-ai/public-demo-data/main/census-income/census-synthetic.csv",
    },
    "D7": {
        "name": "MostlyAI CDNOW Purchases",
        "file": "mostlyai_cdnow_purchases.csv",
        "source": "https://github.com/mostly-ai/public-demo-data",
        "method": "download",
        "url": "https://raw.githubusercontent.com/mostly-ai/public-demo-data/main/cdnow/cdnow_synthetic.csv",
    },
    "D8": {
        "name": "CMS DE-SynPUF Demo",
        "file": "cms_synpuf_sample_demo.csv",
        "source": "https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf",
        "method": "manual",
        "instructions": """
        1. Visit: https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
        2. Download DE-SynPUF Sample 1 Beneficiary Summary File
        3. Extract 5,000 record sample
        4. Save as: validation/data/cms_synpuf_sample_demo.csv
        """
    },
    "D9": {
        "name": "US Census SynLBD Demo",
        "file": "synlbd_demo.csv",
        "source": "https://www.census.gov/programs-surveys/ces/data/restricted-use-data/synthetic-data.html",
        "method": "manual",
        "instructions": """
        1. Visit: https://www.census.gov/programs-surveys/ces/data/restricted-use-data/synthetic-data.html
        2. Download SynLBD demo data
        3. Save as: validation/data/synlbd_demo.csv
        """
    },
    "D10": {
        "name": "Jupyter Agent Dataset",
        "file": "jupyter_agent_sample.csv",
        "source": "https://huggingface.co/datasets/xingyaoww/code-act",
        "method": "manual",
        "instructions": """
        1. Visit: https://huggingface.co/datasets/xingyaoww/code-act
        2. Download the dataset
        3. Extract 1,000 record sample with QA pairs
        4. Save as: validation/data/jupyter_agent_sample.csv
        """
    },
}


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL."""
    try:
        print(f"  Downloading from {url[:60]}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ Saved to {dest.name}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def check_dataset(dataset_id: str, info: dict) -> bool:
    """Check if dataset file exists."""
    file_path = DATA_DIR / info["file"]
    return file_path.exists()


def download_datasets():
    """Download or provide instructions for all datasets."""
    
    print("\n" + "="*70)
    print("SDCF Bronze Tier Validation - Dataset Download")
    print("="*70)
    
    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    print(f"\nData directory: {DATA_DIR}")
    
    downloaded = 0
    manual_required = []
    
    for dataset_id, info in DATASETS.items():
        print(f"\n[{dataset_id}] {info['name']}")
        
        if check_dataset(dataset_id, info):
            print(f"  ✓ Already exists: {info['file']}")
            downloaded += 1
            continue
        
        if info["method"] == "download" and "url" in info:
            dest = DATA_DIR / info["file"]
            if download_file(info["url"], dest):
                downloaded += 1
            else:
                manual_required.append((dataset_id, info))
        else:
            manual_required.append((dataset_id, info))
            print(f"  ⚠ Manual download required")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Downloaded/Available: {downloaded}/10")
    print(f"  Manual required: {len(manual_required)}/10")
    
    if manual_required:
        print("\n" + "="*70)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("="*70)
        for dataset_id, info in manual_required:
            print(f"\n[{dataset_id}] {info['name']}")
            print(f"  Source: {info['source']}")
            print(f"  Save as: {DATA_DIR / info['file']}")
            if "instructions" in info:
                print(info["instructions"])
    
    return downloaded == 10


def verify_datasets():
    """Verify all datasets are present and loadable."""
    import pandas as pd
    
    print("\n" + "="*70)
    print("DATASET VERIFICATION")
    print("="*70)
    
    all_valid = True
    
    for dataset_id, info in DATASETS.items():
        file_path = DATA_DIR / info["file"]
        print(f"\n[{dataset_id}] {info['name']}")
        
        if not file_path.exists():
            print(f"  ✗ Missing: {info['file']}")
            all_valid = False
            continue
        
        try:
            df = pd.read_csv(file_path)
            print(f"  ✓ Loaded: {len(df):,} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"  ✗ Error loading: {e}")
            all_valid = False
    
    return all_valid


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_datasets()
    else:
        download_datasets()



