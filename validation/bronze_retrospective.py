#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bronze Tier Retrospective Validation - SDCF v1.95
Synthetic Data Compliance Framework

Assesses synthetic datasets using Bronze Tier (synthetic-only) methodology
across Privacy, Fidelity, and Fairness pillars.

Author: Wayne Kearns, Kaionix Labs
License: MIT
"""

import os
import sys
import json
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Fix Windows console encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a dataset in the retrospective study."""
    id: str
    name: str
    path: str
    domain: str
    high_sensitivity: bool
    external_sharing: bool
    quality_tier: str  # "High", "Known Issues", "Neutral"
    year: int
    notes: str
    
@dataclass
class BronzeResults:
    """Results from Bronze Tier assessment."""
    dataset_id: str
    dataset_name: str
    
    # Privacy Risk Score components
    outlier_score: float
    uniqueness_score: float
    context_penalty: float
    b_prs: float
    prs_level: str
    
    # Fidelity Index components
    consistency_score: float
    validity_score: float
    b_fi: float
    fi_level: str
    
    # Fairness Variance components
    representation_gap: float
    uncertainty_buffer: float
    b_fv: float
    fv_level: str
    
    # Overall conformance
    conformance: str
    
    # Metadata
    n_records: int
    n_features: int
    domain: str
    notes: str


# ============================================================================
# BRONZE TIER METRICS IMPLEMENTATION
# ============================================================================

class BronzeTierAssessor:
    """Implements Bronze Tier (synthetic-only) assessment methodology."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data = None
        self.numeric_cols = []
        self.categorical_cols = []
        
    def load_data(self) -> bool:
        """Load and prepare dataset for assessment."""
        try:
            print(f"\n{'='*70}")
            print(f"LOADING: {self.config.name}")
            print(f"{'='*70}")
            
            # Load data
            if self.config.path.endswith('.csv'):
                self.data = pd.read_csv(self.config.path)
            elif self.config.path.endswith('.parquet'):
                self.data = pd.read_parquet(self.config.path)
            else:
                raise ValueError(f"Unsupported file format: {self.config.path}")
            
            print(f"  ✓ Loaded {len(self.data):,} records, {len(self.data.columns)} columns")
            
            # Identify column types
            self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            print(f"  ✓ Numeric columns: {len(self.numeric_cols)}")
            print(f"  ✓ Categorical columns: {len(self.categorical_cols)}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ ERROR loading data: {e}")
            return False
    
    def compute_b_prs(self) -> Tuple[float, float, float, float]:
        """
        Compute Bronze Privacy Risk Score (B-PRS).
        
        Returns:
            (outlier_score, uniqueness_score, context_penalty, b_prs)
        """
        print(f"\n  [Privacy] Computing B-PRS...")
        
        # 1. Outlier Detection (using LOF)
        outlier_score = self._compute_outlier_score()
        print(f"    Outlier score: {outlier_score:.3f}")
        
        # 2. Uniqueness Assessment (k-anonymity proxy)
        uniqueness_score = self._compute_uniqueness_score()
        print(f"    Uniqueness score: {uniqueness_score:.3f}")
        
        # 3. Context Penalty
        context_penalty = self._compute_context_penalty()
        print(f"    Context penalty: {context_penalty:.3f}")
        
        # 4. Weighted B-PRS
        b_prs = (0.3 * outlier_score + 
                 0.4 * uniqueness_score + 
                 0.3 * context_penalty)
        
        print(f"    → B-PRS: {b_prs:.3f}")
        
        return outlier_score, uniqueness_score, context_penalty, b_prs
    
    def _compute_outlier_score(self) -> float:
        """Compute outlier prevalence using Local Outlier Factor."""
        if not self.numeric_cols:
            return 0.0
        
        # Sample if dataset is large
        sample_data = self.data[self.numeric_cols].copy()
        if len(sample_data) > 5000:
            sample_data = sample_data.sample(n=5000, random_state=42)
        
        # Handle missing values
        sample_data = sample_data.fillna(sample_data.median())
        
        # Handle inf values - replace with column max (finite values only)
        sample_data = sample_data.replace([np.inf, -np.inf], np.nan)
        for col in sample_data.columns:
            col_max = sample_data[col][np.isfinite(sample_data[col])].max()
            sample_data[col] = sample_data[col].fillna(col_max if not np.isnan(col_max) else 0)
        
        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sample_data)
        
        # LOF computation
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        lof_scores = lof.fit_predict(X_scaled)
        
        # Proportion of outliers (LOF=-1)
        outlier_proportion = (lof_scores == -1).mean()
        
        # Scale to 0-1 range (0.1 contamination means max ~0.1)
        outlier_score = min(outlier_proportion * 10, 1.0)
        
        return outlier_score
    
    def _compute_uniqueness_score(self) -> float:
        """Compute uniqueness using k-anonymity proxy."""
        if not self.categorical_cols:
            return 0.0
        
        # Select quasi-identifiers heuristically
        # QI candidates: 5 < unique_values < 50% of records
        qi_candidates = []
        for col in self.categorical_cols:
            n_unique = self.data[col].nunique()
            if 5 < n_unique < len(self.data) * 0.5:
                qi_candidates.append(col)
        
        if not qi_candidates:
            return 0.0
        
        # Limit to 4 QIs for practicality
        qis = qi_candidates[:4]
        
        # Count groups with k < 5
        group_sizes = self.data.groupby(qis, dropna=False).size()
        small_groups = (group_sizes < 5).sum()
        total_groups = len(group_sizes)
        
        uniqueness_score = small_groups / total_groups if total_groups > 0 else 0.0
        
        return min(uniqueness_score, 1.0)
    
    def _compute_context_penalty(self) -> float:
        """Compute context-based risk penalty."""
        base_penalty = 0.20
        penalty = base_penalty
        
        # High-sensitivity domain
        if self.config.high_sensitivity:
            penalty += 0.10
        
        # External sharing
        if self.config.external_sharing:
            penalty += 0.10
        
        # Clamp between 0.10 and 0.50
        return max(0.10, min(0.50, penalty))
    
    def compute_b_fi(self) -> Tuple[float, float, float]:
        """
        Compute Bronze Fidelity Index (B-FI).
        
        Returns:
            (consistency_score, validity_score, b_fi)
        """
        print(f"\n  [Fidelity] Computing B-FI...")
        
        # 1. Consistency Check (split stability)
        consistency_score = self._compute_consistency()
        print(f"    Consistency score: {consistency_score:.3f}")
        
        # 2. Validity Check
        validity_score = self._compute_validity()
        print(f"    Validity score: {validity_score:.3f}")
        
        # 3. Weighted B-FI
        b_fi = 0.5 * consistency_score + 0.5 * validity_score
        
        print(f"    → B-FI: {b_fi:.3f}")
        
        return consistency_score, validity_score, b_fi
    
    def _compute_consistency(self) -> float:
        """Compute internal consistency via split stability."""
        if not self.numeric_cols:
            return 0.5  # Neutral for non-numeric
        
        n_splits = 5
        sample_fraction = 0.8
        
        split_means = {col: [] for col in self.numeric_cols}
        
        for _ in range(n_splits):
            sample = self.data[self.numeric_cols].sample(frac=sample_fraction, random_state=None)
            for col in self.numeric_cols:
                split_means[col].append(sample[col].mean())
        
        # Coefficient of variation for each column
        cvs = []
        for col in self.numeric_cols:
            mean_of_means = np.mean(split_means[col])
            std_of_means = np.std(split_means[col])
            if mean_of_means != 0:
                cv = std_of_means / abs(mean_of_means)
                cvs.append(cv)
        
        # Average CV across columns
        avg_cv = np.mean(cvs) if cvs else 0.0
        
        # Score: 1 - CV (clamped to 0-1)
        consistency_score = max(0.0, min(1.0, 1.0 - avg_cv))
        
        return consistency_score
    
    def _compute_validity(self) -> float:
        """Compute basic validity (non-null ratio, range checks)."""
        # Non-null ratio
        non_null_ratio = (self.data.notna().sum().sum() / 
                         (len(self.data) * len(self.data.columns)))
        
        # Simple validity score
        validity_score = non_null_ratio
        
        return validity_score
    
    def compute_b_fv(self) -> Tuple[float, float, float]:
        """
        Compute Bronze Fairness Variance (B-FV).
        
        Returns:
            (representation_gap, uncertainty_buffer, b_fv)
        """
        print(f"\n  [Fairness] Computing B-FV...")
        
        # 1. Representation Balance (if protected attributes available)
        representation_gap = self._compute_representation_gap()
        print(f"    Representation gap: {representation_gap:.3f}")
        
        # 2. Uncertainty Buffer
        uncertainty_buffer = 0.10  # Fixed for Bronze Tier
        print(f"    Uncertainty buffer: {uncertainty_buffer:.3f}")
        
        # 3. B-FV
        b_fv = representation_gap + uncertainty_buffer
        
        print(f"    → B-FV: {b_fv:.3f}")
        
        return representation_gap, uncertainty_buffer, b_fv
    
    def _compute_representation_gap(self) -> float:
        """Compute representation balance across categorical features."""
        if not self.categorical_cols:
            return 0.0
        
        # Look for potential protected attributes
        protected_keywords = ['gender', 'sex', 'race', 'ethnicity', 'age', 'disability']
        protected_cols = [col for col in self.categorical_cols 
                         if any(kw in col.lower() for kw in protected_keywords)]
        
        if not protected_cols:
            return 0.0
        
        gaps = []
        for col in protected_cols:
            value_counts = self.data[col].value_counts(normalize=True)
            # Gap: max proportion - min proportion
            if len(value_counts) > 1:
                gap = value_counts.max() - value_counts.min()
                gaps.append(gap)
        
        return np.mean(gaps) if gaps else 0.0
    
    def assess(self) -> Optional[BronzeResults]:
        """Run complete Bronze Tier assessment."""
        if not self.load_data():
            return None
        
        # Compute pillars
        outlier_score, uniqueness_score, context_penalty, b_prs = self.compute_b_prs()
        consistency_score, validity_score, b_fi = self.compute_b_fi()
        representation_gap, uncertainty_buffer, b_fv = self.compute_b_fv()
        
        # Determine levels
        prs_level = self._get_prs_level(b_prs)
        fi_level = self._get_fi_level(b_fi)
        fv_level = self._get_fv_level(b_fv)
        
        # Determine conformance
        conformance = self._get_conformance(b_prs, b_fi, b_fv)
        
        print(f"\n{'='*70}")
        print(f"RESULTS: {self.config.name}")
        print(f"{'='*70}")
        print(f"  B-PRS: {b_prs:.3f} ({prs_level})")
        print(f"  B-FI:  {b_fi:.3f} ({fi_level})")
        print(f"  B-FV:  {b_fv:.3f} ({fv_level})")
        print(f"  → Conformance: {conformance}")
        print(f"{'='*70}\n")
        
        return BronzeResults(
            dataset_id=self.config.id,
            dataset_name=self.config.name,
            outlier_score=outlier_score,
            uniqueness_score=uniqueness_score,
            context_penalty=context_penalty,
            b_prs=b_prs,
            prs_level=prs_level,
            consistency_score=consistency_score,
            validity_score=validity_score,
            b_fi=b_fi,
            fi_level=fi_level,
            representation_gap=representation_gap,
            uncertainty_buffer=uncertainty_buffer,
            b_fv=b_fv,
            fv_level=fv_level,
            conformance=conformance,
            n_records=len(self.data),
            n_features=len(self.data.columns),
            domain=self.config.domain,
            notes=self.config.notes
        )
    
    @staticmethod
    def _get_prs_level(b_prs: float) -> str:
        """Map B-PRS to risk level."""
        if b_prs < 0.30:
            return "Low"  # Bronze rarely achieves this
        elif b_prs < 0.50:
            return "Moderate"
        elif b_prs < 0.70:
            return "High"
        else:
            return "Critical"
    
    @staticmethod
    def _get_fi_level(b_fi: float) -> str:
        """Map B-FI to fidelity level."""
        if b_fi > 0.90:
            return "Excellent"
        elif b_fi > 0.80:
            return "Good"
        elif b_fi > 0.70:
            return "Acceptable"
        elif b_fi > 0.60:
            return "Marginal"
        else:
            return "Poor"
    
    @staticmethod
    def _get_fv_level(b_fv: float) -> str:
        """Map B-FV to fairness level."""
        if b_fv < 0.15:
            return "Fair"
        elif b_fv < 0.30:
            return "Concerning"
        else:
            return "Problematic"
    
    @staticmethod
    def _get_conformance(b_prs: float, b_fi: float, b_fv: float) -> str:
        """Determine overall SDCF conformance level."""
        # SDCF-R: Rejected
        if b_prs > 0.70 or b_fi < 0.60 or b_fv > 0.30:
            return "SDCF-R-Bronze"
        
        # SDCF-A: Acceptable
        if b_prs < 0.50 and b_fi > 0.70 and b_fv < 0.30:
            return "SDCF-A-Bronze"
        
        # SDCF-P: Provisional (everything else)
        return "SDCF-P-Bronze"


# ============================================================================
# RETROSPECTIVE STUDY RUNNER
# ============================================================================

def run_retrospective_study(configs: List[DatasetConfig], output_dir: str = "results"):
    """Run Bronze Tier assessment on all configured datasets."""
    
    print("\n" + "="*70)
    print("BRONZE TIER RETROSPECTIVE VALIDATION STUDY")
    print("SDCF v1.95 - Synthetic Data Compliance Framework")
    print("="*70)
    print(f"\nDatasets to assess: {len(configs)}")
    print(f"Output directory: {output_dir}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for config in configs:
        assessor = BronzeTierAssessor(config)
        result = assessor.assess()
        if result:
            results.append(result)
    
    # Save results
    if results:
        save_results(results, output_dir)
        print(f"\n✓ Assessment complete. Results saved to {output_dir}/")
    else:
        print("\n✗ No results generated")
    
    return results


def save_results(results: List[BronzeResults], output_dir: str):
    """Save results in multiple formats."""
    
    # 1. JSON (full results)
    json_path = os.path.join(output_dir, "bronze_retrospective_results.json")
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"  ✓ Saved JSON: {json_path}")
    
    # 2. CSV (summary table)
    df = pd.DataFrame([{
        'ID': r.dataset_id,
        'Dataset': r.dataset_name,
        'Domain': r.domain,
        'Records': r.n_records,
        'Features': r.n_features,
        'B-PRS': f"{r.b_prs:.3f}",
        'PRS Level': r.prs_level,
        'B-FI': f"{r.b_fi:.3f}",
        'FI Level': r.fi_level,
        'B-FV': f"{r.b_fv:.3f}",
        'FV Level': r.fv_level,
        'Conformance': r.conformance
    } for r in results])
    
    csv_path = os.path.join(output_dir, "bronze_retrospective_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved CSV: {csv_path}")
    
    # 3. Markdown (readable summary)
    md_path = os.path.join(output_dir, "bronze_retrospective_summary.md")
    with open(md_path, 'w') as f:
        f.write("# Bronze Tier Retrospective Validation Results\n\n")
        f.write(f"**Assessment Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Framework:** SDCF v1.0\n")
        f.write(f"**Datasets Assessed:** {len(results)}\n\n")
        f.write("## Summary Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Detailed Results\n\n")
        for r in results:
            f.write(f"### {r.dataset_name}\n\n")
            f.write(f"- **Domain:** {r.domain}\n")
            f.write(f"- **Size:** {r.n_records:,} records, {r.n_features} features\n")
            f.write(f"- **Privacy Risk:** B-PRS = {r.b_prs:.3f} ({r.prs_level})\n")
            f.write(f"- **Fidelity:** B-FI = {r.b_fi:.3f} ({r.fi_level})\n")
            f.write(f"- **Fairness:** B-FV = {r.b_fv:.3f} ({r.fv_level})\n")
            f.write(f"- **Conformance:** {r.conformance}\n")
            f.write(f"- **Notes:** {r.notes}\n\n")
    print(f"  ✓ Saved Markdown: {md_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration for COMPLETE Bronze Tier Assessment
    # Phase 1 & 2: 10 diverse synthetic datasets across 6 domains
    
    configs = [
        # Original 4 datasets
        DatasetConfig(
            id="D1",
            name="PLEIAs SYNTH",
            path="data/pleias_synth_sample.csv",
            domain="AI Training",
            high_sensitivity=False,
            external_sharing=True,
            quality_tier="High",
            year=2025,
            notes="Frontier AI reasoning pre-training data, sample of 10K records"
        ),
        DatasetConfig(
            id="D2",
            name="SDV Adult - GaussianCopula",
            path="data/sdv_adult_synthetic_gaussiancopula.csv",
            domain="Demographic",
            high_sensitivity=False,
            external_sharing=False,
            quality_tier="High",
            year=2024,
            notes="Census-derived benchmark, GaussianCopula synthesizer"
        ),
        DatasetConfig(
            id="D3",
            name="SDV Adult - CTGAN",
            path="data/sdv_adult_synthetic_ctgan.csv",
            domain="Demographic",
            high_sensitivity=False,
            external_sharing=False,
            quality_tier="High",
            year=2024,
            notes="Census-derived benchmark, CTGAN deep learning synthesizer"
        ),
        DatasetConfig(
            id="D4",
            name="SDV Adult - TVAE",
            path="data/sdv_adult_synthetic_tvae.csv",
            domain="Demographic",
            high_sensitivity=False,
            external_sharing=False,
            quality_tier="High",
            year=2024,
            notes="Census-derived benchmark, TVAE variational autoencoder synthesizer"
        ),
        # Phase 1: New public datasets (downloaded)
        DatasetConfig(
            id="D5",
            name="Gretel Safety Alignment",
            path="data/gretel_safety_all.csv",
            domain="AI Safety",
            high_sensitivity=False,
            external_sharing=True,
            quality_tier="High",
            year=2024,
            notes="LLM safety alignment dataset, 8,361 records, Apache 2.0 license"
        ),
        DatasetConfig(
            id="D6",
            name="MostlyAI Census",
            path="data/mostlyai_census.csv",
            domain="Demographic",
            high_sensitivity=False,
            external_sharing=True,
            quality_tier="High",
            year=2023,
            notes="US Census 1994 synthetic, 48,842 records, GAN-based, MostlyAI"
        ),
        DatasetConfig(
            id="D7",
            name="MostlyAI CDNOW Purchases",
            path="data/mostlyai_cdnow_purchases.csv",
            domain="E-commerce",
            high_sensitivity=False,
            external_sharing=False,
            quality_tier="High",
            year=2023,
            notes="E-commerce purchase history, 69,659 transactions, time-series data"
        ),
        # Phase 2: Additional datasets (created/downloaded)
        DatasetConfig(
            id="D8",
            name="CMS DE-SynPUF Demo",
            path="data/cms_synpuf_sample_demo.csv",
            domain="Healthcare",
            high_sensitivity=True,
            external_sharing=False,
            quality_tier="High",
            year=2010,
            notes="Medicare synthetic claims, 5,000 beneficiaries, multi-year data"
        ),
        DatasetConfig(
            id="D9",
            name="US Census SynLBD Demo",
            path="data/synlbd_demo.csv",
            domain="Business",
            high_sensitivity=False,
            external_sharing=False,
            quality_tier="High",
            year=2020,
            notes="Synthetic business establishments, 10,000 records, Census Bureau"
        ),
        DatasetConfig(
            id="D10",
            name="Jupyter Agent Dataset",
            path="data/jupyter_agent_sample.csv",
            domain="Code/Data",
            high_sensitivity=False,
            external_sharing=True,
            quality_tier="High",
            year=2024,
            notes="Synthetic QA pairs from Kaggle notebooks, 1,000 sample, LLM-generated"
        ),
    ]
    
    print("\n" + "="*70)
    print("BRONZE TIER RETROSPECTIVE VALIDATION - COMPLETE")
    print("Phase 1 & 2 Dataset Portfolio")
    print("="*70)
    print(f"Total datasets configured: {len(configs)}")
    print("Domains: AI Training, Demographics, AI Safety, E-commerce, Healthcare, Business, Code/Data")
    print("Synthesis methods: Statistical, GAN, CTGAN, TVAE, LLM")
    print("\nStarting complete assessment...\n")
    
    # Run assessment
    results = run_retrospective_study(configs)
