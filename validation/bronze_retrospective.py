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
    quality_tier: str
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
            
            if self.config.path.endswith('.csv'):
                self.data = pd.read_csv(self.config.path)
            elif self.config.path.endswith('.parquet'):
                self.data = pd.read_parquet(self.config.path)
            else:
                raise ValueError(f"Unsupported file format: {self.config.path}")
            
            print(f"  Loaded {len(self.data):,} records, {len(self.data.columns)} columns")
            
            self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            print(f"  Numeric columns: {len(self.numeric_cols)}")
            print(f"  Categorical columns: {len(self.categorical_cols)}")
            
            return True
            
        except Exception as e:
            print(f"  ERROR loading data: {e}")
            return False
    
    def compute_b_prs(self) -> Tuple[float, float, float, float]:
        """Compute Bronze Privacy Risk Score (B-PRS)."""
        print(f"\n  [Privacy] Computing B-PRS...")
        
        outlier_score = self._compute_outlier_score()
        print(f"    Outlier score: {outlier_score:.3f}")
        
        uniqueness_score = self._compute_uniqueness_score()
        print(f"    Uniqueness score: {uniqueness_score:.3f}")
        
        context_penalty = self._compute_context_penalty()
        print(f"    Context penalty: {context_penalty:.3f}")
        
        b_prs = (0.3 * outlier_score + 
                 0.4 * uniqueness_score + 
                 0.3 * context_penalty)
        
        print(f"    B-PRS: {b_prs:.3f}")
        
        return outlier_score, uniqueness_score, context_penalty, b_prs
    
    def _compute_outlier_score(self) -> float:
        """Compute outlier prevalence using Local Outlier Factor."""
        if not self.numeric_cols:
            return 0.0
        
        sample_data = self.data[self.numeric_cols].copy()
        if len(sample_data) > 5000:
            sample_data = sample_data.sample(n=5000, random_state=42)
        
        sample_data = sample_data.fillna(sample_data.median())
        sample_data = sample_data.replace([np.inf, -np.inf], np.nan)
        for col in sample_data.columns:
            col_max = sample_data[col][np.isfinite(sample_data[col])].max()
            sample_data[col] = sample_data[col].fillna(col_max if not np.isnan(col_max) else 0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(sample_data)
        
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        lof_scores = lof.fit_predict(X_scaled)
        
        outlier_proportion = (lof_scores == -1).mean()
        outlier_score = min(outlier_proportion * 10, 1.0)
        
        return outlier_score
    
    def _compute_uniqueness_score(self) -> float:
        """Compute uniqueness using k-anonymity proxy."""
        if not self.categorical_cols:
            return 0.0
        
        qi_candidates = []
        for col in self.categorical_cols:
            n_unique = self.data[col].nunique()
            if 5 < n_unique < len(self.data) * 0.5:
                qi_candidates.append(col)
        
        if not qi_candidates:
            return 0.0
        
        qis = qi_candidates[:4]
        group_sizes = self.data.groupby(qis, dropna=False).size()
        small_groups = (group_sizes < 5).sum()
        total_groups = len(group_sizes)
        
        uniqueness_score = small_groups / total_groups if total_groups > 0 else 0.0
        
        return min(uniqueness_score, 1.0)
    
    def _compute_context_penalty(self) -> float:
        """Compute context-based risk penalty."""
        base_penalty = 0.20
        penalty = base_penalty
        
        if self.config.high_sensitivity:
            penalty += 0.10
        
        if self.config.external_sharing:
            penalty += 0.10
        
        return max(0.10, min(0.50, penalty))
    
    def compute_b_fi(self) -> Tuple[float, float, float]:
        """Compute Bronze Fidelity Index (B-FI)."""
        print(f"\n  [Fidelity] Computing B-FI...")
        
        consistency_score = self._compute_consistency()
        print(f"    Consistency score: {consistency_score:.3f}")
        
        validity_score = self._compute_validity()
        print(f"    Validity score: {validity_score:.3f}")
        
        b_fi = 0.5 * consistency_score + 0.5 * validity_score
        
        print(f"    B-FI: {b_fi:.3f}")
        
        return consistency_score, validity_score, b_fi
    
    def _compute_consistency(self) -> float:
        """Compute internal consistency via split stability."""
        if not self.numeric_cols:
            return 0.5
        
        n_splits = 5
        sample_fraction = 0.8
        
        split_means = {col: [] for col in self.numeric_cols}
        
        for _ in range(n_splits):
            sample = self.data[self.numeric_cols].sample(frac=sample_fraction, random_state=None)
            for col in self.numeric_cols:
                split_means[col].append(sample[col].mean())
        
        cvs = []
        for col in self.numeric_cols:
            mean_of_means = np.mean(split_means[col])
            std_of_means = np.std(split_means[col])
            if mean_of_means != 0:
                cv = std_of_means / abs(mean_of_means)
                cvs.append(cv)
        
        avg_cv = np.mean(cvs) if cvs else 0.0
        consistency_score = max(0.0, min(1.0, 1.0 - avg_cv))
        
        return consistency_score
    
    def _compute_validity(self) -> float:
        """Compute basic validity (non-null ratio)."""
        non_null_ratio = (self.data.notna().sum().sum() / 
                         (len(self.data) * len(self.data.columns)))
        return non_null_ratio
    
    def compute_b_fv(self) -> Tuple[float, float, float]:
        """Compute Bronze Fairness Variance (B-FV)."""
        print(f"\n  [Fairness] Computing B-FV...")
        
        representation_gap = self._compute_representation_gap()
        print(f"    Representation gap: {representation_gap:.3f}")
        
        uncertainty_buffer = 0.10
        print(f"    Uncertainty buffer: {uncertainty_buffer:.3f}")
        
        b_fv = representation_gap + uncertainty_buffer
        
        print(f"    B-FV: {b_fv:.3f}")
        
        return representation_gap, uncertainty_buffer, b_fv
    
    def _compute_representation_gap(self) -> float:
        """Compute representation balance across categorical features."""
        if not self.categorical_cols:
            return 0.0
        
        protected_keywords = ['gender', 'sex', 'race', 'ethnicity', 'age', 'disability']
        protected_cols = [col for col in self.categorical_cols 
                         if any(kw in col.lower() for kw in protected_keywords)]
        
        if not protected_cols:
            return 0.0
        
        gaps = []
        for col in protected_cols:
            value_counts = self.data[col].value_counts(normalize=True)
            if len(value_counts) > 1:
                gap = value_counts.max() - value_counts.min()
                gaps.append(gap)
        
        return np.mean(gaps) if gaps else 0.0
    
    def assess(self) -> Optional[BronzeResults]:
        """Run complete Bronze Tier assessment."""
        if not self.load_data():
            return None
        
        outlier_score, uniqueness_score, context_penalty, b_prs = self.compute_b_prs()
        consistency_score, validity_score, b_fi = self.compute_b_fi()
        representation_gap, uncertainty_buffer, b_fv = self.compute_b_fv()
        
        prs_level = self._get_prs_level(b_prs)
        fi_level = self._get_fi_level(b_fi)
        fv_level = self._get_fv_level(b_fv)
        
        conformance = self._get_conformance(b_prs, b_fi, b_fv)
        
        print(f"\n{'='*70}")
        print(f"RESULTS: {self.config.name}")
        print(f"{'='*70}")
        print(f"  B-PRS: {b_prs:.3f} ({prs_level})")
        print(f"  B-FI:  {b_fi:.3f} ({fi_level})")
        print(f"  B-FV:  {b_fv:.3f} ({fv_level})")
        print(f"  Conformance: {conformance}")
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
        if b_prs < 0.30:
            return "Low"
        elif b_prs < 0.50:
            return "Moderate"
        elif b_prs < 0.70:
            return "High"
        else:
            return "Critical"
    
    @staticmethod
    def _get_fi_level(b_fi: float) -> str:
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
        if b_fv < 0.15:
            return "Fair"
        elif b_fv < 0.30:
            return "Concerning"
        else:
            return "Problematic"
    
    @staticmethod
    def _get_conformance(b_prs: float, b_fi: float, b_fv: float) -> str:
        if b_prs > 0.70 or b_fi < 0.60 or b_fv > 0.30:
            return "SDCF-R-Bronze"
        if b_prs < 0.50 and b_fi > 0.70 and b_fv < 0.30:
            return "SDCF-A-Bronze"
        return "SDCF-P-Bronze"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BRONZE TIER RETROSPECTIVE VALIDATION")
    print("SDCF v1.0 - Synthetic Data Compliance Framework")
    print("="*70)
    print("\nTo run validation on your own dataset:")
    print("  1. Create a DatasetConfig object")
    print("  2. Initialize BronzeTierAssessor(config)")
    print("  3. Call assessor.assess()")
    print("\nSee the SDCF paper for methodology details.")
    print("="*70)

