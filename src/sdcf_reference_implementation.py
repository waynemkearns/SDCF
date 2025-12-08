# SDCF Reference Implementation
# Synthetic Data Compliance Framework - Python Package
# Version: 1.0
# Author: Wayne Kearns, Kaionix Labs
# License: MIT

"""
SDCF Reference Implementation

This package provides reference implementations of:
1. Metric computation for Privacy, Fidelity, and Fairness
2. SDCF decision logic for conformance determination
3. Certificate generation helpers

USAGE:
    from sdcf import SDCFAssessor
    
    assessor = SDCFAssessor(tier='gold')
    results = assessor.assess(source_data=df_real, synthetic_data=df_synthetic)
    print(results.conformance)  # 'SDCF-A', 'SDCF-P', or 'SDCF-R'

REQUIREMENTS:
    - pandas >= 1.3.0
    - numpy >= 1.20.0
    - scipy >= 1.7.0
    - scikit-learn >= 0.24.0
    - sdmetrics >= 0.10.0 (optional, for advanced metrics)

INSTALLATION:
    pip install pandas numpy scipy scikit-learn
    pip install sdmetrics  # optional
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import distance
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# ENUMERATIONS
# ============================================================================

class AssessmentTier(Enum):
    """SDCF Assessment Tier"""
    GOLD = "gold"      # Full source data access
    SILVER = "silver"  # Partial source data
    BRONZE = "bronze"  # No source data


class Conformance(Enum):
    """SDCF Conformance Determination"""
    ACCEPTABLE = "SDCF-A"      # Acceptable for purpose
    PROVISIONAL = "SDCF-P"     # Provisional with restrictions
    REJECTED = "SDCF-R"         # Rejected, not fit for purpose


class DimensionStatus(Enum):
    """Status for each assessment dimension"""
    ACCEPTABLE = "acceptable"
    PROVISIONAL = "provisional"
    REJECTED = "rejected"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DimensionScore:
    """Score for a single dimension (Privacy, Fidelity, or Fairness)"""
    score: float  # 0.0 to 1.0
    status: DimensionStatus
    metrics: Dict[str, float]
    concerns: List[str]


@dataclass
class SDCFResults:
    """Complete SDCF Assessment Results"""
    conformance: Conformance
    overall_score: float
    privacy: DimensionScore
    fidelity: DimensionScore
    fairness: DimensionScore
    tier: AssessmentTier
    limitations: List[str]
    restrictions: List[str]
    
    def to_dict(self) -> Dict:
        """Convert results to dictionary for JSON serialization"""
        return {
            'conformance': self.conformance.value,
            'overall_score': self.overall_score,
            'tier': self.tier.value,
            'privacy': {
                'score': self.privacy.score,
                'status': self.privacy.status.value,
                'metrics': self.privacy.metrics,
                'concerns': self.privacy.concerns
            },
            'fidelity': {
                'score': self.fidelity.score,
                'status': self.fidelity.status.value,
                'metrics': self.fidelity.metrics,
                'concerns': self.fidelity.concerns
            },
            'fairness': {
                'score': self.fairness.score,
                'status': self.fairness.status.value,
                'metrics': self.fairness.metrics,
                'concerns': self.fairness.concerns
            },
            'limitations': self.limitations,
            'restrictions': self.restrictions
        }


# ============================================================================
# PRIVACY METRICS
# ============================================================================

class PrivacyMetrics:
    """Compute privacy metrics for SDCF assessment"""
    
    @staticmethod
    def distance_to_closest_record(source: pd.DataFrame, 
                                   synthetic: pd.DataFrame,
                                   categorical_columns: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute Distance to Closest Record (DCR) metric.
        
        DCR measures how far each synthetic record is from its nearest neighbor
        in the source data. Low DCR indicates high re-identification risk.
        
        Args:
            source: Source dataset
            synthetic: Synthetic dataset
            categorical_columns: List of categorical column names
            
        Returns:
            Dictionary with DCR percentiles
        """
        # Identify numeric and categorical columns
        if categorical_columns is None:
            categorical_columns = source.select_dtypes(include=['object', 'category']).columns.tolist()
        
        numeric_columns = source.select_dtypes(include=[np.number]).columns.tolist()
        
        # Normalize numeric columns
        source_numeric = source[numeric_columns].copy()
        synthetic_numeric = synthetic[numeric_columns].copy()
        
        # Standardize
        means = source_numeric.mean()
        stds = source_numeric.std()
        stds[stds == 0] = 1  # Avoid division by zero
        
        source_normalized = (source_numeric - means) / stds
        synthetic_normalized = (synthetic_numeric - means) / stds
        
        # One-hot encode categorical columns
        source_cat = pd.get_dummies(source[categorical_columns]) if categorical_columns else pd.DataFrame()
        synthetic_cat = pd.get_dummies(synthetic[categorical_columns]) if categorical_columns else pd.DataFrame()
        
        # Align columns
        all_columns = set(source_cat.columns) | set(synthetic_cat.columns)
        for col in all_columns:
            if col not in source_cat.columns:
                source_cat[col] = 0
            if col not in synthetic_cat.columns:
                synthetic_cat[col] = 0
        
        source_cat = source_cat[sorted(all_columns)]
        synthetic_cat = synthetic_cat[sorted(all_columns)]
        
        # Combine normalized numeric and categorical
        source_combined = pd.concat([source_normalized, source_cat], axis=1).fillna(0).values
        synthetic_combined = pd.concat([synthetic_normalized, synthetic_cat], axis=1).fillna(0).values
        
        # Compute distances
        dcr_values = []
        for syn_record in synthetic_combined:
            distances = [distance.euclidean(syn_record, src_record) 
                        for src_record in source_combined]
            min_distance = min(distances)
            dcr_values.append(min_distance)
        
        # Normalize DCR by dimensionality (sqrt of feature count)
        n_features = synthetic_combined.shape[1]
        dcr_values = [d / np.sqrt(n_features) for d in dcr_values]
        
        return {
            'dcr_10th_percentile': float(np.percentile(dcr_values, 10)),
            'dcr_50th_percentile': float(np.percentile(dcr_values, 50)),
            'dcr_90th_percentile': float(np.percentile(dcr_values, 90)),
            'dcr_mean': float(np.mean(dcr_values)),
            'dcr_min': float(np.min(dcr_values))
        }
    
    @staticmethod
    def k_anonymity_proxy(source: pd.DataFrame, 
                         synthetic: pd.DataFrame,
                         quasi_identifiers: List[str]) -> Dict[str, float]:
        """
        Compute k-anonymity proxy metric.
        
        Groups records by quasi-identifiers and computes minimum group size.
        
        Args:
            source: Source dataset
            synthetic: Synthetic dataset
            quasi_identifiers: List of quasi-identifier column names
            
        Returns:
            Dictionary with k-anonymity metrics
        """
        # Group synthetic records by quasi-identifiers
        grouped = synthetic[quasi_identifiers].groupby(quasi_identifiers).size()
        
        k_values = grouped.values
        min_k = int(np.min(k_values))
        mean_k = float(np.mean(k_values))
        
        # Count records with k < 3
        records_below_k3 = int(np.sum(k_values[k_values < 3]))
        pct_below_k3 = float(records_below_k3 / len(synthetic) * 100)
        
        return {
            'min_k': min_k,
            'mean_k': mean_k,
            'records_below_k3': records_below_k3,
            'pct_records_below_k3': pct_below_k3
        }
    
    @staticmethod
    def attribute_disclosure_risk(source: pd.DataFrame, 
                                  synthetic: pd.DataFrame,
                                  sensitive_attributes: List[str]) -> Dict[str, float]:
        """
        Compute attribute disclosure risk.
        
        Measures how much sensitive attribute values can be inferred from
        synthetic data.
        
        Args:
            source: Source dataset
            synthetic: Synthetic dataset
            sensitive_attributes: List of sensitive attribute column names
            
        Returns:
            Dictionary with disclosure risk metrics
        """
        risks = []
        
        for attr in sensitive_attributes:
            # Simple proxy: compare distribution similarity
            if source[attr].dtype in [np.int64, np.float64]:
                # Numeric attribute - use KS test
                ks_stat, _ = stats.ks_2samp(source[attr].dropna(), 
                                           synthetic[attr].dropna())
                risk = float(1 - ks_stat)  # High similarity = high risk
            else:
                # Categorical attribute - compare proportions
                source_dist = source[attr].value_counts(normalize=True)
                synthetic_dist = synthetic[attr].value_counts(normalize=True)
                
                # Align indices
                all_values = set(source_dist.index) | set(synthetic_dist.index)
                similarity = 0
                for val in all_values:
                    source_pct = source_dist.get(val, 0)
                    synthetic_pct = synthetic_dist.get(val, 0)
                    similarity += min(source_pct, synthetic_pct)
                
                risk = float(similarity)
            
            risks.append(risk)
        
        return {
            'mean_disclosure_risk': float(np.mean(risks) * 100),  # As percentage
            'max_disclosure_risk': float(np.max(risks) * 100),
            'per_attribute_risk': {attr: float(risk * 100) 
                                  for attr, risk in zip(sensitive_attributes, risks)}
        }


# ============================================================================
# FIDELITY METRICS
# ============================================================================

class FidelityMetrics:
    """Compute fidelity metrics for SDCF assessment"""
    
    @staticmethod
    def univariate_distribution_similarity(source: pd.DataFrame,
                                          synthetic: pd.DataFrame) -> Dict[str, float]:
        """
        Compute univariate distribution similarity using KS tests.
        
        Args:
            source: Source dataset
            synthetic: Synthetic dataset
            
        Returns:
            Dictionary with distribution similarity metrics
        """
        results = []
        failed_variables = []
        
        numeric_columns = source.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in synthetic.columns:
                ks_stat, p_value = stats.ks_2samp(source[col].dropna(), 
                                                  synthetic[col].dropna())
                
                passed = p_value > 0.05
                results.append({
                    'variable': col,
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'passed': passed
                })
                
                if not passed:
                    failed_variables.append(col)
        
        pass_rate = float(sum(r['passed'] for r in results) / len(results) * 100) if results else 0.0
        
        return {
            'ks_test_pass_rate_pct': pass_rate,
            'n_variables_tested': len(results),
            'n_variables_failed': len(failed_variables),
            'failed_variables': failed_variables,
            'per_variable_results': results
        }
    
    @staticmethod
    def correlation_preservation(source: pd.DataFrame,
                                synthetic: pd.DataFrame) -> Dict[str, float]:
        """
        Compute correlation preservation metrics.
        
        Args:
            source: Source dataset
            synthetic: Synthetic dataset
            
        Returns:
            Dictionary with correlation metrics
        """
        numeric_columns = source.select_dtypes(include=[np.number]).columns.tolist()
        common_columns = [col for col in numeric_columns if col in synthetic.columns]
        
        if len(common_columns) < 2:
            return {
                'mean_correlation_difference': 0.0,
                'correlation_preservation_pct': 100.0,
                'note': 'Insufficient numeric columns for correlation analysis'
            }
        
        source_corr = source[common_columns].corr()
        synthetic_corr = synthetic[common_columns].corr()
        
        # Compute absolute differences (excluding diagonal)
        diff_matrix = np.abs(source_corr.values - synthetic_corr.values)
        n = len(common_columns)
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle_indices = np.triu_indices(n, k=1)
        differences = diff_matrix[upper_triangle_indices]
        
        mean_diff = float(np.mean(differences))
        preservation_pct = float((1 - mean_diff) * 100)
        
        # Count correlations preserved within ±0.15
        preserved_count = int(np.sum(differences < 0.15))
        total_count = len(differences)
        preservation_rate = float(preserved_count / total_count * 100) if total_count > 0 else 100.0
        
        return {
            'mean_correlation_difference': mean_diff,
            'correlation_preservation_pct': preservation_pct,
            'correlations_within_015_pct': preservation_rate,
            'n_correlations_tested': total_count
        }
    
    @staticmethod
    def summary_statistics_comparison(source: pd.DataFrame,
                                     synthetic: pd.DataFrame) -> Dict[str, float]:
        """
        Compare summary statistics between source and synthetic data.
        
        Useful for Silver Tier assessments.
        
        Args:
            source: Source dataset
            synthetic: Synthetic dataset
            
        Returns:
            Dictionary with summary statistic comparisons
        """
        numeric_columns = source.select_dtypes(include=[np.number]).columns.tolist()
        common_columns = [col for col in numeric_columns if col in synthetic.columns]
        
        mean_diffs = []
        std_diffs = []
        median_diffs = []
        
        for col in common_columns:
            source_mean = source[col].mean()
            synthetic_mean = synthetic[col].mean()
            mean_diff = abs(source_mean - synthetic_mean) / (abs(source_mean) + 1e-10) * 100
            mean_diffs.append(mean_diff)
            
            source_std = source[col].std()
            synthetic_std = synthetic[col].std()
            std_diff = abs(source_std - synthetic_std) / (abs(source_std) + 1e-10) * 100
            std_diffs.append(std_diff)
            
            source_median = source[col].median()
            synthetic_median = synthetic[col].median()
            median_diff = abs(source_median - synthetic_median) / (abs(source_median) + 1e-10) * 100
            median_diffs.append(median_diff)
        
        return {
            'mean_difference_pct': float(np.mean(mean_diffs)),
            'std_difference_pct': float(np.mean(std_diffs)),
            'median_difference_pct': float(np.mean(median_diffs)),
            'max_mean_difference_pct': float(np.max(mean_diffs)) if mean_diffs else 0.0
        }


# ============================================================================
# FAIRNESS METRICS
# ============================================================================

class FairnessMetrics:
    """Compute fairness metrics for SDCF assessment"""
    
    @staticmethod
    def demographic_parity(source: pd.DataFrame,
                          synthetic: pd.DataFrame,
                          protected_attributes: List[str]) -> Dict[str, float]:
        """
        Compute demographic parity ratios for protected attributes.
        
        Args:
            source: Source dataset
            synthetic: Synthetic dataset
            protected_attributes: List of protected attribute column names
            
        Returns:
            Dictionary with demographic parity metrics
        """
        ratios = []
        group_results = {}
        concerns = []
        
        for attr in protected_attributes:
            if attr not in source.columns or attr not in synthetic.columns:
                continue
            
            source_dist = source[attr].value_counts(normalize=True)
            synthetic_dist = synthetic[attr].value_counts(normalize=True)
            
            # Compute ratio for each group
            for group in source_dist.index:
                source_pct = source_dist.get(group, 0)
                synthetic_pct = synthetic_dist.get(group, 0)
                
                if source_pct > 0:
                    ratio = synthetic_pct / source_pct
                    ratios.append(ratio)
                    
                    group_key = f"{attr}_{group}"
                    group_results[group_key] = {
                        'source_pct': float(source_pct * 100),
                        'synthetic_pct': float(synthetic_pct * 100),
                        'ratio': float(ratio)
                    }
                    
                    # Check if ratio is outside acceptable range (0.80-1.25)
                    if ratio < 0.80 or ratio > 1.25:
                        concerns.append(f"{attr}={group}: ratio {ratio:.2f} outside 0.80-1.25")
        
        min_ratio = float(np.min(ratios)) if ratios else 1.0
        max_ratio = float(np.max(ratios)) if ratios else 1.0
        mean_ratio = float(np.mean(ratios)) if ratios else 1.0
        
        # Check if all ratios within acceptable range
        all_acceptable = all(0.80 <= r <= 1.25 for r in ratios)
        
        return {
            'min_parity_ratio': min_ratio,
            'max_parity_ratio': max_ratio,
            'mean_parity_ratio': mean_ratio,
            'all_ratios_acceptable': all_acceptable,
            'n_groups_tested': len(ratios),
            'per_group_results': group_results,
            'concerns': concerns
        }
    
    @staticmethod
    def group_fidelity_preservation(source: pd.DataFrame,
                                   synthetic: pd.DataFrame,
                                   protected_attributes: List[str]) -> Dict[str, float]:
        """
        Assess whether statistical fidelity is preserved within protected groups.
        
        Args:
            source: Source dataset
            synthetic: Synthetic dataset
            protected_attributes: List of protected attribute column names
            
        Returns:
            Dictionary with group-specific fidelity metrics
        """
        group_preservation_scores = []
        
        for attr in protected_attributes:
            if attr not in source.columns or attr not in synthetic.columns:
                continue
            
            for group in source[attr].unique():
                source_group = source[source[attr] == group]
                synthetic_group = synthetic[synthetic[attr] == group]
                
                if len(source_group) < 10 or len(synthetic_group) < 10:
                    continue  # Skip small groups
                
                # Compute fidelity for this group
                numeric_cols = source_group.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) == 0:
                    continue
                
                # Simple fidelity proxy: mean absolute percentage difference
                diffs = []
                for col in numeric_cols:
                    source_mean = source_group[col].mean()
                    synthetic_mean = synthetic_group[col].mean()
                    
                    if abs(source_mean) > 1e-10:
                        diff = abs(source_mean - synthetic_mean) / abs(source_mean)
                        diffs.append(diff)
                
                if diffs:
                    group_score = 1 - np.mean(diffs)
                    group_preservation_scores.append(group_score)
        
        if not group_preservation_scores:
            return {
                'mean_group_preservation': 1.0,
                'note': 'Insufficient data for group-level analysis'
            }
        
        return {
            'mean_group_preservation': float(np.mean(group_preservation_scores)),
            'min_group_preservation': float(np.min(group_preservation_scores)),
            'n_groups_analyzed': len(group_preservation_scores)
        }

