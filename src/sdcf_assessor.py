# SDCF Reference Implementation (Part 2)
# Main Assessor Class and Decision Logic
# Version: 1.95
# Author: Wayne Kearns, Kaionix Labs
# License: MIT

"""
This file contains the main SDCFAssessor class that orchestrates the assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from .sdcf_reference_implementation import (
    AssessmentTier,
    Conformance,
    DimensionStatus,
    DimensionScore,
    SDCFResults,
    PrivacyMetrics,
    FidelityMetrics,
    FairnessMetrics
)


# ============================================================================
# MAIN ASSESSOR CLASS
# ============================================================================

class SDCFAssessor:
    """
    Main SDCF Assessment class.
    
    Usage:
        assessor = SDCFAssessor(tier='gold')
        results = assessor.assess(
            source_data=df_source,
            synthetic_data=df_synthetic,
            categorical_columns=['gender', 'region'],
            protected_attributes=['gender', 'age_group'],
            sensitive_attributes=['medical_condition']
        )
        print(results.conformance)
    """
    
    # SDCF Thresholds (provisional values from framework)
    THRESHOLDS = {
        'privacy': {
            'dcr_90th_acceptable': 0.05,    # Threshold: DCR 90th > 0.05
            'dcr_50th_acceptable': 0.01,    # Threshold: DCR 50th > 0.01
            'k_anonymity_min': 3,            # Threshold: k >= 3
            'disclosure_risk_max': 5.0,      # Threshold: < 5%
        },
        'fidelity': {
            'ks_pass_rate_acceptable': 85.0,      # Threshold: >85% pass KS tests
            'correlation_preservation': 80.0,      # Threshold: >80% correlations preserved
            'mean_diff_max': 10.0,                 # Threshold: mean difference <10%
        },
        'fairness': {
            'parity_ratio_min': 0.80,       # Threshold: ratio >= 0.80
            'parity_ratio_max': 1.25,       # Threshold: ratio <= 1.25
            'group_preservation_min': 0.75, # Threshold: >75% preservation
        },
        'conformance': {
            'acceptable_threshold': 0.85,    # All dimensions >= 0.85 → SDCF-A
            'provisional_threshold': 0.70,   # Any dimension 0.70-0.84 → SDCF-P
            # Any dimension < 0.70 → SDCF-R
        }
    }
    
    def __init__(self, tier: str = 'gold'):
        """
        Initialize SDCF Assessor.
        
        Args:
            tier: Assessment tier - 'gold', 'silver', or 'bronze'
        """
        tier_lower = tier.lower()
        if tier_lower not in ['gold', 'silver', 'bronze']:
            raise ValueError(f"Invalid tier: {tier}. Must be 'gold', 'silver', or 'bronze'")
        
        self.tier = AssessmentTier(tier_lower)
        self.privacy_metrics = PrivacyMetrics()
        self.fidelity_metrics = FidelityMetrics()
        self.fairness_metrics = FairnessMetrics()
    
    def assess(self,
              source_data: Optional[pd.DataFrame] = None,
              synthetic_data: pd.DataFrame = None,
              categorical_columns: Optional[List[str]] = None,
              protected_attributes: Optional[List[str]] = None,
              sensitive_attributes: Optional[List[str]] = None,
              quasi_identifiers: Optional[List[str]] = None) -> SDCFResults:
        """
        Perform complete SDCF assessment.
        
        Args:
            source_data: Source dataset (required for Gold/Silver tier)
            synthetic_data: Synthetic dataset (required)
            categorical_columns: List of categorical column names
            protected_attributes: List of protected attributes for fairness
            sensitive_attributes: List of sensitive attributes for privacy
            quasi_identifiers: List of quasi-identifier columns
            
        Returns:
            SDCFResults object with complete assessment
        """
        if synthetic_data is None:
            raise ValueError("synthetic_data is required")
        
        # Tier-specific validations
        if self.tier in [AssessmentTier.GOLD, AssessmentTier.SILVER]:
            if source_data is None:
                raise ValueError(f"{self.tier.value} tier requires source_data")
        
        # Assess each dimension
        privacy_result = self._assess_privacy(
            source_data, synthetic_data, categorical_columns, 
            sensitive_attributes, quasi_identifiers
        )
        
        fidelity_result = self._assess_fidelity(
            source_data, synthetic_data
        )
        
        fairness_result = self._assess_fairness(
            source_data, synthetic_data, protected_attributes
        )
        
        # Compute overall score
        overall_score = (privacy_result.score + fidelity_result.score + fairness_result.score) / 3
        
        # Determine conformance
        conformance = self._determine_conformance(
            privacy_result, fidelity_result, fairness_result
        )
        
        # Generate limitations and restrictions
        limitations = self._generate_limitations()
        restrictions = self._generate_restrictions(
            conformance, privacy_result, fidelity_result, fairness_result
        )
        
        return SDCFResults(
            conformance=conformance,
            overall_score=overall_score,
            privacy=privacy_result,
            fidelity=fidelity_result,
            fairness=fairness_result,
            tier=self.tier,
            limitations=limitations,
            restrictions=restrictions
        )
    
    def _assess_privacy(self,
                       source_data: Optional[pd.DataFrame],
                       synthetic_data: pd.DataFrame,
                       categorical_columns: Optional[List[str]],
                       sensitive_attributes: Optional[List[str]],
                       quasi_identifiers: Optional[List[str]]) -> DimensionScore:
        """Assess privacy dimension"""
        
        metrics = {}
        concerns = []
        
        if self.tier == AssessmentTier.GOLD:
            # Gold Tier: Full privacy assessment
            
            # DCR metrics
            dcr_results = self.privacy_metrics.distance_to_closest_record(
                source_data, synthetic_data, categorical_columns
            )
            metrics.update(dcr_results)
            
            # Check DCR thresholds
            if dcr_results['dcr_90th_percentile'] < self.THRESHOLDS['privacy']['dcr_90th_acceptable']:
                concerns.append(f"DCR 90th percentile ({dcr_results['dcr_90th_percentile']:.3f}) "
                              f"below threshold ({self.THRESHOLDS['privacy']['dcr_90th_acceptable']})")
            
            if dcr_results['dcr_50th_percentile'] < self.THRESHOLDS['privacy']['dcr_50th_acceptable']:
                concerns.append(f"DCR 50th percentile ({dcr_results['dcr_50th_percentile']:.3f}) "
                              f"below threshold ({self.THRESHOLDS['privacy']['dcr_50th_acceptable']})")
            
            # K-anonymity proxy
            if quasi_identifiers:
                k_anon_results = self.privacy_metrics.k_anonymity_proxy(
                    source_data, synthetic_data, quasi_identifiers
                )
                metrics.update(k_anon_results)
                
                if k_anon_results['min_k'] < self.THRESHOLDS['privacy']['k_anonymity_min']:
                    concerns.append(f"Minimum k-anonymity ({k_anon_results['min_k']}) "
                                  f"below threshold ({self.THRESHOLDS['privacy']['k_anonymity_min']})")
            
            # Attribute disclosure risk
            if sensitive_attributes:
                disclosure_results = self.privacy_metrics.attribute_disclosure_risk(
                    source_data, synthetic_data, sensitive_attributes
                )
                metrics.update(disclosure_results)
                
                if disclosure_results['mean_disclosure_risk'] > self.THRESHOLDS['privacy']['disclosure_risk_max']:
                    concerns.append(f"Mean disclosure risk ({disclosure_results['mean_disclosure_risk']:.1f}%) "
                                  f"above threshold ({self.THRESHOLDS['privacy']['disclosure_risk_max']}%)")
            
            # Compute privacy score
            score = self._compute_privacy_score_gold(metrics)
            
        elif self.tier == AssessmentTier.SILVER:
            # Silver Tier: Aggregate-level privacy
            concerns.append("Silver Tier: Limited privacy assessment without full source data")
            metrics['note'] = 'Aggregate-level privacy assessment only'
            score = 0.75  # Medium confidence
            
        else:  # Bronze
            # Bronze Tier: No privacy verification possible
            concerns.append("Bronze Tier: Privacy cannot be verified without source data")
            metrics['note'] = 'Privacy verification not possible at Bronze Tier'
            score = 0.60  # Low confidence - default provisional
        
        # Determine status
        if score >= self.THRESHOLDS['conformance']['acceptable_threshold']:
            status = DimensionStatus.ACCEPTABLE
        elif score >= self.THRESHOLDS['conformance']['provisional_threshold']:
            status = DimensionStatus.PROVISIONAL
        else:
            status = DimensionStatus.REJECTED
        
        return DimensionScore(
            score=score,
            status=status,
            metrics=metrics,
            concerns=concerns
        )
    
    def _assess_fidelity(self,
                        source_data: Optional[pd.DataFrame],
                        synthetic_data: pd.DataFrame) -> DimensionScore:
        """Assess fidelity dimension"""
        
        metrics = {}
        concerns = []
        
        if self.tier == AssessmentTier.GOLD:
            # Gold Tier: Full fidelity assessment
            
            # Univariate distributions
            univariate_results = self.fidelity_metrics.univariate_distribution_similarity(
                source_data, synthetic_data
            )
            metrics.update(univariate_results)
            
            if univariate_results['ks_test_pass_rate_pct'] < self.THRESHOLDS['fidelity']['ks_pass_rate_acceptable']:
                concerns.append(f"KS test pass rate ({univariate_results['ks_test_pass_rate_pct']:.1f}%) "
                              f"below threshold ({self.THRESHOLDS['fidelity']['ks_pass_rate_acceptable']}%)")
            
            # Correlation preservation
            corr_results = self.fidelity_metrics.correlation_preservation(
                source_data, synthetic_data
            )
            metrics.update(corr_results)
            
            if corr_results['correlations_within_015_pct'] < self.THRESHOLDS['fidelity']['correlation_preservation']:
                concerns.append(f"Correlation preservation ({corr_results['correlations_within_015_pct']:.1f}%) "
                              f"below threshold ({self.THRESHOLDS['fidelity']['correlation_preservation']}%)")
            
            # Compute fidelity score
            score = self._compute_fidelity_score_gold(metrics)
            
        elif self.tier == AssessmentTier.SILVER:
            # Silver Tier: Summary statistics comparison
            summary_results = self.fidelity_metrics.summary_statistics_comparison(
                source_data, synthetic_data
            )
            metrics.update(summary_results)
            
            if summary_results['mean_difference_pct'] > self.THRESHOLDS['fidelity']['mean_diff_max']:
                concerns.append(f"Mean difference ({summary_results['mean_difference_pct']:.1f}%) "
                              f"above threshold ({self.THRESHOLDS['fidelity']['mean_diff_max']}%)")
            
            score = self._compute_fidelity_score_silver(metrics)
            
        else:  # Bronze
            # Bronze Tier: Intrinsic quality only
            concerns.append("Bronze Tier: Limited fidelity assessment (intrinsic quality only)")
            
            # Basic quality checks
            metrics['completeness_pct'] = float((1 - synthetic_data.isnull().sum().sum() / 
                                               (synthetic_data.shape[0] * synthetic_data.shape[1])) * 100)
            metrics['n_records'] = len(synthetic_data)
            metrics['n_columns'] = len(synthetic_data.columns)
            
            score = 0.70  # Bronze tier gets provisional by default
        
        # Determine status
        if score >= self.THRESHOLDS['conformance']['acceptable_threshold']:
            status = DimensionStatus.ACCEPTABLE
        elif score >= self.THRESHOLDS['conformance']['provisional_threshold']:
            status = DimensionStatus.PROVISIONAL
        else:
            status = DimensionStatus.REJECTED
        
        return DimensionScore(
            score=score,
            status=status,
            metrics=metrics,
            concerns=concerns
        )
    
    def _assess_fairness(self,
                        source_data: Optional[pd.DataFrame],
                        synthetic_data: pd.DataFrame,
                        protected_attributes: Optional[List[str]]) -> DimensionScore:
        """Assess fairness dimension"""
        
        metrics = {}
        concerns = []
        
        if not protected_attributes:
            # No protected attributes identified
            metrics['note'] = 'No protected attributes specified for fairness assessment'
            score = 1.0  # N/A - acceptable by default
            status = DimensionStatus.ACCEPTABLE
            
        elif self.tier in [AssessmentTier.GOLD, AssessmentTier.SILVER] and source_data is not None:
            # Fairness assessment with source data
            
            # Demographic parity
            parity_results = self.fairness_metrics.demographic_parity(
                source_data, synthetic_data, protected_attributes
            )
            metrics.update(parity_results)
            
            if not parity_results['all_ratios_acceptable']:
                concerns.extend(parity_results['concerns'])
            
            # Group-level fidelity
            group_results = self.fairness_metrics.group_fidelity_preservation(
                source_data, synthetic_data, protected_attributes
            )
            metrics.update(group_results)
            
            if group_results['mean_group_preservation'] < self.THRESHOLDS['fairness']['group_preservation_min']:
                concerns.append(f"Group-level preservation ({group_results['mean_group_preservation']:.2f}) "
                              f"below threshold ({self.THRESHOLDS['fairness']['group_preservation_min']})")
            
            # Compute fairness score
            score = self._compute_fairness_score(metrics)
            
            # Determine status
            if score >= self.THRESHOLDS['conformance']['acceptable_threshold']:
                status = DimensionStatus.ACCEPTABLE
            elif score >= self.THRESHOLDS['conformance']['provisional_threshold']:
                status = DimensionStatus.PROVISIONAL
            else:
                status = DimensionStatus.REJECTED
                
        else:
            # Bronze tier or missing source data
            concerns.append("Limited fairness assessment without source data comparison")
            metrics['note'] = 'Fairness verification limited at this tier'
            score = 0.75  # Provisional by default
            status = DimensionStatus.PROVISIONAL
        
        return DimensionScore(
            score=score,
            status=status,
            metrics=metrics,
            concerns=concerns
        )
    
    def _compute_privacy_score_gold(self, metrics: Dict) -> float:
        """Compute privacy score for Gold tier"""
        score_components = []
        
        # DCR score (weight: 40%)
        dcr_90 = metrics.get('dcr_90th_percentile', 0)
        dcr_score = min(1.0, dcr_90 / 0.10)  # Normalize: 0.10 = excellent
        score_components.append(('dcr', dcr_score, 0.40))
        
        # K-anonymity score (weight: 30%)
        if 'min_k' in metrics:
            k_score = min(1.0, metrics['min_k'] / 5.0)  # Normalize: k=5 = excellent
            score_components.append(('k_anon', k_score, 0.30))
        
        # Disclosure risk score (weight: 30%)
        if 'mean_disclosure_risk' in metrics:
            disclosure_risk = metrics['mean_disclosure_risk']
            disclosure_score = max(0.0, 1.0 - disclosure_risk / 10.0)  # Lower is better
            score_components.append(('disclosure', disclosure_score, 0.30))
        
        # Weighted average
        total_weight = sum(w for _, _, w in score_components)
        if total_weight > 0:
            weighted_score = sum(s * w for _, s, w in score_components) / total_weight
        else:
            weighted_score = 0.5
        
        return float(weighted_score)
    
    def _compute_fidelity_score_gold(self, metrics: Dict) -> float:
        """Compute fidelity score for Gold tier"""
        score_components = []
        
        # KS test pass rate (weight: 40%)
        if 'ks_test_pass_rate_pct' in metrics:
            ks_score = metrics['ks_test_pass_rate_pct'] / 100.0
            score_components.append(('ks', ks_score, 0.40))
        
        # Correlation preservation (weight: 40%)
        if 'correlations_within_015_pct' in metrics:
            corr_score = metrics['correlations_within_015_pct'] / 100.0
            score_components.append(('corr', corr_score, 0.40))
        
        # Distribution preservation (weight: 20%)
        if 'correlation_preservation_pct' in metrics:
            dist_score = metrics['correlation_preservation_pct'] / 100.0
            score_components.append(('dist', dist_score, 0.20))
        
        # Weighted average
        total_weight = sum(w for _, _, w in score_components)
        if total_weight > 0:
            weighted_score = sum(s * w for _, s, w in score_components) / total_weight
        else:
            weighted_score = 0.5
        
        return float(weighted_score)
    
    def _compute_fidelity_score_silver(self, metrics: Dict) -> float:
        """Compute fidelity score for Silver tier"""
        score_components = []
        
        # Mean difference (lower is better)
        if 'mean_difference_pct' in metrics:
            mean_diff = metrics['mean_difference_pct']
            mean_score = max(0.0, 1.0 - mean_diff / 20.0)  # 20% diff = 0 score
            score_components.append(mean_score)
        
        # Std difference
        if 'std_difference_pct' in metrics:
            std_diff = metrics['std_difference_pct']
            std_score = max(0.0, 1.0 - std_diff / 30.0)  # 30% diff = 0 score
            score_components.append(std_score)
        
        return float(np.mean(score_components)) if score_components else 0.5
    
    def _compute_fairness_score(self, metrics: Dict) -> float:
        """Compute fairness score"""
        score_components = []
        
        # Demographic parity (weight: 50%)
        if 'all_ratios_acceptable' in metrics:
            if metrics['all_ratios_acceptable']:
                parity_score = 1.0
            else:
                # Partial credit based on how many ratios are acceptable
                min_ratio = metrics.get('min_parity_ratio', 0.80)
                max_ratio = metrics.get('max_parity_ratio', 1.25)
                
                # Distance from acceptable range
                min_distance = max(0, 0.80 - min_ratio)
                max_distance = max(0, max_ratio - 1.25)
                total_distance = min_distance + max_distance
                
                parity_score = max(0.0, 1.0 - total_distance / 0.50)  # 0.50 distance = 0 score
            
            score_components.append(('parity', parity_score, 0.50))
        
        # Group preservation (weight: 50%)
        if 'mean_group_preservation' in metrics:
            group_score = metrics['mean_group_preservation']
            score_components.append(('group', group_score, 0.50))
        
        # Weighted average
        total_weight = sum(w for _, _, w in score_components)
        if total_weight > 0:
            weighted_score = sum(s * w for _, s, w in score_components) / total_weight
        else:
            weighted_score = 0.75  # Default if no metrics
        
        return float(weighted_score)
    
    def _determine_conformance(self,
                              privacy: DimensionScore,
                              fidelity: DimensionScore,
                              fairness: DimensionScore) -> Conformance:
        """
        Apply SDCF decision logic to determine overall conformance.
        
        Logic:
        - If ANY dimension < 0.70 → REJECTED
        - If all dimensions >= 0.85 → ACCEPTABLE
        - Otherwise → PROVISIONAL
        """
        min_score = min(privacy.score, fidelity.score, fairness.score)
        
        if min_score < self.THRESHOLDS['conformance']['provisional_threshold']:
            return Conformance.REJECTED
        elif all(score >= self.THRESHOLDS['conformance']['acceptable_threshold'] 
                for score in [privacy.score, fidelity.score, fairness.score]):
            return Conformance.ACCEPTABLE
        else:
            return Conformance.PROVISIONAL
    
    def _generate_limitations(self) -> List[str]:
        """Generate tier-specific limitations"""
        limitations = []
        
        if self.tier == AssessmentTier.BRONZE:
            limitations.append("Bronze Tier: Privacy cannot be verified without source data access")
            limitations.append("Bronze Tier: Fidelity assessment limited to intrinsic quality only")
            limitations.append("Bronze Tier: Fairness assessment has low confidence")
        elif self.tier == AssessmentTier.SILVER:
            limitations.append("Silver Tier: Privacy assessment based on aggregate-level analysis only")
            limitations.append("Silver Tier: Fidelity assessment limited to summary statistics")
        
        limitations.append("Assessment is purpose-bounded - valid only for specified use case")
        limitations.append("Thresholds are provisional and subject to empirical validation")
        
        return limitations
    
    def _generate_restrictions(self,
                              conformance: Conformance,
                              privacy: DimensionScore,
                              fidelity: DimensionScore,
                              fairness: DimensionScore) -> List[str]:
        """Generate usage restrictions based on assessment results"""
        restrictions = []
        
        if conformance == Conformance.REJECTED:
            restrictions.append("DO NOT USE for intended purpose without remediation")
            restrictions.append("Requires re-generation or significant improvement")
        
        elif conformance == Conformance.PROVISIONAL:
            restrictions.append("Use only with explicit stakeholder acknowledgment of risks")
            restrictions.append("Implement mandatory safeguards (human review, monitoring)")
            
            if privacy.status != DimensionStatus.ACCEPTABLE:
                restrictions.append("HIGH RISK: Privacy concerns - restrict to internal use only")
            
            if fidelity.status != DimensionStatus.ACCEPTABLE:
                restrictions.append("Fidelity concerns - not suitable for precision-critical applications")
            
            if fairness.status != DimensionStatus.ACCEPTABLE:
                restrictions.append("Fairness concerns - avoid fairness-sensitive decisions")
        
        else:  # ACCEPTABLE
            restrictions.append("Use only for assessed purpose - do not repurpose without re-assessment")
            restrictions.append("Re-assess annually or when conditions change")
        
        return restrictions


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("SDCF Reference Implementation - Example Usage")
    print("=" * 60)
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    source_df = pd.DataFrame({
        'age': np.random.normal(45, 15, n_samples).clip(18, 90),
        'income': np.random.normal(50000, 20000, n_samples).clip(20000, 150000),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'score': np.random.normal(70, 15, n_samples).clip(0, 100)
    })
    
    # Generate synthetic data (with some noise)
    synthetic_df = source_df.copy()
    synthetic_df['age'] += np.random.normal(0, 2, n_samples)
    synthetic_df['income'] += np.random.normal(0, 5000, n_samples)
    
    # Perform SDCF assessment
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
    print(f"\nConformance: {results.conformance.value}")
    print(f"Overall Score: {results.overall_score:.2f}")
    print(f"\nPrivacy: {results.privacy.score:.2f} ({results.privacy.status.value})")
    print(f"Fidelity: {results.fidelity.score:.2f} ({results.fidelity.status.value})")
    print(f"Fairness: {results.fairness.score:.2f} ({results.fairness.status.value})")
    
    print(f"\nLimitations:")
    for limitation in results.limitations:
        print(f"  - {limitation}")
    
    print(f"\nRestrictions:")
    for restriction in results.restrictions:
        print(f"  - {restriction}")
    
    # Export to JSON
    import json
    print(f"\nJSON Export:")
    print(json.dumps(results.to_dict(), indent=2))

