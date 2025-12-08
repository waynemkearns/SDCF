# SDCF ASSESSMENT CHECKLIST
## Synthetic Data Compliance Framework - Assessment Checklist

**Assessment ID:** SDCF-[YYYY]-[NNN]  
**Assessment Date:** [Date]  
**Assessor Name:** [Name]  
**Dataset Name:** [Synthetic dataset identifier]

---

## PURPOSE

This checklist ensures comprehensive SDCF assessment. Complete all applicable sections based on assessment tier.

**Assessment Tier:** ☐ Gold   ☐ Silver   ☐ Bronze

**Legend:**
- ✅ = Completed and passed
- ⚠️ = Completed with concerns
- ❌ = Completed and failed
- N/A = Not applicable for this tier
- ⏸️ = Not yet completed

---

## SECTION 1: PRE-ASSESSMENT (Mandatory for All Tiers)

### 1.1 Dataset Identification
☐ Synthetic dataset name and version documented  
☐ Generation method/tool identified  
☐ Generation date recorded  
☐ Source data reference documented (if applicable)

### 1.2 Purpose Definition
☐ Intended use case clearly specified (use Purpose Definition Template)  
☐ Critical variables identified  
☐ Acceptable risk level defined (Conservative/Moderate/Aggressive)  
☐ Regulatory constraints documented (GDPR, EU AI Act, etc.)  
☐ Success criteria established

### 1.3 Tier Selection
☐ Source data availability assessed  
☐ Assessment tier selected (Gold/Silver/Bronze)  
☐ Tier selection rationale documented  
☐ Confidence level implications understood

### 1.4 Stakeholder Alignment
☐ Data owner identified and engaged  
☐ Risk officer notified  
☐ Compliance officer consulted (if applicable)  
☐ Business sponsor aligned on purpose and restrictions

**Pre-Assessment Status:** ☐ Complete   ☐ Incomplete

---

## SECTION 2: PRIVACY ASSESSMENT

### 2.1 Gold Tier Privacy Metrics (Required for Gold, N/A for Silver/Bronze)

#### Distance to Closest Record (DCR)
☐ DCR computed for all records  
☐ 10th percentile DCR: ______ (Reference: >0.01)  
☐ 50th percentile DCR: ______ (Threshold: >0.01, Target: >0.05)  
☐ 90th percentile DCR: ______ (Threshold: >0.05, Target: >0.10)  
☐ Outlier records (DCR <0.01) identified: ______ records  
☐ Outlier analysis conducted

**DCR Status:** ☐ Acceptable   ☐ Provisional   ☐ Rejected   ☐ N/A

#### K-Anonymity Proxy
☐ K-anonymity proxy computed  
☐ Minimum k-value: ______ (Threshold: k≥3, Target: k≥5)  
☐ Records failing k≥3: ______ (Threshold: <5%)  
☐ Quasi-identifiers used: [List]

**K-Anonymity Status:** ☐ Acceptable   ☐ Provisional   ☐ Rejected   ☐ N/A

#### Attribute Disclosure Risk
☐ Attribute disclosure risk computed for sensitive attributes  
☐ Mean disclosure risk: ______ % (Threshold: <5%, Target: <2%)  
☐ Maximum disclosure risk: ______ % (Threshold: <10%)  
☐ High-risk attributes identified: [List]

**Disclosure Risk Status:** ☐ Acceptable   ☐ Provisional   ☐ Rejected   ☐ N/A

### 2.2 Silver Tier Privacy Metrics (Required for Silver, N/A for Gold/Bronze)

#### Aggregate-Level Privacy
☐ Summary statistics comparison conducted  
☐ Cell suppression rules applied (if aggregates shared)  
☐ Group-level disclosure risk assessed  
☐ Privacy budget tracking implemented (if applicable)

**Silver Privacy Status:** ☐ Acceptable   ☐ Provisional   ☐ Rejected   ☐ N/A

### 2.3 Bronze Tier Privacy Limitations (Required for Bronze)

☐ Privacy cannot be verified (documented in limitations)  
☐ Generation method privacy guarantees documented (if claimed)  
☐ Limitations explicitly communicated to stakeholders  
☐ Usage restrictions defined to mitigate unknown privacy risks

**Bronze Privacy Status:** ☐ Acceptable (intrinsic only)   ☐ Rejected   ☐ N/A

### 2.4 Privacy Assessment Summary

**Overall Privacy Score:** ______ / 1.00  
**Privacy Classification:** ☐ Acceptable   ☐ Provisional   ☐ Rejected

**Privacy Concerns Documented:**
1. [Concern]
2. [Concern]

---

## SECTION 3: FIDELITY ASSESSMENT

### 3.1 Gold Tier Fidelity Metrics (Required for Gold, N/A for Silver/Bronze)

#### Univariate Distributions
☐ KS tests conducted for all continuous variables  
☐ Chi-square tests conducted for all categorical variables  
☐ % variables passing KS/chi-square (p>0.05): ______ % (Threshold: >85%)  
☐ Failed variables documented: [List]  
☐ Visual inspection of distributions completed

**Univariate Status:** ☐ Acceptable   ☐ Provisional   ☐ Rejected   ☐ N/A

#### Correlation Preservation
☐ Source data correlation matrix computed  
☐ Synthetic data correlation matrix computed  
☐ Mean absolute correlation difference: ______ (Threshold: <0.15, Target: <0.10)  
☐ % correlations preserved (±0.15): ______ % (Threshold: >80%)  
☐ Critical variable correlations specifically validated: [List]

**Correlation Status:** ☐ Acceptable   ☐ Provisional   ☐ Rejected   ☐ N/A

#### Multivariate Relationships
☐ Principal component analysis compared  
☐ Mutual information preservation assessed  
☐ Conditional distributions validated for critical relationships  
☐ Purpose-specific utility validated (e.g., model performance)

**Multivariate Status:** ☐ Acceptable   ☐ Provisional   ☐ Rejected   ☐ N/A

### 3.2 Silver Tier Fidelity Metrics (Required for Silver, N/A for Gold/Bronze)

#### Summary Statistics Comparison
☐ Means compared (% difference): ______ % (Threshold: <10%)  
☐ Standard deviations compared (% difference): ______ % (Threshold: <15%)  
☐ Medians compared (% difference): ______ % (Threshold: <10%)  
☐ Ranges validated (min/max reasonable)

**Summary Stats Status:** ☐ Acceptable   ☐ Provisional   ☐ Rejected   ☐ N/A

#### Schema and Constraints
☐ Schema validation passed (data types, column names)  
☐ Domain constraints validated (value ranges)  
☐ Business rules validated (e.g., age 0-120)  
☐ Referential integrity checked (if applicable)

**Schema Status:** ☐ Acceptable   ☐ Provisional   ☐ Rejected   ☐ N/A

### 3.3 Bronze Tier Fidelity Metrics (Required for Bronze)

#### Intrinsic Quality Checks
☐ Completeness: Missing values ______ % (Threshold: <5%)  
☐ Validity: Invalid values ______ % (Threshold: <1%)  
☐ Consistency: Logical inconsistencies ______ records (Threshold: <1%)  
☐ Uniqueness: Duplicate records ______ % (Threshold: <0.5%)

**Intrinsic Quality Status:** ☐ Acceptable   ☐ Provisional   ☐ Rejected   ☐ N/A

#### Face Validity
☐ Domain expert review conducted  
☐ Expert reviewer: [Name and Title]  
☐ Expert assessment: [Summary of findings]  
☐ Domain constraint violations: [List if any]

**Face Validity Status:** ☐ Acceptable   ☐ Provisional   ☐ Rejected   ☐ N/A

### 3.4 Fidelity Assessment Summary

**Overall Fidelity Score:** ______ / 1.00  
**Fidelity Classification:** ☐ Acceptable   ☐ Provisional   ☐ Rejected

**Fidelity Concerns Documented:**
1. [Concern]
2. [Concern]

---

## SECTION 4: FAIRNESS ASSESSMENT

### 4.1 Protected Attributes Identification

☐ Protected attributes identified based on:
  - ☐ GDPR Article 9 special categories (race, ethnicity, health, etc.)
  - ☐ EU AI Act Annex III considerations
  - ☐ Jurisdiction-specific protected classes
  - ☐ Domain-specific sensitive attributes

**Protected Attributes:** [List all protected attributes in dataset]

### 4.2 Demographic Parity Assessment

#### Group Representation
☐ Protected group distributions computed (source vs synthetic)  
☐ Demographic parity ratio computed for each group: [List ratios]  
☐ Threshold check: All ratios within 0.80-1.25? ☐ Yes  ☐ No  
☐ Underrepresented groups: [List]  
☐ Overrepresented groups: [List]

**Demographic Parity Status:** ☐ Acceptable   ☐ Provisional   ☐ Rejected   ☐ N/A

### 4.3 Group-Specific Fidelity

☐ Separate fidelity analysis conducted for each protected group  
☐ Within-group distribution preservation: ______ % (Threshold: >75%)  
☐ Groups with poor preservation: [List]  
☐ Intersectional analysis conducted (if applicable)

**Group Fidelity Status:** ☐ Acceptable   ☐ Provisional   ☐ Rejected   ☐ N/A

### 4.4 Utility Fairness (If Applicable)

☐ Downstream task fairness assessed (e.g., ML model performance)  
☐ Equalized odds computed (if classification task)  
☐ Equal opportunity computed (if classification task)  
☐ Calibration fairness assessed (if probabilistic outputs)

**Utility Fairness Status:** ☐ Acceptable   ☐ Provisional   ☐ Rejected   ☐ N/A

### 4.5 Fairness Assessment Summary

**Overall Fairness Score:** ______ / 1.00  
**Fairness Classification:** ☐ Acceptable   ☐ Provisional   ☐ Rejected

**Fairness Concerns Documented:**
1. [Concern]
2. [Concern]

---

## SECTION 5: INTEGRATED ASSESSMENT

### 5.1 Overall Scoring

**Dimension Scores:**
- Privacy: ______ / 1.00
- Fidelity: ______ / 1.00
- Fairness: ______ / 1.00

**Overall Score:** ______ / 1.00 (Average of three dimensions)

### 5.2 Conformance Determination Logic

☐ All three dimensions ≥0.70? (Minimum for any positive determination)

**If all ≥0.70:**
☐ All three dimensions ≥0.85? → **SDCF-A (Acceptable)**  
☐ Any dimension 0.70-0.84? → **SDCF-P (Provisional)**

**If any dimension <0.70:**
☐ → **SDCF-R (Rejected)**

**Conformance Determination:** ☐ SDCF-A   ☐ SDCF-P   ☐ SDCF-R

### 5.3 Limitations Documentation

☐ All assessment tier limitations documented  
☐ Metric limitations documented  
☐ Tool limitations documented  
☐ Temporal limitations documented (e.g., validity period)  
☐ Scope limitations documented (e.g., specific use case only)

**Key Limitations:**
1. [Limitation]
2. [Limitation]
3. [Limitation]

### 5.4 Usage Restrictions Definition

**Mandatory Restrictions:**
1. [Restriction]
2. [Restriction]

**Recommended Restrictions:**
1. [Restriction]
2. [Restriction]

**Prohibited Uses:**
1. [Prohibited use]
2. [Prohibited use]

---

## SECTION 6: DOCUMENTATION & SIGN-OFF

### 6.1 Required Documentation

☐ Purpose Definition Template completed  
☐ Technical Report generated  
☐ Conformance Certificate/Risk Statement prepared  
☐ Assessment Checklist completed (this document)  
☐ Tool outputs archived

### 6.2 Assessment Team Sign-Off

☐ Lead Assessor: [Name] - Signed: ______ - Date: ______  
☐ Technical Reviewer: [Name] - Signed: ______ - Date: ______  
☐ Domain Expert: [Name] - Signed: ______ - Date: ______

---

**SDCF Version:** 1.95  
**Checklist Template Version:** 1.0  
**Framework Author:** Wayne Kearns, Kaionix Labs  
**Website:** https://kaionixlabs.com/sdcf

© 2025 Wayne Kearns, Kaionix Labs. SDCF is licensed under CC BY-NC-SA 4.0.

