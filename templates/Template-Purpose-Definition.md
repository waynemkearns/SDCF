# SDCF PURPOSE DEFINITION TEMPLATE
## Define Your Synthetic Data Use Case

**Assessment ID:** SDCF-[YYYY]-[NNN]  
**Date:** [Date]  
**Completed By:** [Name, Title, Organization]

---

## INTRODUCTION

**Why Purpose Definition Matters:**

SDCF assessments are **purpose-bounded** - meaning synthetic data is evaluated for fitness for a *specific* use case, not generic "quality." The same dataset might be acceptable for software testing but unacceptable for ML model training.

This template helps you clearly define:
- **What** you want to use the synthetic data for
- **How** you will use it
- **Why** specific qualities matter for your use case
- **What** risks are acceptable

**Complete this template BEFORE beginning technical assessment.**

---

## SECTION 1: PRIMARY USE CASE

### 1.1 High-Level Purpose

**In one sentence, what is the primary purpose of this synthetic data?**

[Example: "Train a logistic regression model to predict customer churn for use in marketing campaigns"]

_Your answer:_  
________________________________________________________________________________

### 1.2 Detailed Use Case Description

**Provide a detailed description (3-5 sentences) of exactly how this synthetic data will be used:**

Include:
- Who will use it (team, role)
- What they will do with it (specific activities)
- How outputs will be used (decisions, insights, systems)
- Frequency of use (one-time, recurring, continuous)

_Your answer:_  
________________________________________________________________________________

### 1.3 Use Case Category

**Select the primary category that best describes your use case:**

☐ **Machine Learning Training**  
☐ **Machine Learning Testing/Validation**  
☐ **Statistical Analysis**  
☐ **Software Development/QA**  
☐ **Data Sharing (External)**  
☐ **Data Sharing (Internal)**  
☐ **Exploratory Data Analysis**  
☐ **Reporting/Dashboards**  
☐ **Compliance Demonstration**  
☐ **Other:** [Specify] ________________________

---

## SECTION 2: CRITICAL VARIABLES

### 2.1 Variable Identification

**List the variables (columns) that are MOST CRITICAL for your use case:**

| Variable Name | Type | Why Critical for Purpose |
|--------------|------|-------------------------|
| [Example: Age] | [Continuous] | [Primary predictor in churn model] |
| | | |
| | | |

### 2.2 Critical Relationships

**Which relationships between variables are essential to preserve?**

[Example: "Correlation between Age and Income must be preserved to ±0.10 for model validity"]

_Your answer:_  
________________________________________________________________________________

---

## SECTION 3: ACCEPTABLE RISK LEVEL

### 3.1 Risk Tolerance Selection

**Select the risk tolerance level for your use case:**

☐ **Conservative (Low Risk)**  
   - High-stakes decisions  
   - Strict regulatory requirements  
   **→ Requires Gold Tier, SDCF-A determination**

☐ **Moderate (Medium Risk)**  
   - Internal use with oversight  
   - Standard regulatory compliance  
   **→ May accept Silver Tier, SDCF-P acceptable with restrictions**

☐ **Aggressive (Higher Risk)**  
   - Testing, development, exploratory use only  
   - Minimal regulatory constraints  
   **→ Bronze Tier acceptable, SDCF-P/R manageable**

### 3.2 Consequences of Failure

**What are the consequences if synthetic data is poor quality?**

| Impact Area | Consequence | Severity (L/M/H) |
|-------------|-------------|------------------|
| Privacy | [Example: Individual re-identification] | [H] |
| Business | [Example: Inaccurate ML predictions] | [ ] |
| Compliance | [Example: GDPR Article 32 violation] | [ ] |
| Reputation | [Example: Loss of customer trust] | [ ] |

---

## SECTION 4: REGULATORY CONSTRAINTS

### 4.1 Applicable Regulations

**Which regulations apply to your use case?**

☐ **GDPR (General Data Protection Regulation)**  
☐ **EU AI Act**  
☐ **HIPAA**  
☐ **Industry-Specific:** [Specify]  
☐ **Other:** [Specify]

---

## SECTION 5: SUCCESS CRITERIA

### 5.1 Minimum Acceptable Outcome

**What is the MINIMUM acceptable outcome?**

☐ **SDCF-A (Acceptable)** - Only fully acceptable data will be used  
☐ **SDCF-P (Provisional)** - Provisional data acceptable with restrictions  
☐ **SDCF-R (Rejected)** - Even rejected data useful for learning/iteration

**If SDCF-P acceptable, what restrictions are tolerable?**  
________________________________________________________________________________

---

## APPROVAL

**This purpose definition is approved for SDCF assessment:**

_______________________________          Date: _______________  
[Defined By - Name and Title]

_______________________________          Date: _______________  
[Approved By - Data Owner]

---

**SDCF Version:** 1.95  
**Framework Author:** Wayne Kearns, Kaionix Labs  
**Website:** https://kaionixlabs.com/sdcf

© 2025 Wayne Kearns, Kaionix Labs. SDCF is licensed under CC BY-NC-SA 4.0.

