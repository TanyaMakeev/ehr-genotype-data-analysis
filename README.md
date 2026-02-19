# Clinical Genomic Analytics Pipeline
## Overview

This project demonstrates a production-style clinical analytics pipeline integrating synthetic Electronic Health Record (EHR) data with genotype data to evaluate genotype–phenotype associations in allergic disease.

The goal is to simulate real-world healthcare data engineering challenges including:

Data ingestion and validation

Referential integrity enforcement

Event-to-patient aggregation

Feature engineering

Statistical modeling

Reproducible clinical analytics workflows

All data are synthetic and HIPAA-safe.

## Clinical Question

Does cumulative genetic risk — or individual SNPs — correlate with allergic urticaria occurrence and severity?

This project simulates a realistic clinical research scenario using structured EHR and genotype data to demonstrate defensible analytics in a healthcare context.

## Cohort Summary

179 patients

794 allergy encounters

6 literature-supported SNPs

Derived Genetic Risk Score (GRS)

## Architecture

The repository follows a modular, layered structure consistent with production data platform design.

```
clinical-genomic-analytics/
├── src/
│ ├── ingestion.py
│ ├── preprocessing.py
│ ├── ehr_integration.py
│ ├── feature_engineering.py
│ ├── genotype_generation.py
│ ├── statistical_modeling.py
│ └── visualization.py
├── notebooks/
│ └── pipeline_demo.ipynb
├── data/
│ └── raw/
└── README.md
```
## Design principle:
Business logic lives in modular Python files.
The notebook orchestrates execution and presents analysis results.

## Data Engineering Capabilities Demonstrated
1. Defensive Data Validation

Required column enforcement

Null checks on key identifiers

Unique patient_id validation

Duplicate detection

Merge integrity safeguards

2. Clinical Event Integration

Left-join strategy preserving all clinical events

Foreign key validation between patient and allergy tables

Prevention of row explosion during merges

Age-at-onset derivation from timestamps (365.25 normalization)

3. Feature Engineering

Derived variables include:

age — age at allergy onset

reaction_burden — total allergy encounters per patient

severity — ordinal phenotype

genetic_risk_score — cumulative risk allele count


4. Statistical Modeling

Logistic regression for urticaria presence

Ordinal regression for severity modeling

Genotype–phenotype association interpretation

Technologies Used

Python

Pandas / NumPy

Scikit-learn

Jupyter

PostgreSQL-ready schema design principles

Modular ETL-style architecture

Example Workflow
```python
# Ingest
patients_df = load_patients("patients.csv")
allergies_df = load_allergies("allergies.csv")

# Validate
assert_no_nulls(patients_df, ["patient_id"])
assert_unique_patient_id(patients_df)

# Preprocess
patients_df = preprocess_patients(patients_df)
allergies_df = preprocess_allergies(allergies_df)

# Integrate
ehr_df = merge_allergies_with_patients(allergies_df, patients_df)

# Feature Engineering
ehr_df = calculate_age_at_onset(ehr_df)
patient_core = build_patient_level_dataset(ehr_df)

# Model
results = run_logistic_regression(patient_core)
```

## Key Findings

All 6 SNPs and genetic risk score were associated with urticaria (logistic model)

All 6 SNPs and genetic risk score associated with increasing severity (ordinal model)

Genetic risk score correlated with reaction burden

## Limitations

Synthetic enrichment assumptions

Moderate cohort size (n=179)

No linkage disequilibrium modeling

No temporal EHR sequence modeling

No covariate adjustment (age, medications, environment)

Results are for workflow demonstration only and are not biological or clinical evidence.

## Why This Project Matters

Clinical analytics requires more than modeling. It requires:

Strict data contracts

Referential integrity awareness

Clear separation of event-level and patient-level data

Reproducible, auditable pipelines

Engineering discipline in healthcare data contexts


This project demonstrates those competencies in a clinically relevant domain.
