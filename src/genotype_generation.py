import numpy as np
import pandas as pd

def compute_patient_risk_multiplier(patient_row):
    """
    Calculate patient-specific genetic risk multiplier based on phenotype.
    
    Parameters:
    -----------
    patient_row : pd.Series
        Row containing patient phenotype data
        
    Returns:
    --------
    float
        Risk multiplier to be applied to base MAF
    """
    risk_multiplier = 1.0  # Baseline multiplier

    # Patients with any allergy records have higher chance of risk variants
    if patient_row.get("n_allergy_records", 0) > 0:
        risk_multiplier *= 1.4

    # Urticaria/angioedema is a strong indicator for genetic risk
    # Check for various possible column names
    urticaria_cols = ["reaction_urticaria", "reaction_uticaria", "reaction_urticaria/angioedema"]
    for col in urticaria_cols:
        if col in patient_row and patient_row[col] == 1:
            risk_multiplier *= 2.0
            break

    return risk_multiplier

def simulate_genotype(minor_allele_freq):
    """
    Simulate genotype dosage (0, 1, 2) using Hardy-Weinberg equilibrium.

    Parameters:
    -----------
    minor_allele_freq : float
        Adjusted minor allele frequency for the patient

    Returns:
    --------
    int
        Genotype dosage: 0 (no risk alleles), 1 (heterozygous), 2 (homozygous risk)
    """
    # Clip MAF to valid range to avoid edge cases
    maf = np.clip(minor_allele_freq, 0.001, 0.95)

    # Hardy-Weinberg genotype probabilities
    genotype_probabilities = [
        (1 - maf) ** 2,      # Probability of genotype 0 (no risk alleles)
        2 * maf * (1 - maf),  # Probability of genotype 1 (heterozygous)
        maf ** 2            # Probability of genotype 2 (homozygous risk)
    ]

    # Simulate genotype based on probabilities
  
    genotype = np.random.choice([0, 1, 2], p=genotype_probabilities)

    return genotype

def create_genotype_data(snp_info, patient_phenotypes):
    np.random.seed(42)
    for snp_id, snp_metadata in snp_info.items():
        patient_genotypes = []

        # Generate genotype for each patient based on their phenotype
        for _, patient_row in patient_phenotypes.iterrows():
            # Calculate adjusted MAF based on patient's phenotype burden
            adjusted_maf = (
                snp_metadata["base_maf"]           # Base population MAF
                * patient_row["risk_multiplier"]   # Patient-specific enrichment
                * snp_metadata["effect"]           # SNP effect size
            )

            # Simulate genotype dosage
            genotype = simulate_genotype(adjusted_maf)
            patient_genotypes.append(genotype)

        # Add SNP genotypes to patient dataframe
        patient_phenotypes[snp_id] = patient_genotypes

    # Create final genotype dataframe with patient IDs and SNP genotypes
    genotype_df = patient_phenotypes[["patient_id"] + list(snp_info.keys())].copy()
    return genotype_df

def calculate_genotype_risk(snp_info,genotype_df):
    # Calculate genetic risk score as sum of risk allele dosages across all SNPs
    # This provides a single measure of genetic burden for each patient
    snp_list = list(snp_info.keys())
    
    genotype_df["genetic_risk_score"] = genotype_df[snp_list].sum(axis=1)
    return genotype_df