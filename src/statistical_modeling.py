# Data manipulation and analysis
import pandas as pd
import numpy as np

# Statistical analysis
from scipy.stats import fisher_exact, chi2_contingency, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel

def chi_square_test(df, col1, col2):
    """
    Perform chi-square test of independence between two categorical variables.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data
    col1 (str): Name of the first categorical column
    col2 (str): Name of the second categorical column
    
    Returns:
    tuple: (chi2_statistic, p_value, dof, expected_frequencies)
    """
    # Create contingency table
    contingency_table = pd.crosstab(df[col1], df[col2])
    
    # Perform chi-square test if table has sufficient dimensions (at least 2x2)
    if contingency_table.shape[0] >= 2 and contingency_table.shape[1] == 2:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        return chi2, p_value, dof, expected
    else:
        return None, None, None, None

def logistic_regression(df, feature_cols, target_col):
    """
    Perform logistic regression to model the relationship between features and a binary target variable.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data
    target_col (str): Name of the binary target column
    feature_cols (list): List of feature column names
    
    Returns:
    statsmodels.regression.linear_model.RegressionResults: The fitted logistic regression model
    """
    # Prepare data for logistic regression
    predictor_variables = sm.add_constant(df[feature_cols])
    outcome_variable = df[target_col]

    # Fit logistic regression model
    logistic_model = sm.Logit(outcome_variable, predictor_variables).fit(disp=False)

    # Extract and report results
    odds_ratio = np.exp(logistic_model.params[feature_cols])
    p_value = logistic_model.pvalues[feature_cols]
    confidence_interval = np.exp(logistic_model.conf_int().loc[feature_cols])
    
    return odds_ratio, p_value, confidence_interval

def genetic_risk_score_logistic(df, feature_cols, target_col):
    """
    Perform logistic regression to model the relationship between genetic risk score and a binary target variable.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data
    target_col (str): Name of the binary target column
    feature_cols (list): List of feature column names
    
    Returns:
    statsmodels.regression.linear_model.RegressionResults: The fitted logistic regression model
    """
    # Prepare data for logistic regression
    predictor_variables = sm.add_constant(df[feature_cols])
    outcome_variable = df[target_col]

    # Fit logistic regression model
    logistic_model = sm.Logit(outcome_variable, predictor_variables).fit(disp=False)

    # Extract and report results
    odds_ratio = np.exp(logistic_model.params[feature_cols])
    p_value = logistic_model.pvalues[feature_cols]
    confidence_interval = np.exp(logistic_model.conf_int().loc[feature_cols])
    
    return odds_ratio, p_value, confidence_interval

def ordinal_regression(df, feature_cols, target_col):
    """
    Perform ordinal regression to model the relationship between features and an ordinal target variable.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data
    target_col (str): Name of the ordinal target column
    feature_cols (list): List of feature column names
    
    Returns:
    statsmodels.regression.linear_model.RegressionResults: The fitted ordinal regression model
    """
    # Prepare data for ordinal regression (remove missing severity values)
    severity_data = df[[feature_cols, target_col]].dropna()

    if len(severity_data) < 10:  # Ensure sufficient sample size
        print(f"SNP: {feature_cols} - Insufficient data for ordinal regression")
        return None

    # Fit ordinal logistic regression model
    ordinal_model = OrderedModel(
        severity_data[target_col],
        severity_data[[feature_cols]],
        distr='logit'
    )

    try:
        ordinal_result = ordinal_model.fit(method='bfgs', disp=False)
        return ordinal_result
    except Exception as e:
        print(f"SNP: {feature_cols} - Error fitting ordinal regression: {e}")
        return None

def spearman_correlation_burden(df):
    """
    Calculate Spearman rank correlation between features and target variable.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the data
    
    Returns:
    pd.Series: Spearman correlation coefficients
    """
    all_reaction_columns = [
    "reaction_anaphylaxis",
    "reaction_gi_symptoms",
    "reaction_lower_respiratory",
    "reaction_skin_problems",
    "reaction_systemic",
    "reaction_upper_respiratory",
    "reaction_urticaria",
]

    # Calculate reaction burden as sum of all reaction types per patient
    df["reaction_burden"] = df[all_reaction_columns].sum(axis=1)
    # Calculate Spearman correlation
    correlation_coefficient, p_value = spearmanr(
        df["genetic_risk_score"],
        df["reaction_burden"]
    )
    
    return correlation_coefficient, p_value
