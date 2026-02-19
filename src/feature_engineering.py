import pandas as pd

def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms wide-formatted data to long-formatted data.
    """
    # Transform wide format to long format for reaction analysis
    df_long = pd.wide_to_long(
        df,
        stubnames=['reaction', 'description', 'severity'],
        i=['start', 'patient_id', 'encounter', 'code', 'system', 'allergen', 'type', 'category'],
        j='record_id'
    ).reset_index()

    # Remove unnecessary columns that won't be used in analysis
    df_long = df_long.drop(columns=['code', 'system'])

    # Remove duplicate reaction records to ensure data quality
    df_long = df_long.drop_duplicates(
        subset=["patient_id", "encounter", "allergen", "reaction", "severity"]
    )

    return df_long

def map_category(df, column_name, new_column_name, mapping_dict):
    """
    Maps category values to a new column.
    Supports both simple key-value mappings and list-based mappings.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column_name : str
        Column to map from
    new_column_name : str
        New column to create with mapped values
    mapping_dict : dict
        Dictionary where values can be either single values or lists of values
    """
    df = df.copy(deep=True)
          
    def map_value(value):
        # Handle NaN values
        if pd.isna(value):
            return None
            
        # Convert to string for consistent comparison
        value_str = str(value).strip()
        
        # Direct mapping
        if value_str in mapping_dict:
            return mapping_dict[value_str]
        
        # Case-insensitive mapping
        for key, mapped_value in mapping_dict.items():
            if isinstance(mapped_value, list):
                # Check if value matches any in the list
                if value_str in [str(v).strip() for v in mapped_value]:
                    return key
            else:
                # Single value mapping
                if value_str.lower() == str(mapped_value).lower():
                    return key
                
        return None  # Return None for unmapped values
    
    df[new_column_name] = df[column_name].apply(map_value)
    
    return df

def create_patient_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create patient_core dataframe with one record per patient
    This aggregates longitudinal allergy data into stable patient-level phenotypes
    """
    # Check what severity columns exist
    severity_cols = [col for col in df.columns if 'severity' in col.lower()]
    print(f"Available severity columns: {severity_cols}")
    
    # Show sample of severity data
    if severity_cols:
        for col in severity_cols:
            print(f"\nColumn '{col}' sample values:")
            print(df[col].value_counts(dropna=False))
            print(f"Unique values: {df[col].unique()}")
            print(f"Data type: {df[col].dtype}")
    
    # Use severity_numeric if available, otherwise fall back to other severity columns
    if 'severity_numeric' in df.columns:
        severity_col = 'severity_numeric'
        print(f"\nUsing severity_numeric column")
    else:
        # Use first severity column found, or 'severity' as default
        severity_col = 'severity' if 'severity' in df.columns else severity_cols[0] if severity_cols else None
        
        if severity_col is None:
            print("No severity column found, setting max_severity to -1 for all patients")
            df['max_severity'] = -1
            severity_col = 'max_severity'
        else:
            print(f"\nUsing severity column: {severity_col}")
            
            # Convert string severity to numeric if needed
            if severity_col == 'severity' and df[severity_col].dtype == 'object':
                print("Converting string severity values to numeric...")
                severity_mapping = {
                    'MILD': 1,
                    'MODERATE': 2, 
                    'SEVERE': 3
                }
                df['severity_numeric_converted'] = df[severity_col].map(severity_mapping)
                severity_col = 'severity_numeric_converted'
                print(f"Converted severity values: {df['severity_numeric_converted'].value_counts(dropna=False)}")
    
    # Check if this column has any non-NaN values
    non_null_count = df[severity_col].notna().sum()
    print(f"Non-null values in {severity_col}: {non_null_count}")
    if non_null_count == 0:
        print(f"WARNING: Column {severity_col} has no non-null values!")
    
    patient_core = (
    df
    .groupby("patient_id", as_index=False)
    .agg(
        n_allergy_records=("record_id", "count"),           # Total allergy episodes
        n_unique_allergens=("allergen", "nunique"),        # Number of different allergens
        max_severity=(severity_col, "max"),                # Worst reaction severity
        race=("race", "first"),                            # Demographic info
        gender=("gender", "first"),                        # Demographic info
        age_at_start=("age", "first")             # Age at first allergy
    )
    )
    # Handle missing values appropriately for each column type
    
    # Fill count/aggregation columns with 0 (no records = 0)
    count_cols = ["n_allergy_records", "n_unique_allergens", "age_at_start"]
    patient_core[count_cols] = patient_core[count_cols].fillna(0).astype(int)
    
    # For severity, fill with -1 to indicate no severity data recorded
    # This distinguishes missing data from actual severity level 0
    patient_core["max_severity"] = patient_core["max_severity"].fillna(-1)
    
    # For demographics, keep original values (don't fill missing demographics)
    # race and gender should remain NaN if missing

    return patient_core

def create_one_hot_encoding(init_df: pd.DataFrame, final_df: pd.DataFrame, column_name: str, prefix: str) -> pd.DataFrame:
    """
    Creates one-hot encoded features for categorical variables.
    Aggregates to patient level (one row per patient).
    """
    init_df = init_df.copy(deep=True)
    dummies = pd.get_dummies(init_df[column_name], prefix = prefix).astype(int)
    # Aggregate to patient level: if patient has any record of allergen type, set flag to 1
    flags = (
        pd.concat([init_df['patient_id'], dummies], axis=1)
        .groupby('patient_id', as_index=False)
        .max()  # Use max() to get 1 if patient has any record of that allergen type
    )
    final_df = final_df.merge(flags, on="patient_id", how="left")
    return final_df