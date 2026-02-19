import pandas as pd

# Function to convert columns in dataframe to lowercase
def lowercase_columns(df):
    df.columns = [col.lower() for col in df.columns]
    return df

# Function to remove leading and trailing whitespace from dataframe
def remove_whitespace(df):
    df = df.copy()

    string_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()

    return df

def normalize_clinical_tables(allergies_df, patients_df):

    allergies_df = allergies_df.copy()
    patients_df = patients_df.copy()

    # Set active status based on stop column
    allergies_df["active"] = allergies_df["stop"].isna().replace({True: "yes", False: "no"})
    allergies_df = allergies_df.drop(columns=["stop"])

    # Standardize patient ID naming
    allergies_df = allergies_df.rename(columns={"patient": "patient_id"})
    patients_df = patients_df.rename(columns={"id": "patient_id"})

    # Normalize race formatting (already lowercase column names)
    if "race" in patients_df.columns:
        patients_df["race"] = (
            patients_df["race"]
            .str.lower()
            .str.replace(" ", "_")
        )

    return allergies_df, patients_df