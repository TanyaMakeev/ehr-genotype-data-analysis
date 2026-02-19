# Open EHR files

import pandas as pd 
from typing import List


# Function to load EHR files
def load_ehr(ehr_file, usecols=None):
    ehr_file_df = pd.read_csv(ehr_file, usecols=usecols)
    return ehr_file_df

def validate_no_nulls(
    df: pd.DataFrame,
    required_columns: List[str],
    table_name: str = "DataFrame"
) -> None:
    """
    Validates that required columns contain no null values.
    Raises ValueError if nulls are found.
    """

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{table_name} is missing required columns: {missing_cols}"
        )

    null_counts = df[required_columns].isnull().sum()

    columns_with_nulls = null_counts[null_counts > 0]

    if not columns_with_nulls.empty:
        raise ValueError(
            f"Null values detected in {table_name}:\n"
            f"{columns_with_nulls}"
        )

    print(f"✔ {table_name}: No nulls in required columns.")

def validate_unique_identifier(
    df: pd.DataFrame,
    id_column: str,
    table_name: str = "DataFrame"
) -> None:
    """
    Ensures identifier column contains unique values.
    """

    if id_column not in df.columns:
        raise ValueError(
            f"{id_column} not found in {table_name}"
        )

    duplicates = df[id_column].duplicated().sum()

    if duplicates > 0:
        raise ValueError(
            f"{table_name} contains {duplicates} duplicate {id_column} values."
        )

    print(f"✔ {table_name}: {id_column} is unique.")

def validate_foreign_key(
    child_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    key_column: str,
    child_name: str = "Child Table",
    parent_name: str = "Parent Table"
) -> None:
    """
    Ensures all child keys exist in parent table.
    """

    missing_keys = set(child_df[key_column]) - set(parent_df[key_column])

    if missing_keys:
        raise ValueError(
            f"{child_name} contains {len(missing_keys)} {key_column} values "
            f"not found in {parent_name}."
        )

    print(f"✔ All {key_column} values in {child_name} exist in {parent_name}.")