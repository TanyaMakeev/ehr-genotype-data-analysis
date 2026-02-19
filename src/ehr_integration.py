# src/ehr_integration.py

import pandas as pd
from typing import Union, List


def merge_dataframes(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    merge_on: Union[str, List[str]],
    columns_to_add: List[str] = None,
    how: str = "left"
    ) -> pd.DataFrame:
    """
    Merges two DataFrames with optional join validation.

    Parameters:
    - left_df: Left DataFrame
    - right_df: Right DataFrame
    - merge_on: Column(s) to merge on
    - columns_to_add: Subset of right_df columns to include
    - how: Merge type ('left', 'right', 'inner', 'outer')
    - validate: Expected join type:
        'one_to_one'
        'one_to_many'
        'many_to_one'
        'many_to_many'
    """

    # Select subset if specified
    if columns_to_add is not None:
        merged_df = left_df.merge(
        right_df[columns_to_add],
        on=merge_on,
        how=how
        #,validate=validate  # pandas built-in validation
        )
    else:
        merged_df = left_df.merge(
        right_df,
        on=merge_on,
        how=how
        #,validate=validate  # pandas built-in validation
    )

    return merged_df


def convert_dates(ehr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts relevant date columns to datetime.
    """

    ehr_df = ehr_df.copy()

    ehr_df["start"] = pd.to_datetime(ehr_df["start"], errors="coerce")
    ehr_df["birthdate"] = pd.to_datetime(ehr_df["birthdate"], errors="coerce")

    return ehr_df


def calculate_age_at_onset(ehr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates patient age (years) at allergy onset.
    Handles missing or invalid dates safely.
    """

    ehr_df = ehr_df.copy()

    age_years = (
        (ehr_df["start"] - ehr_df["birthdate"])
        .dt.days
        .div(365.25)
    )

    # Convert to numeric safely
    ehr_df["age"] = (
        pd.to_numeric(age_years, errors="coerce")
        .round()
        .astype("Int64")  # nullable integer
    )

    return ehr_df



def rename_columns(ehr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns for clarity.
    """

    ehr_df = ehr_df.copy()
    ehr_df = ehr_df.rename(columns={"description": "allergen"})

    return ehr_df
