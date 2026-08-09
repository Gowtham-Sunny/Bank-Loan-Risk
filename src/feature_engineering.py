"""
Feature engineering pipeline for the "Give Me Some Credit" dataset.

This module is a faithful Python port of the R feature-engineering logic used
in the original top-1 (Kaggle) LGBM notebook. It is imported by both the
training notebook and the Streamlit app so that a record scored in the
dashboard is engineered in exactly the same way as the training data.

Design notes
------------
* All "mean" based transforms (subtract-mean / divide-by-mean) must use
  statistics computed on the TRAINING set only. Those stats are fit once
  with `fit_stats` and saved to disk (model/feature_stats.json); at
  inference time we always `transform` with the saved stats, never refit.
* The R script builds every pairwise combination (multiply / subtract /
  divide / add) of the 12 base columns, plus per-column transforms
  (subtract-mean / divide-by-mean / log / square). We reproduce that here.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# The 12 base predictor columns used in the original notebook (after adding
# the two manual features). Order matters only for readability.
BASE_RAW_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

MANUAL_COLUMNS = ["TotalPastDue", "DebtRatio1000"]

TARGET = "SeriousDlqin2yrs"


def add_manual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the two manually engineered columns from the R notebook."""
    df = df.copy()
    df["TotalPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    df["DebtRatio1000"] = np.where(df["MonthlyIncome"] < 1000, np.nan, df["DebtRatio"])
    return df


def fix_age_outlier(df: pd.DataFrame, train_mean_age: float | None = None) -> tuple[pd.DataFrame, float]:
    """Replace age == 0 with the mean age (computed on training data)."""
    df = df.copy()
    if train_mean_age is None:
        train_mean_age = df.loc[df["age"] != 0, "age"].mean()
    df.loc[df["age"] == 0, "age"] = round(train_mean_age)
    return df, train_mean_age


def fit_stats(df: pd.DataFrame, variables: List[str]) -> Dict[str, float]:
    """Compute the per-column means needed for subtract/divide-by-mean transforms."""
    return {var: float(df[var].mean(skipna=True)) for var in variables}


def transform_single_columns(df: pd.DataFrame, variables: List[str], stats: Dict[str, float]) -> pd.DataFrame:
    """subtract-mean / divide-by-mean / log / square transforms for each column."""
    new_cols = {}
    for var in variables:
        mean_val = stats[var]
        new_cols[f"{var}_minus_mean"] = df[var] - mean_val
        if mean_val != 0:
            new_cols[f"{var}_divided_by_mean"] = df[var] / mean_val
        new_cols[f"log_{var}"] = np.where(df[var] == 0, np.nan, np.log(df[var].where(df[var] > 0)))
        new_cols[f"{var}_squared"] = df[var] ** 2
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def combine_column_pairs(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """multiply / subtract / divide / add for every pair of base variables."""
    new_cols = {}
    for var1, var2 in itertools.combinations(variables, 2):
        denom = df[var2]
        new_cols[f"{var1}_times_{var2}"] = df[var1] * df[var2]
        new_cols[f"{var1}_minus_{var2}"] = df[var1] - df[var2]
        new_cols[f"{var1}_divided_by_{var2}"] = np.where(
            denom.isna() | (denom == 0), np.nan, df[var1] / denom
        )
        new_cols[f"{var1}_plus_{var2}"] = df[var1] + df[var2]
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def engineer_features(
    df: pd.DataFrame,
    stats: Dict[str, float],
    train_mean_age: float | None = None,
    is_training: bool = False,
):
    """
    Full pipeline: outlier fix -> manual features -> single-column transforms
    -> pairwise combinations. Returns (engineered_df, train_mean_age_used).
    """
    df, used_mean_age = fix_age_outlier(df, train_mean_age)
    df = add_manual_features(df)

    variables = [c for c in BASE_RAW_COLUMNS + MANUAL_COLUMNS if c in df.columns]

    if is_training:
        stats = fit_stats(df, variables)

    df = transform_single_columns(df, variables, stats)
    df = combine_column_pairs(df, variables)
    return df, stats, used_mean_age


def save_artifacts(
    out_dir: Path,
    stats: Dict[str, float],
    train_mean_age: float,
    selected_features: List[str],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "feature_stats.json", "w") as f:
        json.dump(
            {
                "column_means": stats,
                "train_mean_age": train_mean_age,
                "selected_features": selected_features,
            },
            f,
            indent=2,
        )


def load_artifacts(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    with open(out_dir / "feature_stats.json") as f:
        return json.load(f)
