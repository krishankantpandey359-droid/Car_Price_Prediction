"""Reusable validation and cleaning helpers for the used-car dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "brand", "model", "model_year", "milage", "fuel_type", "engine",
    "transmission", "ext_col", "int_col", "accident", "clean_title", "price",
}


def validate_raw_cars(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if frame.empty:
        raise ValueError("The car dataset is empty.")
    if frame.columns.has_duplicates:
        duplicates = frame.columns[frame.columns.duplicated()].unique().tolist()
        raise ValueError(f"Car dataset contains duplicate column names: {duplicates}")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def clean_cars(frame: pd.DataFrame, reference_year: int = 2026) -> pd.DataFrame:
    """Return a cleaned copy, preserving rows with missing predictors for imputation."""
    validate_raw_cars(frame)
    if (
        isinstance(reference_year, (bool, np.bool_))
        or not isinstance(reference_year, (int, np.integer))
        or reference_year < 1900
    ):
        raise ValueError("reference_year must be an integer of at least 1900.")
    reference_year = int(reference_year)

    cleaned = frame.copy(deep=True).rename(columns={"milage": "mileage"})
    cleaned["price"] = pd.to_numeric(
        cleaned["price"].astype(str).str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False).str.strip(), errors="coerce"
    )
    cleaned["mileage"] = pd.to_numeric(
        cleaned["mileage"].astype(str).str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False), errors="coerce"
    )
    cleaned["model_year"] = pd.to_numeric(cleaned["model_year"], errors="coerce")

    valid_price = np.isfinite(cleaned["price"]) & (cleaned["price"] > 0)
    valid_year = (
        np.isfinite(cleaned["model_year"])
        & (cleaned["model_year"] >= 1886)
        & (cleaned["model_year"] <= reference_year)
        & (cleaned["model_year"] % 1 == 0)
    )
    cleaned = cleaned.loc[valid_price & valid_year].copy()

    # Missing mileage remains missing for train-only median imputation.
    # Impossible negative or non-finite values are treated as missing predictors.
    mileage_finite = np.isfinite(cleaned["mileage"])
    cleaned.loc[~mileage_finite | (cleaned["mileage"] < 0), "mileage"] = np.nan

    cleaned["car_age"] = reference_year - cleaned["model_year"]
    return cleaned.reset_index(drop=True)


def clean_cars_with_audit(
    frame: pd.DataFrame, reference_year: int = 2026
) -> tuple[pd.DataFrame, dict]:
    """Clean car records and return transparent row-retention diagnostics."""
    validate_raw_cars(frame)
    cleaned = clean_cars(frame, reference_year=reference_year)

    parsed_price = pd.to_numeric(
        frame["price"].astype(str).str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    parsed_year = pd.to_numeric(frame["model_year"], errors="coerce")
    valid_target = (
        np.isfinite(parsed_price)
        & (parsed_price > 0)
        & np.isfinite(parsed_year)
        & parsed_year.between(1886, reference_year)
        & (parsed_year % 1 == 0)
    )

    raw_mileage = frame["milage"]
    mileage_text = raw_mileage.astype("string").str.strip()
    missing_mileage = raw_mileage.isna() | mileage_text.eq("")
    parsed_mileage = pd.to_numeric(
        mileage_text.str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )
    invalid_mileage = (
        ~missing_mileage
        & (
            parsed_mileage.isna()
            | ~np.isfinite(parsed_mileage)
            | (parsed_mileage < 0)
        )
    )
    flagged_mileage = missing_mileage | invalid_mileage
    duplicate_source_rows = int(frame.duplicated().sum())

    audit = {
        "input_rows": int(len(frame)),
        "retained_rows": int(len(cleaned)),
        "dropped_invalid_target_rows": int((~valid_target).sum()),
        "retained_rows_with_missing_or_invalid_mileage": int(
            (valid_target & flagged_mileage).sum()
        ),
        "retained_rows_with_missing_mileage": int(
            (valid_target & missing_mileage).sum()
        ),
        "retained_rows_with_invalid_mileage": int(
            (valid_target & invalid_mileage).sum()
        ),
        "duplicate_source_rows": duplicate_source_rows,
        "reference_year": int(reference_year),
        "row_retention_percent": round(
            len(cleaned) / len(frame) * 100, 2
        ) if len(frame) else 0.0,
    }
    return cleaned, audit
