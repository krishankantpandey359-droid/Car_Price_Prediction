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
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def clean_cars(frame: pd.DataFrame, reference_year: int = 2026) -> pd.DataFrame:
    """Return a cleaned copy, preserving rows with missing predictors for imputation."""
    validate_raw_cars(frame)
    if isinstance(reference_year, bool) or not isinstance(reference_year, int) or reference_year < 1900:
        raise ValueError("reference_year must be an integer of at least 1900.")

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
