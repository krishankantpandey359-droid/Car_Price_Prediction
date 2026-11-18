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
    """Return a cleaned copy; never mutate the caller's source dataframe."""
    validate_raw_cars(frame)
    if not isinstance(reference_year, int) or reference_year < 1900:
        raise ValueError("reference_year must be an integer of at least 1900.")
    cleaned = frame.copy(deep=True).rename(columns={"milage": "mileage"})
    cleaned["price"] = pd.to_numeric(
        cleaned["price"].astype(str).str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False).str.strip(), errors="coerce"
    )
    cleaned["mileage"] = pd.to_numeric(
        cleaned["mileage"].astype(str).str.replace(",", "", regex=False)
        .str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce"
    )
    cleaned["model_year"] = pd.to_numeric(cleaned["model_year"], errors="coerce")
    cleaned = cleaned.dropna(subset=["price", "model_year"]).copy()
    finite_targets = np.isfinite(cleaned["price"]) & np.isfinite(cleaned["model_year"])
    cleaned = cleaned.loc[
        finite_targets & (cleaned["price"] > 0) & (cleaned["model_year"] > 0)
    ].copy()
    cleaned["car_age"] = (reference_year - cleaned["model_year"]).clip(lower=0)
    return cleaned.reset_index(drop=True)
