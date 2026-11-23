from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NOTEBOOK = Path(__file__).parents[1] / "CodeAlpha_Car_Price_Prediction.ipynb"


def test_notebook_is_valid_and_uses_uploaded_filename():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    upload_cell = next(source for source in sources if "files.upload()" in source)
    read_cell = next(source for source in sources if "pd.read_csv" in source)
    assert "uploaded_filename" in upload_cell
    assert "uploaded_filename" in read_cell
    assert "used_cars.csv" not in read_cell


def test_notebook_uses_shared_cleaning_audit_without_replacing_modeling_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    cleaning_cell = next(source for source in sources if "clean_cars_with_audit" in source)
    assert "cleaning_audit" in cleaning_cell
    assert any("cv_comparison" in source for source in sources)
    assert any("Untouched holdout RMSE" in source for source in sources)


def test_car_price_pipeline_trains_and_returns_finite_metrics():
    frame = pd.DataFrame({
        "brand": ["A", "B", "A", "C", "B", "C", "A", "B", "C", "A", "B", "C"],
        "model": ["x", "y", "z"] * 4,
        "model_year": [2018, 2019, 2020, 2017, 2018, 2021, 2019, 2020, 2016, 2022, 2017, 2021],
        "mileage": [20_000, 30_000, 15_000, 55_000, 42_000, 12_000, 25_000, 18_000, 70_000, 9_000, 60_000, 14_000],
        "fuel_type": ["Gas", "Diesel"] * 6,
        "engine": ["2.0L", "3.0L", "1.5L"] * 4,
        "transmission": ["Auto", "Manual"] * 6,
        "ext_col": ["Black", "White", "Blue"] * 4,
        "int_col": ["Black", "Tan"] * 6,
        "accident": ["None", "Reported"] * 6,
        "clean_title": ["Yes", "No"] * 6,
        "car_age": [8, 7, 6, 9, 8, 5, 7, 6, 10, 4, 9, 5],
    })
    target = pd.Series([12000, 15000, 18000, 9000, 11000, 22000, 14000, 19000, 7000, 26000, 8000, 23000])
    categorical = frame.select_dtypes(include=["object"]).columns.tolist()
    numeric = frame.select_dtypes(exclude=["object"]).columns.tolist()
    preprocess = ColumnTransformer([
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=1)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        frame, target, test_size=0.25, random_state=42
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    metrics = [
        mean_absolute_error(y_test, prediction),
        np.sqrt(mean_squared_error(y_test, prediction)),
        r2_score(y_test, prediction),
    ]
    assert prediction.shape == y_test.shape
    assert np.isfinite(metrics).all()
    assert np.isfinite(prediction).all()
