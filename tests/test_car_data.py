import pandas as pd
import pytest

from src.car_data import clean_cars, validate_raw_cars


@pytest.fixture
def raw_cars():
    return pd.DataFrame(
        {
            "brand": ["Toyota", "Honda", "Ford"],
            "model": ["Camry", "Civic", "Focus"],
            "model_year": [2020, 2018, 2016],
            "milage": ["25,000 mi", "40,000 mi", "18,500 mi"],
            "fuel_type": ["Gasoline", "Gasoline", "Diesel"],
            "engine": ["2.5L", "2.0L", "1.5L"],
            "transmission": ["Automatic", "Manual", "Automatic"],
            "ext_col": ["White", "Black", "Blue"],
            "int_col": ["Black", "Gray", "Black"],
            "accident": ["None reported", "None reported", "At least 1 accident"],
            "clean_title": ["Yes", "Yes", "Yes"],
            "price": ["$20,000", "$15,500", "invalid"],
        }
    )


def test_clean_cars_parses_numeric_fields_and_drops_invalid_targets(raw_cars):
    cleaned = clean_cars(raw_cars)
    assert len(cleaned) == 2
    assert cleaned["price"].tolist() == [20000, 15500]
    assert cleaned["mileage"].tolist() == [25000, 40000]
    assert cleaned["car_age"].tolist() == [6, 8]
    assert cleaned["price"].gt(0).all()


def test_clean_cars_does_not_mutate_source(raw_cars):
    original = raw_cars.copy(deep=True)
    clean_cars(raw_cars)
    pd.testing.assert_frame_equal(raw_cars, original)


def test_validate_raw_cars_rejects_missing_columns(raw_cars):
    with pytest.raises(ValueError, match="price"):
        validate_raw_cars(raw_cars.drop(columns=["price"]))


def test_validate_raw_cars_rejects_empty_dataframe(raw_cars):
    with pytest.raises(ValueError, match="empty"):
        validate_raw_cars(raw_cars.iloc[0:0])


def test_clean_cars_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        clean_cars([{"price": 1000}])


def test_clean_cars_drops_zero_and_negative_prices(raw_cars):
    invalid_targets = raw_cars.copy()
    invalid_targets["price"] = ["$0", "$-100", "$0"]
    assert clean_cars(invalid_targets).empty


def test_clean_cars_accepts_explicit_reference_year(raw_cars):
    cleaned = clean_cars(raw_cars, reference_year=2025)
    assert cleaned["car_age"].tolist() == [5, 7]


def test_clean_cars_drops_non_finite_price_and_model_year(raw_cars):
    raw_cars["price"] = raw_cars["price"].astype(object)
    raw_cars["model_year"] = raw_cars["model_year"].astype(float)
    raw_cars.loc[0, "price"] = float("inf")
    raw_cars.loc[1, "model_year"] = float("-inf")
    cleaned = clean_cars(raw_cars)
    assert cleaned.empty


def test_clean_cars_marks_negative_and_non_finite_mileage_missing(raw_cars):
    raw_cars.loc[0, "milage"] = "-120 mi"
    raw_cars.loc[1, "milage"] = "inf"
    cleaned = clean_cars(raw_cars)
    assert cleaned["mileage"].isna().all()


def test_clean_cars_rejects_fractional_model_years(raw_cars):
    raw_cars["model_year"] = raw_cars["model_year"].astype(float)
    raw_cars.loc[0, "model_year"] = 2020.5
    cleaned = clean_cars(raw_cars)
    assert cleaned["model_year"].tolist() == [2018.0]


def test_clean_cars_rejects_boolean_reference_year(raw_cars):
    with pytest.raises(ValueError, match="reference_year"):
        clean_cars(raw_cars, reference_year=True)
