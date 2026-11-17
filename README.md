# Used Car Price Prediction

A machine-learning project that explores 4,009 used-car listings and compares Linear Regression with Random Forest Regression.

## Dataset

This project uses the public [Used Car Price Prediction Dataset by Taeef Najib on Kaggle](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset).

- File: `used_cars.csv`
- Shape: 4,009 rows × 12 columns
- Fields include brand, model, model year, mileage, fuel type, engine, transmission, exterior/interior colour, accident history, clean-title status, and price.
- The dataset is downloaded at notebook runtime using KaggleHub. It is not copied into this repository.

## Workflow

1. Download and validate the public dataset.
2. Inspect schema, missing values and numeric distributions.
3. Convert price and mileage strings to numeric values.
4. Create a car-age feature.
5. Split data into training and test sets (80/20, random state 42).
6. Fit imputation and one-hot encoding only on training data through scikit-learn pipelines.
7. Train Linear Regression and Random Forest Regressor models.
8. Compare MAE, RMSE and R², then inspect predictions and feature importance.

## Recorded baseline results

These values are from the notebook's previously saved run. They are a baseline, not a guarantee of performance on a fresh run.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 25,811.27 | 137,184.62 | 0.0793 |
| Random Forest | 19,010.81 | 134,779.84 | 0.1113 |

Random Forest performed better in the saved run, but an R² of 0.1113 indicates weak predictive performance. The model should be treated as an exploratory baseline, not a reliable pricing tool.

## Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- KaggleHub
- Jupyter / Google Colab

## Run locally

1. Clone or download this repository.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Open `CodeAlpha_Car_Price_Prediction.ipynb` in Jupyter.
4. Run the notebook. Outside Colab, KaggleHub downloads the public dataset automatically.

In Google Colab, the notebook uses the upload widget. To use a local copy in another environment, set `CAR_DATASET_PATH` to the CSV path before starting Jupyter.

## Automated verification

GitHub Actions validates the notebook structure, installs dependencies, downloads the public dataset through KaggleHub, and executes all notebook code cells.

## Limitations and next improvements

- The saved evaluation uses one random 80/20 split. Add cross-validation and compare results across folds.
- The target contains extreme high-price values. Inspect outliers and consider robust metrics or a log-transformed target without blindly removing valid luxury-car records.
- Compare against a median-price baseline.
- Evaluate models on a genuinely unseen dataset before making practical pricing claims.
- Consolidate repeated actual-versus-predicted plots.
