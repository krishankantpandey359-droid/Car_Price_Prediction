# Used Car Price Prediction

A machine-learning project that explores used-car listings and compares Linear Regression with Random Forest Regression.

## Dataset

The notebook's recorded run used 4,009 rows and 12 original columns. The data includes make, model, model year, mileage, fuel type, engine, transmission, exterior/interior colour, accident history, clean-title status, and price.

The notebook currently asks the user to upload the CSV in Google Colab. The dataset is not included in this repository, so you must provide your own copy of the source CSV.

## Workflow

1. Load and inspect the uploaded dataset.
2. Review missing values and fill selected categorical fields.
3. Convert price and mileage strings to numeric values.
4. Create a car-age feature.
5. Split the data into training and test sets (80/20, random state 42).
6. Encode categorical columns with OneHotEncoder inside a scikit-learn pipeline.
7. Train Linear Regression and Random Forest Regressor models.
8. Compare MAE, RMSE, and R², then inspect predictions and feature importance.

## Recorded model results

The following values are copied from the notebook's saved output. They are not a guarantee of performance on a new dataset or a new split.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 25,811.27 | 137,184.62 | 0.0793 |
| Random Forest | 19,010.81 | 134,779.84 | 0.1113 |

Random Forest performed better than Linear Regression in this saved run. However, an R² of 0.1113 is weak predictive performance. Treat this as an exploratory baseline, not a reliable car-pricing tool.

## Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- Google Colab / Jupyter Notebook

## Run the notebook

1. Open `CodeAlpha_Car_Price_Prediction.ipynb` in Google Colab or Jupyter.
2. Install the libraries listed above if they are not already available.
3. Upload the used-car CSV when the notebook prompts you.
4. Run the cells from top to bottom.

## Current limitations and next improvements

- The saved evaluation uses one random 80/20 split. Add cross-validation and compare results across folds.
- The target has extreme high-price values. Inspect outliers and consider robust metrics or a log-transformed target, without removing valid luxury-car records blindly.
- Compare against a simple median-price baseline.
- Fit all imputation steps on training data inside the preprocessing pipeline to avoid preprocessing leakage.
- Validate model quality on a genuinely unseen dataset before making practical pricing claims.
- The notebook contains repeated actual-versus-predicted plots that can be consolidated.