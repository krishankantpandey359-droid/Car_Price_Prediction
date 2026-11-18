# Used Car Price Prediction

A reproducible machine-learning project that explores 4,009 used-car listings. It compares a median-price baseline, Linear Regression, Random Forest, Extra Trees, and log-target tree models.

## Dataset

This project uses the public [Used Car Price Prediction Dataset by Taeef Najib on Kaggle](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset).

- File: `used_cars.csv`
- Source shape: 4,009 rows × 12 columns
- Fields include brand, model, model year, mileage, fuel type, engine, transmission, exterior/interior colour, accident history, clean-title status, and price.
- The dataset is downloaded at notebook runtime using KaggleHub. It is not copied into this repository.

## Workflow

1. Download and validate the public dataset.
2. Inspect schema, missing values and numeric distributions.
3. Convert price and mileage strings to numeric values.
4. Create a car-age feature.
5. Split data into training and test sets (80/20, random state 42).
6. Fit imputation and one-hot encoding inside scikit-learn pipelines. Imputers are fitted on each training partition, including each cross-validation fold.
7. Define a median-price baseline, Linear Regression, Random Forest, and Extra Trees candidates, including log-target tree variants.
8. Compare candidates using shuffled five-fold cross-validation on the training split only.
9. Select a candidate using training-only CV MAE, then evaluate it once on the untouched 20% holdout set.
10. Report MAE, RMSE and R² and inspect residuals, predictions and feature importance.

## Saved single-split baseline results

The following values are retained from the notebook's earlier saved 80/20 run. They are a baseline, not a guarantee of performance on a fresh run.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 25,811.27 | 137,184.62 | 0.0793 |
| Random Forest | 19,010.81 | 134,779.84 | 0.1113 |

Random Forest performed better in this saved run, but an R² of 0.1113 indicates weak predictive performance. Do not treat this model as a reliable pricing tool.

## Cross-validation

The notebook runs a five-fold comparison on the training split of:
- Median-price baseline
- Linear Regression
- Random Forest
- Extra Trees
- Random Forest with a log-transformed target
- Extra Trees with a log-transformed target

The workflow prints the fold-averaged MAE, RMSE and R² in the GitHub Actions log. It also uploads the fully executed notebook as the `car-price-executed-notebook` artifact, so the generated result table and charts can be inspected. Cross-validation results are not hard-coded here because they must come from a fresh run. Model selection uses only the training split. The holdout set is not used to choose the model.

The folds are shuffled random splits with a fixed seed and use only the 80% training split. The 20% holdout set is kept out of model selection and used once for final evaluation. These results estimate performance on similar records from the same source, not on a different market, later time period, or independent dataset.

## Data validation tests

The reusable functions in `src/car_data.py` are tested for:
- Required-column and empty-data validation
- Price and mileage parsing
- Removal of invalid, zero and negative target prices
- Source-data immutability
- Invalid input types
- Explicit reference-year handling for car age

## Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- KaggleHub
- Jupyter / Google Colab
- pytest and GitHub Actions

## Run locally

1. Clone or download this repository.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Run the tests:

   `python -m pytest -q`

4. Open `CodeAlpha_Car_Price_Prediction.ipynb` in Jupyter and run all cells. Outside Colab, KaggleHub downloads the public dataset automatically.

In Google Colab, the notebook uses the upload widget. To use a local copy in another environment, set `CAR_DATASET_PATH` to the CSV path before starting Jupyter.

## Automated verification

GitHub Actions:
- Installs the declared dependencies
- Runs unit tests for reusable data validation
- Validates notebook structure
- Downloads the public dataset through KaggleHub
- Executes every notebook code cell
- Prints the cross-validation comparison
- Uploads the executed notebook and results as a workflow artifact

## Limitations and next improvements

- Earlier saved single-split scores are weak and remain historical baseline values only.
- The target contains extreme high-price values. Inspect their validity and compare original-target and log-target models without blindly removing valid luxury-car records.
- Compare candidates using the same folds and preprocessing, then validate the selected model on an independent dataset or later time period.
- Evaluate on a genuinely independent dataset or later time period before making practical pricing claims.
- Remove duplicate actual-versus-predicted visualizations after the final analysis is selected.
