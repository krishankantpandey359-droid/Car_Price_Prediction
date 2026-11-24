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
7. Define a median-price baseline, Linear Regression, Random Forest, Extra Trees, and log-target tree candidates.
8. Compare candidates using shuffled five-fold cross-validation on the training split only.
9. Run a 12-iteration randomized Extra Trees hyperparameter search using only training data and the same folds.
10. Select the candidate using training-only CV MAE, then evaluate it once on the untouched 20% holdout set.
10. Report MAE, RMSE and R² and inspect residuals, predictions and feature importance.

## Saved single-split baseline results

The following values are retained from the notebook's earlier saved 80/20 run. They are a baseline, not a guarantee of performance on a fresh run.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 25,811.27 | 137,184.62 | 0.0793 |
| Random Forest | 19,010.81 | 134,779.84 | 0.1113 |

Random Forest performed better in this saved run, but an R² of 0.1113 indicates weak predictive performance. Do not treat this model as a reliable pricing tool.

## Cross-validation

The notebook compares a median-price baseline, Linear Regression, Random Forest, Extra Trees, and log-target tree variants using five-fold validation on the training split. The 20% holdout is kept separate from model selection. Price-quantile stratification is used for both the initial holdout split and the training-only folds so that low-, middle-, and high-priced listings are represented.

Baseline CI results from the pre-tuning 4,009-row source run:

| Model | CV MAE | CV RMSE | CV R² |
|---|---:|---:|---:|
| Extra Trees | 12,977.97 | 35,724.21 | 0.605 |
| Random Forest | 13,766.35 | 35,919.93 | 0.596 |
| Extra Trees (log target) | 13,369.42 | 37,750.98 | 0.559 |
| Random Forest (log target) | 14,301.07 | 38,991.46 | 0.527 |
| Linear Regression | 19,429.97 | 41,780.55 | 0.449 |
| Median baseline | 25,726.46 | 57,059.37 | -0.052 |

Extra Trees has the lowest training-only cross-validation MAE. Its untouched 20% holdout results are:

- **Holdout MAE:** 18,788.74
- **Holdout RMSE:** 125,581.22
- **Holdout R²:** 0.1202

The large CV-to-holdout gap, especially in RMSE and R², indicates weak generalization and sensitivity to the high-price tail. The model is exploratory and is not reliable enough for real vehicle valuation. A later-period or independent-market test is still required.


### Holdout error by actual-price band

The executed notebook also groups its untouched holdout predictions into five quantile-based actual-price bands. This is a post-hoc diagnostic only; it does not affect training or model selection.

| Actual price band | Listings | Median actual price | MAE | RMSE | WAPE |
|---|---:|---:|---:|---:|---:|
| 2,500–15,000 | 166 | 9,995 | 4,617.00 | 5,916.62 | 47.60% |
| 15,000–25,993.60 | 155 | 19,995 | 6,283.96 | 10,224.71 | 31.23% |
| 25,993.60–37,930 | 160 | 31,250 | 9,411.14 | 13,407.80 | 29.80% |
| 37,930–55,799.20 | 160 | 44,998.50 | 11,367.71 | 23,636.35 | 24.83% |
| 55,799.20–2,954,083 | 161 | 79,999 | 62,133.66 | 278,727.01 | 43.89% |

The highest-price band has much larger absolute errors. It contains a maximum observed holdout price of 2,954,083, so the RMSE is highly sensitive to this upper tail. However, errors also increase across the middle price bands. The issue is therefore broader than one extreme record. The diagnostic does not prove the high-price values are invalid; do not remove them without checking the source listings.

The price bands and the fully executed notebook in this table are from the pre-tuning CI run [44714254582](https://github.com/krishankantpandey359-droid/Car_Price_Prediction/actions/runs/44714254582), commit [`0513944cc8c3bd1bd3849f673dfc18675014ec5c`](https://github.com/krishankantpandey359-droid/Car_Price_Prediction/commit/0513944cc8c3bd1bd3849f673dfc18675014ec5c). The added tuning experiment requires its own separately reported CV and holdout metrics; do not reuse these baseline results as tuning results.

The workflow uploads the fully executed notebook as the `car-price-executed-notebook` artifact, so the generated tables, residual charts and selected-model feature-importance plot can be inspected. The metrics and price-band diagnostics above are documented from that latest successful run.

The folds are shuffled random splits with a fixed seed and use only the 80% training split. This estimates performance on similar records from the same source. It does not establish performance on a different market, later time period, or independent dataset.

The PR #5 tuning experiment reported training-only CV MAE 11,638.73 and untouched holdout MAE 17,500.23, RMSE 125,183.53, and R² 0.1258 in workflow run [45447961434](https://github.com/krishankantpandey359-droid/Car_Price_Prediction/actions/runs/45447961434). These are results from the proposed PR branch, not the current `main` implementation. The low holdout R² confirms that practical pricing reliability remains weak.

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
- Investigate the cross-validation to holdout performance gap, using a chronological or external validation set where available.

## Latest GitHub Actions validation

Latest completed main-branch workflow before this README correction: [GitHub Actions run #45312095547](https://github.com/krishankantpandey359-droid/Car_Price_Prediction/actions/runs/45312095547). A subsequent main-branch documentation workflow is being checked separately; this run verifies only the tests configured in CI.
