# Freight Rate Prediction --- Machine Learning Solution

## Overview

This project develops a machine learning model to predict freight rates
(`posted_rate`) from shipment, market, equipment, and time-based
features.

The workflow covers:

1.  Exploratory Data Analysis (EDA)
2.  Data-quality checks and cleaning
3.  Feature engineering
4.  Train/test validation and leakage checks
5.  Missing-value strategy comparison
6.  Linear Regression vs Random Forest model comparison
7.  Final Random Forest training
8.  Feature-importance analysis
9.  Production prediction generation
10. Fixed December 2025 prediction validation using the provided
    `score.py`
11. Model artifact and output generation

The main solution uses a **Random Forest Regressor**. The final model
uses 300 trees, a maximum depth of 18, a minimum leaf size of 2,
`random_state=42`, and all available CPU cores.

The development dataset contains 48,000 labeled loads and the
production/validation dataset contains 12,000 loads. The required
prediction output is `validation_predictions.csv`.

> **Data privacy:** The client-provided datasets are not included in
> this repository. They should remain local and should not be committed
> to GitHub unless the client explicitly authorizes publication.

------------------------------------------------------------------------

## Project Structure

``` text
freight-rate-prediction/
│
├── Final ML Freight Cost Prediction.ipynb
├── requirements.txt
├── README.md
│
├── feature_columns.txt
├──
├── csv/
├── validation_predictions.csv
├── december_chart_inputs_filled (1)
│
├── Freight_Rate_Report_Updated.docx
│
├── exploratory/
├── EDA Freight Cost Prediction.ipynb
├── imputer vs Drop Freight Cost.ipynb
│
├── images/
├── correlation_matrix.png
├── eda_distributions.png
├── equip_time.png
├── feature_importance.png
├── model_comparison.png
├── prediction_distribution.png

```

### Main files

  ------------------------------------------------------------------------
  File                                 Purpose
  ------------------------------------ -----------------------------------
  `Final ML Freight Cost Prediction.ipynb        Main end-to-end ML solution


  `requirements.txt`                             Python package dependencies
  `feature_columns.txt`                          Final list of model features

  `validation_predictions.csv`                   Required production predictions

   `december_chat_inputs_filled.csv`             Amended december chart data

  `Freight_Rate_Report_Updated.docx`             Validation and December prediction report

  `eda_distributions.png`                        Main EDA distributions

  `correlation_matrix.png`                       Correlation analysis

  `feature_importance.png`                       Final model feature importance

  `prediction_distribution.png`                  Training vs production prediction distribution

  ------------------------------------------------------------------------

The saved model artifacts are generated with `joblib` for reuse. The
solution also saves the feature list used by the final model.

------------------------------------------------------------------------

## Data

The solution expects the client-provided files to be available locally:

-   `train-test.csv` --- 48,000 labeled development records
-   `validation.csv` --- 12,000 production records requiring predictions
-   `validation-predictions-template.csv` --- required output template
-   `december_chart_inputs.csv` --- fixed December 2025 scoring inputs

These files are intentionally excluded from the public repository
because they were supplied by the client.

------------------------------------------------------------------------

## Dependencies

The project uses:

-   Python 3
-   pandas
-   numpy
-   matplotlib
-   seaborn
-   scikit-learn
-   joblib

Install the versions specified by the supplied `requirements.txt`:

``` bash
pip install -r requirements.txt
```

The client's supplied dependency constraints include:

``` text
matplotlib>=3.8,<4
numpy>=1.26,<3
pandas>=2.0,<3
```

The machine-learning workflow additionally requires the packages
imported by the solution, including `scikit-learn`, `seaborn`, and
`joblib`.

------------------------------------------------------------------------

## Run Instructions

### Option 1 --- Run the Python script

1.  Clone the repository.
2.  Create and activate a virtual environment.
3.  Install the dependencies.
4.  Place the client-provided datasets in the expected local data
    location.
5.  Run the main solution:

``` bash
python freight_rate_ml_engineer.py
```

The script performs the EDA, preprocessing, train/test validation, model
comparison, final model training, production prediction, quality checks,
and model-artifact saving.

### Option 2 --- Run from Jupyter Notebook

If using Jupyter Notebook:

1.  Create a virtual environment for the project.
2.  Install the requirements.
3.  Start Jupyter Notebook.
4.  Open the main notebook/code workflow.
5.  Make sure the client datasets are in the working directory or update
    the data paths.
6.  Run the cells from top to bottom.

The notebook/script should be run in sequence because later steps depend
on objects created during preprocessing and model training.

------------------------------------------------------------------------

## Machine Learning Workflow

### 1. Exploratory Data Analysis

The EDA examines:

-   Dataset shape and structure
-   Data types
-   Summary statistics
-   Missing values
-   Duplicate records
-   Negative-weight records
-   Target (`posted_rate`) distribution
-   Equipment distribution
-   Correlation with the target
-   Distance vs freight-rate relationship
-   Rate differences by equipment type
-   Seasonal rate behavior

The EDA showed a strong relationship between `distance` and
`posted_rate`, with approximately **0.91 correlation** in the
development data.

The supplied EDA visuals include target, distance, weight, equipment,
distance-vs-rate, and equipment-vs-rate plots, plus the correlation
matrix.

### 2. Data Cleaning

Negative weight values are treated as sign-flip errors and converted to
absolute values.

Missing values are handled with median imputation rather than removing
production records. This is important because the production dataset
requires predictions for all 12,000 validation loads.

### 3. Feature Engineering

Date information is transformed into:

-   `month`
-   `week`
-   `day_of_week`
-   `quarter`

The final model features are:

``` text
distance
weight
market_index
quote_signal
equipment
month
week
day_of_week
quarter
```

The feature list is also stored in `feature_columns.txt`.

### 4. Train/Test Split

The labeled development data is split using:

``` python
train_test_split(
    X,
    y,
    load_ids,
    test_size=0.2,
    random_state=42
)
```

This creates an 80/20 train/test split.

A `load_id` overlap check is performed to verify that no record appears
in both sets. A zero-overlap result confirms that the split does not
contain record-level leakage.

### 5. Leakage Prevention

The median imputer and equipment encoder are fitted using the training
data and then applied to the test/production data.

Conceptually:

``` python
imputer.fit_transform(X_train)
imputer.transform(X_test)
```

and:

``` python
encoder.fit_transform(X_train)
encoder.transform(X_test)
```

This prevents information from the test set from influencing training
preprocessing.

### 6. Missing-Value Strategy

Two approaches were evaluated:

-   Dropping rows containing missing values
-   Median imputation

Median imputation was selected because it preserves more training
information and allows the production pipeline to generate predictions
for all 12,000 validation records.

### 7. Model Comparison

Linear Regression and Random Forest were evaluated using the same
development/test framework.

The supplied model-comparison visualization shows:

-   Linear Regression MAE: **\$144.87**
-   Random Forest MAE: **\$133.96**

Random Forest was selected because it achieved the lower MAE and can
model nonlinear relationships and interactions more effectively.

> The model-comparison chart represents the comparison experiment. The
> final optimized Random Forest is evaluated separately using the final
> holdout metrics reported below.

### 8. Final Random Forest

The final model configuration is:

``` python
RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

The larger number of trees improves ensemble stability, while the depth
and minimum-leaf settings control model complexity and help limit
overfitting.

------------------------------------------------------------------------

## Final Holdout Performance

The final evaluation reports the following holdout-test results:

  Metric     Training           Test
  -------- ---------- --------------
  MAE         \$77.54   **\$123.84**
  RMSE            ---   **\$551.95**
  R²           0.9262     **0.8575**
  MAPE            ---      **5.74%**

The train/test MAE gap is **\$46.30**.

The test set is treated as the primary performance measure because it
contains records not used to fit the model.

------------------------------------------------------------------------

## Feature Importance

Feature importance is used to understand which variables contribute most
to the Random Forest's predictions.

The final model shows that **distance is the dominant feature**, while
weight, market conditions, quote signal, equipment, and time-based
features provide smaller contributions.

The final feature list contains all nine model features; lower
individual importance does not automatically justify removing a feature.

The resulting visualization is saved as:

``` text
feature_importance.png
```

------------------------------------------------------------------------

## Production Predictions

The final pipeline generates predictions for all 12,000
production/validation records.

Quality checks include:

-   Exactly 12,000 rows
-   No missing `predicted_rate`
-   No duplicate `load_id`
-   No negative predicted rates
-   Predictions mapped explicitly to `load_id`

The required output is:

``` text
validation_predictions.csv
```

with exactly:

``` text
load_id,predicted_rate
```

------------------------------------------------------------------------

## December 2025 Fixed-Route Validation

The supplied `score.py` validates the fixed December prediction input
and generates the December chart.

The fixed route is:

  Parameter   Value
  ----------- ----------------------
  Pickup      Lexington
  Delivery    Fort Wayne
  Distance    360 miles
  Equipment   Dry Van
  Weight      32,000 lb
  Dates       December 1--31, 2025

Only the date changes across the 31 December records.

The resulting chart is:

``` text
candidate_december.png
```

The December predictions are approximately:

-   December 1--28: **\$816--\$819**
-   December 29--31: **approximately \$746**

The scoring workflow validates all 31 December predictions and checks
the fixed route parameters and positive prediction values.

------------------------------------------------------------------------

## Generated Visualizations

The repository can contain the following analytical outputs:

### EDA

-   `eda_distributions.png`
-   `correlation_matrix.png`

These show the target distribution, distance and weight distributions,
equipment counts, distance/rate relationship, equipment-rate
relationship, and correlation structure.

### Model Evaluation

-   `feature_importance.png`
-   Model comparison chart
-   Train-vs-test evaluation
-   Prediction distribution comparison

### Production / Scoring

-   `prediction_distribution.png`
-   `candidate_december.png`

These visualizations document the model-selection process, feature
behavior, production predictions, and the required December scoring
result.

------------------------------------------------------------------------

## Reports and Supporting Analysis

The repository may include:

### `Freight_Rate_Report_Updated.docx`

Contains:

-   Train/test validation methodology
-   Data-leakage prevention
-   Holdout-test evaluation
-   Feature importance
-   Production prediction checks
-   December fixed-route prediction results
-   `score.py` validation results

### Preliminary EDA

The complete EDA and missing-value comparison work are supporting
analysis used to justify the final modeling decisions.

These files are not the primary production pipeline; they document the
analysis performed before selecting the final approach.

------------------------------------------------------------------------

## Model Artifacts

The following files are generated for reproducibility and future
inference:

``` text
freight_rate_model.pkl
imputer.pkl
equipment_encoder.pkl
feature_columns.txt
```

They can be reloaded with:

``` python
import joblib

model = joblib.load("freight_rate_model.pkl")
imputer = joblib.load("imputer.pkl")
encoder = joblib.load("equipment_encoder.pkl")
```

------------------------------------------------------------------------

## GitHub / Data Privacy

The client-provided datasets should **not** be committed to GitHub
unless the client has explicitly given permission.

Recommended `.gitignore` entries include:

``` gitignore
# Client-provided data
data/
*.csv

# Local notebooks / validation work
Score_py Validation.ipynb

# Python cache
__pycache__/
*.py[cod]

# Jupyter
.ipynb_checkpoints/

# Virtual environment
.venv/
venv/
env/
```

If `validation_predictions.csv` is required by the client for
submission, keep that file outside the blanket `*.csv` ignore rule or
force-add it intentionally.

For example:

``` gitignore
*.csv
!validation_predictions.csv
```

This keeps the client datasets private while allowing the required
prediction deliverable to be committed.

------------------------------------------------------------------------

## Expected Submission

The repository should provide:

-   Solution code
-   `requirements.txt`
-   Run instructions
-   Required `validation_predictions.csv`
-   Relevant model artifacts
-   Supporting documentation/report
-   Selected EDA and model-validation visualizations

Client-provided raw datasets are excluded unless publication has been
explicitly authorized.

------------------------------------------------------------------------

## Notes

This repository documents the complete modeling workflow rather than
only the final prediction file. The preliminary EDA and imputation/drop
comparison support the final modeling decision, while the main
production workflow is contained in `freight_rate_ml_engineer.py`.

The final model uses the nine documented features and the saved
preprocessing artifacts so that the same feature preparation can be
reproduced consistently for future predictions.
