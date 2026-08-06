# Freight Rate Prediction

## Overview

This repository contains a machine learning solution developed to predict freight rates from shipment characteristics. The workflow includes exploratory data analysis (EDA), data preprocessing, feature engineering, model training, evaluation, model serialization, and production scoring.

The final model is a Random Forest Regressor trained using Scikit-learn. Supporting notebooks are included to demonstrate model development, validation, and generation of prediction outputs required for the assessment.


Assessment Data

> **Note:** The datasets and assessment deliverables are not included in this repository because they were provided by the hiring company as part of the technical assessment.
        This repository contains the complete implementation, dependency list, and instructions required to reproduce the workflow using the provided assessment data.---

## Features

- Data preprocessing and cleaning
- Handling of missing values
- Categorical feature encoding
- Feature selection
- Random Forest Regression model
- Model evaluation using regression metrics
- Feature importance analysis
- Model serialization for future predictions

---

## Project Structure

```
freight-rate-prediction/
│
├── Final ML Freight Cost Prediction.ipynb
├── Score_py Validation.ipynb
├── December Chart.ipynb
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── images/
│   ├── eda_distributions.png
│   ├── correlation_matrix.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   └── prediction_distribution.png
```

---

## Technologies Used

- Python 3.x
- Pandas >=2.0,<3
- NumPy =1.26,<3
- Scikit-learn
- Matplotlib >=3.8,<4
- Joblib

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/freight-rate-prediction.git
cd freight-rate-prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset used for this project is **not included** in this repository because it was provided by the hiring company as part of a technical assessment.

To run the project, place the provided dataset in the project directory (or update the dataset path in the source code as needed).

---

## Running the Project

Execute the main script:

```bash
python Final ML Freight Cost Prediction.ipynb
```

The script performs:

1. Data loading
2. Data preprocessing
3. Feature engineering
4. Model training
5. Model evaluation
6. Feature importance generation
7. Model serialization (if enabled)

---

## Model

The current implementation uses a **Random Forest Regressor** with tuned hyperparameters for freight rate prediction.

Example configuration:

```python
RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

---

## Output

Depending on the configuration, the project may generate:

- Performance metrics
- Feature importance visualization
- Serialized model (`.pkl`)
- Prediction results

---

## Notes

This repository contains only the implementation and supporting files.

No proprietary or assessment-provided datasets are included in order to respect the ownership of the assessment materials.

---

## Author

**Margrethe Fayne Gonzales**

Data Analyst | Data Scientist | Machine Learning Enthusiast

GitHub: https://github.com/<margrethefaynegonzales>
LinkedIn: https://www.linkedin.com/in/<https://www.linkedin.com/in/margrethe-fayne-gonzales/>

---
