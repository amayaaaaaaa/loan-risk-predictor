# Loan Risk Prediction

## 1. Project Overview

This project uses machine learning to estimate whether a loan applicant may experience repayment difficulties. It is based on the **Home Credit Default Risk** dataset and uses applicant information together with aggregated historical credit, previous-application, installment-payment, and credit-card records.

The target column is `TARGET`:

- `0` — no repayment difficulty recorded
- `1` — repayment difficulty recorded

This is an educational machine-learning project. Its predictions should not be used as the sole basis for real lending decisions.

## 2. Problem Statement

Financial institutions need to assess the risk that an applicant may not repay a loan. The goal of this project is to build and evaluate classification models that identify applicants who may have repayment difficulties, while examining precision, recall, F1-score, confusion matrices, and model explanations.

## 3. Dataset Source

**Dataset:** Home Credit Default Risk  
**Source:** Kaggle — https://www.kaggle.com/competitions/home-credit-default-risk/data

Download the data from Kaggle and place the required CSV files in a folder named `data/` beside the notebook.

The notebook uses these files:

```text
data/
├── application_train.csv
├── application_test.csv
├── bureau.csv
├── previous_application.csv
├── installments_payments.csv
└── credit_card_balance.csv
```

The training application file contains the target label. The test application file does not contain `TARGET`; it is used to generate predictions. The other files contain historical records that are aggregated by `SK_ID_CURR` and merged with the application data.

Do not upload large raw datasets to GitHub unless you have permission and the repository limits allow it. Include instructions for obtaining the data instead.

## 4. Project Structure

```text
Loan-Risk-Prediction/
├── Loan_Risk_Prediction (2).ipynb
├── README.md
├── requirements.txt
└── data/
    ├── application_train.csv
    ├── application_test.csv
    ├── bureau.csv
    ├── previous_application.csv
    ├── installments_payments.csv
    └── credit_card_balance.csv
```

The `data/` directory is expected locally and may be excluded from version control.

## 5. Technologies and Libraries

- Python
- Jupyter Notebook
- pandas and NumPy for data manipulation
- Matplotlib and Seaborn for visualization
- scikit-learn for preprocessing, model training, evaluation, and hyperparameter tuning
- SHAP and LIME for model explainability
- joblib for saving the trained model

## 6. Installation and Setup

Use Python 3.10 or a compatible Python version with the libraries listed below.

1. Download or clone this repository.
2. Download the required CSV files from the Kaggle dataset page.
3. Put the CSV files in the `data/` folder shown above.
4. Create and activate a virtual environment if desired.
5. Install the dependencies:

```bash
pip install -r requirements.txt
```

6. Start Jupyter:

```bash
jupyter notebook
```

7. Open `Loan_Risk_Prediction (2).ipynb` and run the cells from top to bottom.

The notebook contains `%pip install shap` and `%pip install lime` cells. If the environment already has these packages installed, those cells can be skipped. Restart the notebook kernel if an installation requires it.

## 7. Workflow

### Data understanding and exploratory analysis

- Load the application training data.
- Inspect its dimensions, data types, descriptive statistics, missing values, and duplicate records.
- Examine selected financial and demographic variables.
- Identify potential numerical outliers using the interquartile range (IQR) method.
- Visualize the target distribution and relationships between selected variables.

### Preprocessing

The notebook builds scikit-learn preprocessing pipelines:

- Numerical missing values are filled using the median.
- Numerical features are standardized where the pipeline uses `StandardScaler`.
- Categorical missing values are filled using the most frequent value.
- Categorical variables are one-hot encoded using `OneHotEncoder(handle_unknown="ignore")`.
- Data is split into training and test subsets using an 80/20 split, `random_state=42`, and stratification by `TARGET`.

### Feature engineering

Historical records are aggregated by applicant ID (`SK_ID_CURR`) and merged into the application dataset. Examples include:

- **Bureau:** number of loans, active loans, total credit, debt, overdue amount, and maximum overdue days.
- **Previous applications:** application count, approved/refused counts, and credit/application amount summaries.
- **Installment payments:** late-payment days/count and payment differences.
- **Credit-card balances:** record count, average balance/limit/drawings, and maximum days past due.

The same feature-engineering operations are applied to the test application data before prediction.

### Models evaluated

The notebook evaluates classification models including:

- Logistic Regression baseline
- Logistic Regression with historical features
- Random Forest classifier
- Random Forest with engineered historical features
- Tuned Random Forest models

Class weighting is used in several models to help address the imbalanced target classes.

### Hyperparameter tuning

The notebook uses `RandomizedSearchCV` and `GridSearchCV` with cross-validation to tune Random Forest parameters such as:

- Number of trees (`n_estimators`)
- Maximum tree depth (`max_depth`)
- Minimum samples required to split a node (`min_samples_split`)
- Minimum samples per leaf (`min_samples_leaf`)
- Number of features considered at each split (`max_features`)

The best model is selected based on cross-validated F1-score in the search configuration. The notebook also compares several probability thresholds.

### Evaluation

The notebook reports metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

Because the target classes are imbalanced, accuracy alone is not sufficient. Review class-1 precision, recall, and F1-score when comparing models.

### Explainability

- **SHAP** is used to examine global feature importance and an individual prediction.
- **LIME** is used to explain one applicant prediction locally.

Explanations describe how the model behaves; they do not establish that a feature causes repayment difficulty.

## 8. Final Results

Run the notebook from beginning to end to generate the current evaluation results. The final metrics and best hyperparameters are printed in the notebook, and SHAP/LIME visualizations are displayed during execution.

**Insert your actual results here after running the notebook:**

| Metric | Result |
|---|---|
| Final model | Best estimator selected by GridSearchCV |
| Accuracy | Add value from notebook output |
| Macro F1-score | Add value from notebook output |
| Class-1 precision | Add value from classification report |
| Class-1 recall | Add value from classification report |
| Class-1 F1-score | Add value from classification report |
| Best cross-validation F1-score | Add value from GridSearchCV output |

Do not enter estimated or invented values; copy the results from your completed run.

## 9. Predicting on the Test Dataset

The notebook loads `application_test.csv`, merges the engineered historical features, aligns its feature columns with the training data, and predicts:

- `TARGET`: predicted class (`0` or `1`)
- `DEFAULT_PROBABILITY`: estimated probability of class `1`

The predictions are stored in a pandas DataFrame named `final_results`. To save them as a CSV, add and run:

```python
final_results.to_csv("loan_risk_predictions.csv", index=False)
```

## 10. Saved Model

The notebook saves the selected model pipeline to:

```text
HomeCreditModel.joblib
```

using `joblib.dump(final_model, "HomeCreditModel.joblib")`. The saved pipeline includes preprocessing and the fitted estimator. Load it in a compatible Python environment with:

```python
import joblib

model = joblib.load("HomeCreditModel.joblib")
```

To make predictions, the input data must contain the same feature columns and undergo the same historical-feature engineering used during training. The saved model alone does not automatically perform the CSV loading and historical aggregation steps.

Only load joblib files from trusted sources because loading a serialized model can execute code.

## 11. Batch-Inference Instructions

After running the notebook and creating `final_results`, save all predictions in one batch:

```python
final_results.to_csv("loan_risk_predictions.csv", index=False)
```

The resulting CSV contains the applicant ID, predicted class, and estimated default probability.

## 12. Local Application/API and Deployment

This notebook currently demonstrates model development and batch predictions. It does **not** implement a web application, REST API, or live deployment.

If deployment is required, a separate interface/API must be created, tested, and deployed. Add the verified live URL here once deployment is complete:

**Deployment URL:** Not deployed yet.

## 13. Reproducibility

- The train/test split uses `random_state=42`.
- The model configurations and cross-validation settings are defined in the notebook.
- Run notebook cells in order because later cells depend on earlier variables and engineered datasets.
- Use the same dataset files and compatible library versions to reproduce results.
- Use relative paths such as `data/application_train.csv`, rather than machine-specific absolute paths.
- Keep preprocessing and inference feature columns consistent.
- The notebook does not currently use MLflow experiment tracking.

## 14. Limitations and Future Improvements

- The dataset may not represent all current applicants or lending environments.
- The target is imbalanced, so model metrics should be interpreted carefully.
- Historical data may be incomplete for some applicants.
- The model may learn historical biases and should be assessed for fairness before any real-world use.
- Probability estimates may require calibration.
- Further improvements could include threshold selection based on business costs, stronger validation, calibration, fairness analysis, monitoring for data drift, and a tested API or user interface.
- Results on a held-out split do not guarantee performance on future applicants.

## 15. License and Responsible Use

Follow the dataset provider's terms and the competition's rules when using or redistributing the data. This project is intended for learning and experimentation, not as a substitute for a complete, fair, legally compliant credit-risk assessment process.
