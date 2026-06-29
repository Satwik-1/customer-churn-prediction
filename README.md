# customer-churn-prediction
End-to-end customer churn prediction project using XGBoost, SHAP, and scikit-learn with hyperparameter tuning, threshold optimization, and business-focused model evaluation.

# Customer Churn Prediction

This project builds an end-to-end machine learning pipeline to predict customer churn using the Telco Customer Churn dataset. The goal is not only to train a model, but to evaluate it like a real business problem: identifying customers at risk of leaving so a company can prioritize retention efforts.

## Project Overview

Customer churn is a major business problem because retaining existing customers is often cheaper than acquiring new ones. In this project, I built and evaluated several machine learning models to predict whether a customer is likely to churn based on account information, service usage, contract type, payment method, and billing behavior.

The final model is a tuned XGBoost classifier wrapped in a scikit-learn pipeline that includes preprocessing and model prediction.

## Dataset

Dataset: Telco Customer Churn Dataset
Source: Kaggle

The dataset contains 7,043 customers and 21 columns, including:

* Customer demographics
* Account tenure
* Internet and phone services
* Contract type
* Payment method
* Monthly and total charges
* Churn outcome

The target variable is `Churn`, where:

* `No` = customer stayed
* `Yes` = customer churned

The dataset is imbalanced, with approximately 27% of customers churning.

## Data Cleaning

Key cleaning steps included:

* Converted `TotalCharges` from object/string to numeric
* Removed 11 rows with missing `TotalCharges`
* Encoded the target variable as:

  * `No` → 0
  * `Yes` → 1
* Removed `customerID` because it is an identifier, not a predictive feature

## Exploratory Data Analysis

EDA showed several strong churn patterns:

* Customers who churned had much lower average tenure.
* Month-to-month contracts were strongly associated with churn.
* Customers with fiber optic internet had higher churn rates.
* Customers without Tech Support or Online Security were more likely to churn.
* Electronic check payment was associated with higher churn.
* Churners had higher average monthly charges than non-churners.

These findings helped guide model interpretation and business recommendations.

## Modeling Approach

I compared multiple models:

| Model               |               Result | Status                      |
| ------------------- | -------------------: | --------------------------- |
| Logistic Regression |      ROC-AUC ≈ 0.832 | Strong baseline             |
| Decision Tree       |      ROC-AUC ≈ 0.662 | Rejected due to overfitting |
| Tuned Decision Tree |   CV ROC-AUC ≈ 0.834 | Interpretable candidate     |
| Random Forest       |      ROC-AUC ≈ 0.819 | Improved over single tree   |
| Tuned Random Forest |   CV ROC-AUC ≈ 0.850 | Strong candidate            |
| Tuned XGBoost       | Test ROC-AUC ≈ 0.836 | Final model                 |

The final model was selected because it had the strongest overall performance while still allowing interpretation through SHAP values.

## Final Model

The final model uses:

* XGBoost Classifier
* scikit-learn Pipeline
* ColumnTransformer preprocessing
* StandardScaler for numeric variables
* OneHotEncoder for categorical variables

Final XGBoost parameters:

```python
learning_rate = 0.05
max_depth = 2
n_estimators = 200
scale_pos_weight = 1
```

The saved model file is:

```text
churn_pipeline.pkl
```

This file contains both the preprocessing steps and the trained XGBoost model.

## Evaluation

The final model achieved:

```text
Test ROC-AUC ≈ 0.836
```

Accuracy alone was not treated as the main metric because the dataset is imbalanced. A model could predict most customers as non-churners and still achieve decent accuracy.

Instead, I focused on:

* ROC-AUC for model comparison
* Precision and recall for business tradeoffs
* F1-score for threshold selection
* SHAP values for model interpretation

## Threshold Tuning

At the default classification threshold of 0.50, the model caught only about half of the actual churners.

After testing different thresholds, a threshold around 0.30 produced a better business tradeoff by increasing churn recall substantially while maintaining reasonable precision.

This reflects a realistic churn use case: if retention outreach is relatively inexpensive, the business may prefer to flag more at-risk customers rather than miss likely churners.

## Explainability with SHAP

I used SHAP values to understand how features affected predictions.

Important churn drivers included:

* Short tenure
* Month-to-month contract
* Fiber optic internet
* No Tech Support
* No Online Security
* Electronic check payment

SHAP was useful because it showed not only which features mattered, but also whether they pushed predictions toward churn or toward retention.

For example:

* Low tenure pushed predictions toward churn.
* Month-to-month contracts pushed predictions toward churn.
* Higher tenure pushed predictions toward staying.

## Feature Engineering Experiments

I tested additional engineered features, including:

* Average monthly spend
* Protection service count

These features were not kept in the final model because they did not meaningfully improve held-out test performance. This helped keep the final model simpler and more defensible.

## Business Interpretation

The model suggests that churn risk is especially high among customers who are newer, on month-to-month contracts, using fiber optic internet, lacking support/security services, and paying by electronic check.

Potential business actions include:

* Prioritize retention offers for high-risk month-to-month customers.
* Encourage customers to move to longer-term contracts.
* Promote support/security service bundles.
* Investigate customer experience issues among fiber optic users.
* Use churn probability scores to prioritize outreach rather than treating all customers equally.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```text
churn_analysis.ipynb
```

The saved trained pipeline can be loaded with:

```python
import joblib

pipeline = joblib.load("churn_pipeline.pkl")
```

## Future Improvements

Possible next steps include:

* Incorporate business cost/profit into threshold selection
* Evaluate on newer customer data to detect performance degradation
* Calibrate predicted probabilities
* Refactor notebook into reusable training and prediction scripts
