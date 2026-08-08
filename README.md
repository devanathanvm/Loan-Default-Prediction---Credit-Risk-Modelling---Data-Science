Developed an end-to-end credit risk model using 307K+ loan applications to predict borrower default. Engineered financial risk features and compared Logistic Regression with LightGBM, achieving 0.752 AUC-ROC. Used SHAP for explainability and converted default probabilities into practical lending risk tiers.

# Credit Risk Modelling and Loan Default Prediction

## Project Overview

This project develops an end-to-end credit risk model using the Home Credit Default Risk dataset. The objective is to predict whether a loan applicant is likely to experience repayment difficulties and translate the predicted default probabilities into practical lending decisions.

The project covers data exploration, feature engineering, dimensionality reduction, predictive modelling, model explainability, and borrower risk segmentation.

## Dataset

The analysis uses approximately 307,000 loan applications with an observed default rate of 8.07%.

Target variable:

- `TARGET = 0`: Applicant repaid normally
- `TARGET = 1`: Applicant experienced repayment difficulties

Due to class imbalance, model performance was evaluated primarily using AUC-ROC, AUC-PR, and Brier Score rather than accuracy.

Dataset source: [Home Credit Default Risk – Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)

## Project Workflow

1. Explored application and credit-bureau datasets
2. Analysed missing values, outliers, and class imbalance
3. Handled the `DAYS_EMPLOYED` placeholder anomaly
4. Aggregated bureau records at the applicant level
5. Engineered borrower affordability and stability features
6. Removed near-zero-variance and highly correlated variables
7. Applied PCA and Factor Analysis for signal extraction
8. Compared Logistic Regression and LightGBM models
9. Used SHAP and partial dependence plots for explainability
10. Converted predicted probabilities into lending risk tiers

## Feature Engineering

Important engineered features include:

- Age and employment tenure
- Credit-to-income ratio
- Annuity-to-income ratio
- Credit-to-goods-price ratio
- Blended external credit score
- Employment anomaly indicator
- Aggregated bureau credit-history measures

## Models Evaluated

- L1 Logistic Regression
- L2 Logistic Regression
- Polynomial Logistic Regression
- Spline-based Logistic Regression
- LightGBM

## Model Performance

| Model | AUC-ROC | AUC-PR | Brier Score |
|---|---:|---:|---:|
| LightGBM | 0.7518 | 0.2422 | 0.06796 |
| L2 Logistic + Splines | 0.7347 | 0.2252 | 0.06882 |
| L1 Logistic Regression | 0.7338 | 0.2221 | 0.06896 |
| L2 Logistic Regression | 0.7338 | 0.2221 | 0.06896 |
| L2 Logistic + Polynomial | 0.7213 | 0.2059 | 0.06970 |

LightGBM produced the strongest overall performance, achieving an AUC-ROC of 0.7518.

## Model Explainability

SHAP values were used to explain global feature importance and individual borrower predictions. Partial dependence plots were used to examine how important variables affected predicted default risk.

Major risk drivers included:

- External credit scores
- Employment stability
- Age
- Credit-to-goods-price ratio
- Debt and annuity burden

## Risk-Based Lending Decisions

Applicants were grouped into five risk tiers:

| Tier | Predicted Default Probability | Proposed Decision |
|---|---:|---|
| A | Below 5% | Auto-approve |
| B | 5%–10% | Approve |
| C | 10%–20% | Approve with review |
| D | 20%–35% | Manual review |
| E | Above 35% | Decline |

Tiers D and E represented only 7.9% of applicants but contained approximately 28.2% of observed defaults.

These decision rules are illustrative and would require profitability, loss, fairness, and regulatory analysis before production use.

## Repository Structure

```text
├── S13_Solved_DataCartography.ipynb
├── S13_Student_DataCartography.ipynb
├── S13B_Solved_DataDeepDive.ipynb
├── S13B_Student_DataDeepDive.ipynb
├── S14_Solved_SignalExtraction.ipynb
├── S14_Student_SignalExtraction.ipynb
├── S15_17_Solved_ModelsToDecisions.ipynb
├── S15_17_Student_ModelsToDecisions.ipynb
└── README.md
