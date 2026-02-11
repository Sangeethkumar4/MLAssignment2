# ML Classification Models - Assignment 2

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict the presence of heart disease in patients. Given various clinical and demographic attributes, the task is to classify whether a patient has heart disease or not. This is a binary classification problem that helps understand how different ML algorithms perform on medical diagnostic data.

## b. Dataset Description

**Dataset:** Heart Disease UCI Dataset
**Source:** UCI Machine Learning Repository
**URL:** https://archive.ics.uci.edu/dataset/45/heart+disease

| Property | Value |
|----------|-------|
| Number of Instances | 920 (after preprocessing) |
| Number of Features | 13 |
| Target Variable | num (Binary: 0 = No Disease, 1 = Disease) |
| Missing Values | Handled by dropping rows |

### Features:

1. **age** - Age of the patient in years
2. **sex** - Sex of the patient (Male/Female)
3. **cp** - Chest pain type (typical angina, atypical angina, non-anginal, asymptomatic)
4. **trestbps** - Resting blood pressure (mm Hg)
5. **chol** - Serum cholesterol (mg/dl)
6. **fbs** - Fasting blood sugar > 120 mg/dl (True/False)
7. **restecg** - Resting electrocardiographic results (normal, lv hypertrophy, st-t abnormality)
8. **thalch** - Maximum heart rate achieved
9. **exang** - Exercise induced angina (True/False)
10. **oldpeak** - ST depression induced by exercise relative to rest
11. **slope** - Slope of the peak exercise ST segment (upsloping, flat, downsloping)
12. **ca** - Number of major vessels (0-3) colored by fluoroscopy
13. **thal** - Thalassemia (normal, fixed defect, reversable defect)

### Target:
- **0**: No heart disease (num = 0)
- **1**: Heart disease present (num > 0)

## c. Models Used

Six machine learning classification models were implemented and evaluated:

1. **Logistic Regression** - Linear model for binary classification
2. **Decision Tree Classifier** - Tree-based model with interpretable rules
3. **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
4. **Naive Bayes (Gaussian)** - Probabilistic classifier based on Bayes theorem
5. **Random Forest (Ensemble)** - Ensemble of decision trees using bagging
6. **XGBoost (Ensemble)** - Gradient boosting ensemble method

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8424 | 0.9100 | 0.8420 | 0.8424 | 0.8419 | 0.6839 |
| Decision Tree | 0.7772 | 0.7758 | 0.7772 | 0.7772 | 0.7772 | 0.5543 |
| KNN | 0.8315 | 0.8989 | 0.8318 | 0.8315 | 0.8311 | 0.6622 |
| Naive Bayes | 0.8207 | 0.8938 | 0.8207 | 0.8207 | 0.8206 | 0.6407 |
| Random Forest (Ensemble) | 0.8533 | 0.9201 | 0.8530 | 0.8533 | 0.8529 | 0.7058 |
| XGBoost (Ensemble) | 0.8478 | 0.9155 | 0.8477 | 0.8478 | 0.8475 | 0.6949 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Strong baseline performance with high AUC (0.91). The linear decision boundary works well for this dataset indicating good linear separability between classes. Fast training and provides interpretable coefficients for understanding feature importance. |
| Decision Tree | Lowest accuracy among all models. Prone to overfitting on training data leading to poor generalization. However, provides the most interpretable decision rules which can be valuable for medical diagnosis explanation. |
| KNN | Good performance with sensitivity to feature scaling. The instance-based approach captures local patterns in the data. Performance could be improved with hyperparameter tuning of k value. Works well when similar patients have similar outcomes. |
| Naive Bayes | Reasonable performance despite the strong independence assumption between features. Fast training and prediction. Lower accuracy compared to ensemble methods as medical features often have dependencies that violate the naive assumption. |
| Random Forest (Ensemble) | Best overall performer with highest accuracy (0.8533) and AUC (0.9201). Effectively reduces overfitting through bagging and random feature selection. Robust to outliers and provides reliable feature importance rankings. Recommended for production use. |
| XGBoost (Ensemble) | Second best performer with strong metrics across all criteria. Gradient boosting iteratively corrects errors leading to high accuracy. Slightly lower than Random Forest possibly due to the relatively small dataset size where bagging outperforms boosting. |

## Project Structure

```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- heart_disease_uci.csv
│-- model/
    │-- logistic_regression.py
    │-- decision_tree.py
    │-- knn.py
    │-- naive_bayes.py
    │-- random_forest.py
    │-- xgboost_model.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the Streamlit App

```bash
streamlit run app.py
```

## Features of the Streamlit App

- CSV file upload for test data
- Model selection dropdown (6 models available)
- Display of all evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix visualization
- Classification report
- Compare all models functionality with visual comparison charts

## Deployment

The app is deployed on Streamlit Community Cloud.

**Live App URL:** [https://mlassignment2-4tv2edeqpzcxkaz2tfujy4.streamlit.app/]

## Author

M.Tech (AIML/DSE) - Machine Learning Assignment 2
