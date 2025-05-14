# Telco-Customer-Churn-Prediction
This project predicts customer churn for a telecom company using advanced machine learning techniques, including XGBoost and GridSearchCV for hyperparameter tuning. The goal is to help the company identify which customers are likely to leave and take proactive action to retain them
# Telco Customer Churn Prediction

This project uses machine learning to predict whether a customer will churn (cancel service) based on a dataset of customer information from a telecommunications company. The goal is to build a model that can accurately predict churn and provide insights into the factors contributing to it.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Libraries Used](#libraries-used)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Feature Importance](#feature-importance)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Overview

In this project, we perform the following tasks:
- **Data Preprocessing**: Clean and preprocess the dataset to make it suitable for machine learning models.
- **Modeling**: Train a machine learning model (XGBoost) to predict customer churn.
- **Hyperparameter Tuning**: Use GridSearchCV to find the best hyperparameters for the XGBoost model.
- **Evaluation**: Evaluate the model's performance using metrics like accuracy, precision, recall, F1 score, confusion matrix, and ROC-AUC score.
- **Feature Importance**: Identify and visualize the most important features for churn prediction.

## Dataset

The dataset used in this project is the [Telco Customer Churn dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv). It contains information about customers and whether or not they churned. The columns in the dataset include customer demographics, account information, and services.

## Libraries Used

- `pandas`: Data manipulation and analysis.
- `numpy`: Numerical operations and data handling.
- `matplotlib`: Data visualization.
- `seaborn`: Statistical data visualization.
- `scikit-learn`: Machine learning models, data splitting, and evaluation metrics.
- `xgboost`: eXtreme Gradient Boosting model.
- `warnings`: To suppress unnecessary warnings during execution.

## Modeling

1. **Data Preprocessing**:
   - Dropped the `customerID` column.
   - Converted `TotalCharges` to numeric and dropped rows with missing values.
   - Encoded the target variable `Churn` as 1 for 'Yes' and 0 for 'No'.
   - One-hot encoded categorical columns (e.g., `gender`, `Contract`, `PaymentMethod`).
   - Standardized the numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) to have mean = 0 and standard deviation = 1.

2. **Modeling**:
   - Split the dataset into training and test sets (80% train, 20% test).
   - Trained an XGBoost classifier (`XGBClassifier`).
   - Used GridSearchCV to tune hyperparameters like `n_estimators`, `max_depth`, `learning_rate`, and `subsample`.

3. **Model Evaluation**:
   - Evaluated the model using accuracy, precision, recall, F1 score, confusion matrix, and ROC-AUC score.
   
4. **Feature Importance**:
   - Visualized the top 10 most important features using XGBoost’s built-in `plot_importance()` function.

## Evaluation

### Accuracy
The model's accuracy score on the test set is evaluated.

### Classification Report
Provides precision, recall, and F1 score metrics for each class.

### Confusion Matrix
Shows the true positives, true negatives, false positives, and false negatives.

### ROC-AUC Score
The ROC-AUC score is used to evaluate the model's ability to distinguish between the classes.

## Feature Importance

The following chart shows the most important features that contribute to predicting customer churn:

![Feature Importance](feature_importance.png)

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/telco-churn-prediction.git
   cd telco-churn-prediction
