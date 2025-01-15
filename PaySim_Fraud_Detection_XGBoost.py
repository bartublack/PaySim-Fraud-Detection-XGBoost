# 0. Installing necessary packages
!pip install xgboost
!pip install graphviz

# 1. Importing necessary libraries and suppressing warnings for cleaner output
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix,precision_score, recall_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from xgboost import XGBClassifier,plot_importance,plot_tree
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 50)
pd.set_option('chained_assignment',None)

# 2. Importing the data
raw_data = pd.read_csv('train.csv')
data=raw_data.copy()

# 3. Checking for the first few rows and null values
display(data.head())
print(data.isnull().sum())

# 4. Displaying descriptive statistics of the dataset
display(data.describe())

# 5. Data Cleaning and Feature Engineering

# Dropping the 'isFlaggedFraud' column as it contains only 0s
data.drop(columns=['isFlaggedFraud'], inplace=True,errors='ignore')

# Creating temporal features based on the 'step' column
data['hour_of_day'] = data['step'] % 24
data['day_of_week'] = (data['step'] // 24) % 7
data['day']=(data['step'] // 24)

# Checking the distribution of fraudulent transactions
print(data['isFraud'].value_counts())
fraud_rate = data['isFraud'].mean()
print(data['isFraud'].value_counts(normalize=True))

# 6. Exploratory Data Analysis (EDA)

# 1. Distribution of transaction amounts
plt.figure(figsize=(10, 6))
plt.hist(data.amount, bins=25, log=True, edgecolor='k')
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency (log scale)')
plt.show()

# 2. Frequency of transaction types
plt.figure(figsize=(10, 6))
data['action'].value_counts().plot(kind='bar', color='skyblue', edgecolor='k')
plt.title('Frequency of Transaction Types')
plt.xlabel('Transaction Type')
plt.ylabel('Frequency')
plt.show()

# 3. Fraudulent Transactions by Action Type
plt.figure(figsize=(10, 6))
data.groupby('action')['isFraud'].sum().plot(kind='bar', color='red', edgecolor='k')
plt.title('Fraudulent Transactions by Action')
plt.xlabel('Transaction Type')
plt.ylabel('Frequency')
plt.show()

# 4. Transaction Count by Day
plt.figure(figsize=(10, 6))
data.groupby('day').size().plot(kind='line', color='brown',)
plt.title('Transaction Count by Day')
plt.xlabel('Day')
plt.ylabel('Count')
plt.show()

# 5. Fraudulent Transaction Count by Day
plt.figure(figsize=(10, 6))
data[data.isFraud==1].groupby('day').size().plot(kind='line', color='red')
plt.title('Fradulent Transaction Count by Day')
plt.xlabel('Day')
plt.ylabel('Count')
plt.show()

# 6. Transaction Count by Hour of Day
plt.figure(figsize=(10, 6))
data.groupby('hour_of_day').size().plot(kind='bar', color='purple', edgecolor='k')
plt.title('Transaction Count by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.show()

# 7. Transaction Frequency by Day of Week
plt.figure(figsize=(10, 6))
data.groupby('day_of_week').size().plot(kind='bar', color='orange', edgecolor='k')
plt.title('Transaction Frequency by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Count')
plt.show()

# 7. Filtering Transaction Types for Fraud Analysis

# Since fraudulent transactions consist of only 'CASH_OUT' and 'TRANSFER' actions,
# we filter these transaction types for meaningful comparison between fraudulent and non-fraudulent transactions.
data_action_filtered = data[data.action.isin(["CASH_OUT", "TRANSFER"])]

# 1. Fraudulent Transaction Ratio Over Time
plt.figure(figsize=(10, 6))
plt.plot(data_action_filtered.groupby('day')['isFraud'].mean())
plt.xlabel('Day')
plt.ylabel('Fraudulent Ratio')
plt.title('Fraudulent Transaction Ratio Over Time')
plt.show()

# 2. Fraudulent Transaction Ratio by Hour
plt.figure(figsize=(10, 6))
data_action_filtered.groupby('hour_of_day')['isFraud'].mean().plot(kind="bar", edgecolor='k', color='green')
plt.xlabel('Hour')
plt.ylabel('Fraudulent Ratio')
plt.title('Fraudulent Transaction Ratio by Hour of Day')
plt.show()

# 3. Fraudulent Transaction Ratio by Day of Week
plt.figure(figsize=(10, 6))
data_action_filtered.groupby('day_of_week')['isFraud'].mean().plot(kind="bar", edgecolor='k', color='teal')
plt.xlabel('Day of Week')
plt.ylabel('Fraudulent Ratio')
plt.title('Fraudulent Transaction Ratio by Day of Week')
plt.show()

# 8. Binning Transaction Amounts and Analyzing Fraud

# Define bin edges for transaction amounts
bin_edges = [0, 1e5, 5e5, 1e6, 5e6, 1e7, 2e7]
data_action_filtered['amount_bin'] = pd.cut(data_action_filtered['amount'], bins=bin_edges, include_lowest=True)

# Calculate fraud counts and proportions per bin
fraud_counts = data_action_filtered.groupby('amount_bin')['isFraud'].sum().reset_index()
fraud_proportions = data_action_filtered.groupby('amount_bin')['isFraud'].mean().reset_index()

# Visualizing fraud count by transaction amount bins
plt.figure(figsize=(8, 6))
sns.barplot(x='amount_bin', y='isFraud', data=fraud_counts, color='red')
plt.title('Number of Fraudulent Transactions by Amount Intervals')
plt.xlabel('Transaction Amount Range')
plt.ylabel('Fraud Count')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Visualizing fraud proportion by transaction amount bins
plt.figure(figsize=(8, 6))
sns.barplot(x='amount_bin', y='isFraud', data=fraud_proportions, color='purple')
plt.title('Proportion of Fraudulent Transactions by Amount Range')
plt.xlabel('Transaction Amount Range')
plt.ylabel('Fraud Proportion')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 9. Correlation Analysis

# Selecting numeric and boolean columns for correlation
numeric_cols = data_action_filtered.drop(['Id'], axis=1).select_dtypes(include=['number', 'bool']).columns
correlation_matrix = data_action_filtered[numeric_cols].corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
plt.show()

# 10. Encoding Categorical Features

# Extracting the first character from 'nameDest' to create a new categorical feature
data['nameDest_type'] = data['nameDest'].str[0].map({'C': 'customer', 'B': 'merchant'}).fillna('other')

# One-hot encoding categorical variables
data = pd.get_dummies(data, columns=['nameDest_type'], dtype="int64")  # Encode 'nameDest_type'
data = pd.get_dummies(data, columns=['action'], dtype="int64")  # Encode 'action'

# 11. Splitting Features and Target Variable

# Defining feature matrix X and target vector y
X = data.drop(['isFraud', 'Id', 'nameOrig', 'nameDest', 'amount_bin'], axis=1, errors='ignore')
y = data['isFraud'].values

# Splitting the dataset into training and validation sets with stratification
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.25, random_state=0)

# Calculating scale_pos_weight to handle class imbalance
fraud_ratio_train = y_train.mean()
scale_pos_weight = (1 - fraud_ratio_train) / fraud_ratio_train

# 12. Initial Model Training with XGBoost and Hyperparameter Tuning

# Defining the parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.1],
    'subsample': [0.8, 1.0],
}

# Initializing the XGBoost classifier with specified objective and scale_pos_weight
xgb_clf = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='aucpr'
)

# Setting up GridSearchCV for hyperparameter tuning with 3-fold cross-validation
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    n_jobs=-1,
    verbose=2
)

# Fitting Grid Search on the training data
grid_search.fit(X_train, y_train)

# Displaying the best parameters and corresponding AUC score
print("Best parameters found:", grid_search.best_params_)
print("Best AUC score during CV:", grid_search.best_score_)

# Extracting the best model from Grid Search
best_model = grid_search.best_estimator_

# Making predictions on the validation set
y_pred_best = best_model.predict(X_val)
y_pred_best_proba = best_model.predict_proba(X_val)[:, 1]

# Classification metrics on the validation set
print("Classification Report:")
print(classification_report(y_val, y_pred_best))
print("ROC AUC:", roc_auc_score(y_val, y_pred_best_proba))

# Confusion matrix for validation set
cm = confusion_matrix(y_val, y_pred_best)

# Visualizing the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 13. Feature Importance Analysis

# Extracting feature importances from the best model
feature_importances = best_model.feature_importances_

# Getting feature names from the training set
feature_names = X_train.columns if hasattr(X_train, 'columns') else np.arange(X_train.shape[1])

# Creating a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plotting feature importances
plot_importance(best_model)
plt.show()

gc.collect()  # Garbage collection to free memory

# 14. Advanced Feature Engineering to Enhance Model Performance

# Sorting data to ensure proper cumulative calculations
data = data.sort_values(['step', 'nameOrig', 'Id'])

# Creating ratio features based on transaction amounts and balances
data['amountRatioOrig'] = data['amount'] / data['oldBalanceOrig']
data['amountRatioDest'] = data['amount'] / data['newBalanceOrig']  # Assuming 'newBalanceDest' was intended

# Cumulative transaction features for originating users
data['user_cum_count'] = data.groupby('nameOrig').cumcount()
data['user_cumulative_amount'] = data.groupby('nameOrig')['amount'].cumsum().shift(1)
data['user_avg_amount'] = data['user_cumulative_amount'] / data['user_cum_count']
data['user_max_amount'] = data.groupby('nameOrig')['amount'].cummax().shift(1)
data['user_avg_amount_ratio'] = data['amount'] / data['user_avg_amount']
data['user_max_amount_ratio'] = data['amount'] / data['user_max_amount']
data['prev_step'] = data.groupby('nameOrig')['step'].shift(1)
data['time_since_last'] = data['step'] - data['prev_step']
data['is_first_transaction'] = (data['prev_step'].isnull()).astype(int)

# Sorting again for same-step calculations
data = data.sort_values(by=['step', 'nameOrig', 'amount', 'nameDest'])

# Same-step and same-day transaction counts and amounts for users
data['user_count_same_step'] = data.groupby(['nameOrig', 'step']).cumcount() + 1
data['user_amount_same_step'] = data.groupby(['nameOrig', 'step'])['amount'].cumsum().shift(1)
data['user_count_same_day'] = data.groupby(['nameOrig', 'day']).cumcount() + 1
data['user_amount_same_day'] = data.groupby(['nameOrig', 'day'])['amount'].cumsum().shift(1)

# Sorting for destination-based feature engineering
data = data.sort_values(by=['step', 'nameDest', 'Id'])

# Same-day and same-step transaction counts and amounts for destination accounts
data['dest_count_same_day'] = data.groupby(['nameDest', 'day']).cumcount() + 1
data['dest_count_same_step'] = data.groupby(['nameDest', 'step']).cumcount() + 1
data['dest_amount_same_step'] = data.groupby(['nameDest', 'step'])['amount'].cumsum().shift(1)
data['dest_amount_same_day'] = data.groupby(['nameDest', 'day'])['amount'].cumsum().shift(1)

# Dropping 'amount_bin' as it's no longer needed
data.drop(['amount_bin'], axis=1, inplace=True, errors='ignore')

# Handling missing and infinite values
data.fillna(-1, inplace=True)
data.replace([np.inf, -np.inf], -1, inplace=True)

# 15. Preparing Data After Feature Engineering

# Defining feature matrix X and target vector y after feature engineering
X = data.drop(['isFraud', 'Id', 'nameOrig', 'nameDest', 'amount_bin', 'prev_step'], axis=1, errors='ignore')
y = data['isFraud'].values

# Splitting the dataset into training and validation sets with stratification
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.25, random_state=0)

# Recalculating scale_pos_weight to handle class imbalance in the new training set
fraud_ratio_train = y_train.mean()
scale_pos_weight = (1 - fraud_ratio_train) / fraud_ratio_train

# 16. Model Training with Advanced Hyperparameter Tuning Using Stratified K-Fold Cross-Validation

# Defining Stratified K-Fold cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

# Defining a new parameter grid for further tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.1, 0.3],
    'subsample': [0.8, 1.0],
    # Additional parameters can be added here
}

# Initializing the XGBoost classifier with updated parameters
xgb_clf = XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    tree_method='hist',
    use_label_encoder=False,
    eval_metric='aucpr'
)

# Setting up GridSearchCV with the new parameter grid and cross-validation strategy
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

# Fitting Grid Search on the training data
grid_search.fit(X_train, y_train)

# Displaying the best parameters and corresponding AUC score from Grid Search
print("Best parameters found:", grid_search.best_params_)
print("Best AUC score during CV:", grid_search.best_score_)

# Extracting the best model from Grid Search
best_model = grid_search.best_estimator_

# Making predictions on the validation set
y_pred_best = best_model.predict(X_val)
y_pred_best_proba = best_model.predict_proba(X_val)[:, 1]

# Classification metrics on the validation set
print("Classification Report:")
print(classification_report(y_val, y_pred_best))
print("ROC AUC:", roc_auc_score(y_val, y_pred_best_proba))

# Confusion matrix for validation set
cm = confusion_matrix(y_val, y_pred_best)

# Visualizing the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 17. Cross-Validation Predictions on the Entire Dataset

# Defining Stratified K-Fold cross-validation with 5 splits
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Generating cross-validated predictions
y_pred_cv = cross_val_predict(best_model, X, y, cv=cv, method='predict')

# Classification report on the entire dataset using cross-validation predictions
print("Classification Report on Validation Set:")
print(classification_report(y, y_pred_cv))

# Confusion matrix for cross-validation results
cm = confusion_matrix(y, y_pred_cv)

# Visualizing the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix (Cross-Validation)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 18. Final Feature Importance and Tree Visualization

# Extracting final feature importances from the best model
feature_importances_final = best_model.feature_importances_

# Getting feature names from the training set
feature_names_final = X_train.columns if hasattr(X_train, 'columns') else np.arange(X_train.shape[1])

# Creating a DataFrame for final feature importances
importance_df_final = pd.DataFrame({
    'Feature': feature_names_final,
    'Importance': feature_importances_final
}).sort_values(by='Importance', ascending=False)

# Plotting the top 10 features based on gain importance
plot_importance(best_model, max_num_features=10, importance_type="gain")
plt.show()

gc.collect()  # Garbage collection to free memory

# 19. Visualizing the Decision Tree Structure

# Creating a large figure for the decision tree visualization
fig, ax = plt.subplots(figsize=(30, 30))

# Plotting the first tree in the XGBoost model
plot_tree(best_model, num_trees=0, ax=ax)
plt.show()

# 20. Preparing Data for Prediction and Defining Prediction Functions

def prepare_data_for_prediction(data):
    """
    Preprocesses new incoming data to match the training data format.
    Includes feature engineering and encoding.
    """
    # Creating time-based features
    data['hour_of_day'] = data['step'] % 24
    data['day_of_week'] = (data['step'] // 24) % 7
    data['day'] = (data['step'] // 24)

    # Deriving new categories from nameDest
    data['nameDest_type'] = data['nameDest'].str[0].map({'C': 'customer', 'B': 'merchant'}).fillna('other')

    # One-hot encoding categorical features
    data = pd.get_dummies(data, columns=['nameDest_type'], dtype="int64")
    data = pd.get_dummies(data, columns=['action'], dtype="int64")

    # Sorting the data for cumulative calculations
    data = data.sort_values(['step', 'nameOrig', 'Id'])

    # Creating ratio and cumulative features
    data['amountRatioOrig'] = data['amount'] / data['oldBalanceOrig']
    data['amountRatioDest'] = data['amount'] / data['newBalanceOrig']  # Assuming 'newBalanceDest' was intended

    data['user_cum_count'] = data.groupby('nameOrig').cumcount()
    data['user_cumulative_amount'] = data.groupby('nameOrig')['amount'].cumsum().shift(1)
    data['user_avg_amount'] = data['user_cumulative_amount'] / data['user_cum_count']
    data['user_max_amount'] = data.groupby('nameOrig')['amount'].cummax().shift(1)
    data['user_avg_amount_ratio'] = data['amount'] / data['user_avg_amount']
    data['user_max_amount_ratio'] = data['amount'] / data['user_max_amount']
    data['prev_step'] = data.groupby('nameOrig')['step'].shift(1)
    data['time_since_last'] = data['step'] - data['prev_step']
    data['is_first_transaction'] = (data['prev_step'].isnull()).astype(int)

    # Sorting again for same-step calculations
    data = data.sort_values(by=['step', 'nameOrig', 'amount', 'nameDest'])

    # Same-step and same-day transaction features for users
    data['user_count_same_step'] = data.groupby(['nameOrig', 'step']).cumcount() + 1
    data['user_amount_same_step'] = data.groupby(['nameOrig', 'step'])['amount'].cumsum().shift(1)
    data['user_count_same_day'] = data.groupby(['nameOrig', 'day']).cumcount() + 1
    data['user_amount_same_day'] = data.groupby(['nameOrig', 'day'])['amount'].cumsum().shift(1)

    # Sorting for destination-based feature engineering
    data = data.sort_values(by=['step', 'nameDest', 'Id'])

    # Same-day and same-step transaction features for destination accounts
    data['dest_count_same_day'] = data.groupby(['nameDest', 'day']).cumcount() + 1
    data['dest_count_same_step'] = data.groupby(['nameDest', 'step']).cumcount() + 1
    data['dest_amount_same_step'] = data.groupby(['nameDest', 'step'])['amount'].cumsum().shift(1)
    data['dest_amount_same_day'] = data.groupby(['nameDest', 'day'])['amount'].cumsum().shift(1)

    # Dropping unnecessary columns
    data.drop(['amount_bin'], axis=1, inplace=True, errors='ignore')

    # Handling missing and infinite values
    data.fillna(-1, inplace=True)
    data.replace([np.inf, -np.inf], -1, inplace=True)

    # Dropping columns not needed for prediction
    cols_to_drop = ['isFraud', 'Id', 'nameOrig', 'nameDest', 'amount_bin', 'prev_step']
    for c in cols_to_drop:
        if c in data.columns:
            data.drop(columns=[c], axis=1, inplace=True, errors='ignore')

    # Returning the processed DataFrame
    return data

def get_predictions(new_data, model):
    """
    Generates predictions for new incoming data using the trained model.
    """
    # Preparing data
    X_new = prepare_data_for_prediction(new_data.copy())
    # Getting predictions
    preds = model.predict(X_new)
    return preds

# Example usage:
# preds = get_predictions(incoming_data, best_model)
# print(preds)

# 21. Model Evaluation and Training Functions

def evaluate_model(new_data, model):
    """
    Evaluates the model's performance on new data.
    Prints classification metrics and displays a confusion matrix.
    """
    # Extracting features and target
    X = new_data.drop(['isFraud'], axis=1, errors='ignore')
    y = new_data['isFraud'].values

    # Predicting using the trained model
    y_pred = get_predictions(new_data, model)

    # Printing classification report
    print("Classification Report:")
    print(classification_report(y, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Plotting confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def train_model(new_data, model):
    """
    Trains the model on new data.
    """
    # Extracting features and target
    X = new_data.drop(['isFraud'], axis=1, errors='ignore')
    y = new_data['isFraud'].values

    # Training the model on new data
    model.fit(X, y, xgb_model=model.get_booster())

    return model
