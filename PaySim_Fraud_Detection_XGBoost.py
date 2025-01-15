"""PaySim Fraud Detection Case using XGBoost
This script demonstrates the process of detecting fraudulent transactions using the XGBoost classifier.
It includes data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and preparation for deployment.
"""

import subprocess
import sys

# Install required packages if not already installed
def install_required_packages():
    required_packages = {
        'scikit-learn': '1.3.1',
        'xgboost': None,  # Latest version
        'graphviz': None  # Latest version
    }
    
    for package, version in required_packages.items():
        if version:
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Only run installation if this is the main script
if __name__ == "__main__":
    install_required_packages()

# Importing necessary libraries and suppressing warnings for cleaner output
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance, plot_tree
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 50)
pd.set_option('chained_assignment', None)
xgb.set_config(use_rmm=True)

# Create directories for saving plots and data
def create_directories():
    os.makedirs('images', exist_ok=True)
    os.makedirs('data', exist_ok=True)

# Data loading function
def load_data(data_path='data/train.csv'):
    """Load the dataset from the specified path."""
    raw_data = pd.read_csv(data_path)
    return raw_data

def save_plot(plt, filename):
    """Save plot to the images directory"""
    plt.savefig(os.path.join('images', filename))

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
    data['amountRatioDest'] = data['amount'] / data['newBalanceOrig']

    data['user_cum_count'] = data.groupby('nameOrig').cumcount()
    data['user_cumulative_amount'] = data.groupby('nameOrig')['amount'].cumsum().shift(1)
    data['user_avg_amount'] = data['user_cumulative_amount'] / data['user_cum_count']
    data['user_max_amount'] = data.groupby('nameOrig')['amount'].cummax().shift(1)
    data['user_avg_amount_ratio'] = data['amount'] / data['user_avg_amount']
    data['user_max_amount_ratio'] = data['amount'] / data['user_max_amount']
    data['prev_step'] = data.groupby('nameOrig')['step'].shift(1)
    data['time_since_last'] = data['step'] - data['prev_step']
    data['is_first_transaction'] = (data['prev_step'].isnull()).astype(int)

    # Same-step and same-day transaction features
    data['user_count_same_step'] = data.groupby(['nameOrig', 'step']).cumcount() + 1
    data['user_amount_same_step'] = data.groupby(['nameOrig', 'step'])['amount'].cumsum().shift(1)
    data['user_count_same_day'] = data.groupby(['nameOrig', 'day']).cumcount() + 1
    data['user_amount_same_day'] = data.groupby(['nameOrig', 'day'])['amount'].cumsum().shift(1)

    # Destination-based features
    data['dest_count_same_day'] = data.groupby(['nameDest', 'day']).cumcount() + 1
    data['dest_count_same_step'] = data.groupby(['nameDest', 'step']).cumcount() + 1
    data['dest_amount_same_step'] = data.groupby(['nameDest', 'step'])['amount'].cumsum().shift(1)
    data['dest_amount_same_day'] = data.groupby(['nameDest', 'day'])['amount'].cumsum().shift(1)

    # Handle missing and infinite values
    data.fillna(-1, inplace=True)
    data.replace([np.inf, -np.inf], -1, inplace=True)

    # Drop unnecessary columns
    cols_to_drop = ['Id', 'nameOrig', 'nameDest', 'amount_bin', 'prev_step']
    for col in cols_to_drop:
        if col in data.columns:
            data.drop(columns=[col], inplace=True, errors='ignore')

    # Prepare output
    X = data.drop(['isFraud'], axis=1, errors='ignore')
    y = data['isFraud'].values if 'isFraud' in data.columns else None
    
    return X, y

def get_predictions(X, model, process_data=False):
    """
    Generates predictions for new incoming data using the trained model.
    """
    if process_data:
        X, _ = prepare_data_for_prediction(X)

    # Ensure columns match training data
    training_cols = model.get_booster().feature_names
    X = X[training_cols]
    
    return model.predict(X)

def evaluate_model(new_data, target_data, model, process_data=False):
    """
    Evaluates the model's performance on new data.
    Prints classification metrics and displays a confusion matrix.
    """
    # Get predictions
    y_pred = get_predictions(new_data, model, process_data=process_data)

    # Print metrics
    print("\nClassification Report:")
    print(classification_report(target_data, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(target_data, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    save_plot(plt, 'evaluation_confusion_matrix.png')
    plt.close()

def main():
    """Main execution function"""
    # Create necessary directories
    create_directories()
    
    # Load the data
    try:
        raw_data = load_data()
        data = raw_data.copy()
    except FileNotFoundError:
        print("Error: Please ensure the training data file 'train.csv' is in the 'data' directory")
        sys.exit(1)

    print("Starting PaySim Fraud Detection analysis...")
    
    # Check the first few rows and null values
    print("First few rows of the dataset:")
    print(data.head())
    print("\nNull values in the dataset:")
    print(data.isnull().sum())

    # Display descriptive statistics
    print("\nDescriptive statistics of the dataset:")
    print(data.describe())

    # Data Cleaning and Feature Engineering
    # Dropping the 'isFlaggedFraud' column as it contains only 0s
    data.drop(columns=['isFlaggedFraud'], inplace=True, errors='ignore')

    # Checking the distribution of fraudulent transactions
    print("\nFraud distribution:")
    print(data['isFraud'].value_counts())
    fraud_rate = data['isFraud'].mean()
    print("\nFraud rate:")
    print(data['isFraud'].value_counts(normalize=True))

    # Exploratory Data Analysis (EDA)
    
    # 1. Distribution of transaction amounts
    plt.figure(figsize=(10, 6))
    plt.hist(data.amount, bins=25, log=True, edgecolor='k')
    plt.title('Distribution of Transaction Amounts')
    plt.xlabel('Amount')
    plt.ylabel('Frequency (log scale)')
    save_plot(plt, 'distribution_transaction_amounts.png')
    plt.close()

    # 2. Frequency of transaction types
    plt.figure(figsize=(10, 6))
    data['action'].value_counts().plot(kind='bar', color='skyblue', edgecolor='k')
    plt.title('Frequency of Transaction Types')
    plt.xlabel('Transaction Type')
    plt.ylabel('Frequency')
    save_plot(plt, 'frequency_transaction_types.png')
    plt.close()

    # 3. Fraudulent Transactions by Action Type
    plt.figure(figsize=(10, 6))
    data.groupby('action')['isFraud'].sum().plot(kind='bar', color='red', edgecolor='k')
    plt.title('Fraudulent Transactions by Action')
    plt.xlabel('Transaction Type')
    plt.ylabel('Frequency')
    save_plot(plt, 'fraudulent_transactions_by_action.png')
    plt.close()

    # 4. Transaction Count by Day
    plt.figure(figsize=(10, 6))
    data.groupby('day').size().plot(kind='line', color='brown')
    plt.title('Transaction Count by Day')
    plt.xlabel('Day')
    plt.ylabel('Count')
    save_plot(plt, 'transaction_count_by_day.png')
    plt.close()

    # 5. Fraudulent Transaction Count by Day
    plt.figure(figsize=(10, 6))
    data[data.isFraud==1].groupby('day').size().plot(kind='line', color='red')
    plt.title('Fraudulent Transaction Count by Day')
    plt.xlabel('Day')
    plt.ylabel('Count')
    save_plot(plt, 'fraudulent_transaction_count_by_day.png')
    plt.close()

    # 6. Transaction Count by Hour of Day
    plt.figure(figsize=(10, 6))
    data.groupby('hour_of_day').size().plot(kind='bar', color='purple', edgecolor='k')
    plt.title('Transaction Count by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    save_plot(plt, 'transaction_count_by_hour_of_day.png')
    plt.close()

    # 7. Transaction Frequency by Day of Week
    plt.figure(figsize=(10, 6))
    data.groupby('day_of_week').size().plot(kind='bar', color='orange', edgecolor='k')
    plt.title('Transaction Frequency by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Count')
    save_plot(plt, 'transaction_frequency_by_day_of_week.png')
    plt.close()

    # Filtering Transaction Types for Fraud Analysis
    # Since fraudulent transactions consist of only 'CASH_OUT' and 'TRANSFER' actions,
    # we filter these transaction types for meaningful comparison
    data_action_filtered = data[data.action.isin(["CASH_OUT", "TRANSFER"])]

    # 8. Fraudulent Transaction Ratio Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(data_action_filtered.groupby('day')['isFraud'].mean())
    plt.xlabel('Day')
    plt.ylabel('Fraudulent Ratio')
    plt.title('Fraudulent Transaction Ratio Over Time')
    save_plot(plt, 'fraudulent_transaction_ratio_over_time.png')
    plt.close()

    # 9. Fraudulent Transaction Ratio by Hour
    plt.figure(figsize=(10, 6))
    data_action_filtered.groupby('hour_of_day')['isFraud'].mean().plot(kind="bar", edgecolor='k', color='green')
    plt.xlabel('Hour')
    plt.ylabel('Fraudulent Ratio')
    plt.title('Fraudulent Transaction Ratio by Hour of Day')
    save_plot(plt, 'fraudulent_transaction_ratio_by_hour_of_day.png')
    plt.close()

    # 10. Fraudulent Transaction Ratio by Day of Week
    plt.figure(figsize=(10, 6))
    data_action_filtered.groupby('day_of_week')['isFraud'].mean().plot(kind="bar", edgecolor='k', color='teal')
    plt.xlabel('Day of Week')
    plt.ylabel('Fraudulent Ratio')
    plt.title('Fraudulent Transaction Ratio by Day of Week')
    save_plot(plt, 'fraudulent_transaction_ratio_by_day_of_week.png')
    plt.close()

    # 11. Binning Transaction Amounts and Analyzing Fraud
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
    save_plot(plt, 'number_of_fraudulent_transactions_by_amount_intervals.png')
    plt.close()

    # Visualizing fraud proportion by transaction amount bins
    plt.figure(figsize=(8, 6))
    sns.barplot(x='amount_bin', y='isFraud', data=fraud_proportions, color='purple')
    plt.title('Proportion of Fraudulent Transactions by Amount Range')
    plt.xlabel('Transaction Amount Range')
    plt.ylabel('Fraud Proportion')
    plt.xticks(rotation=30)
    plt.tight_layout()
    save_plot(plt, 'proportion_of_fraudulent_transactions_by_amount_range.png')
    plt.close()

    # 12. Correlation Analysis
    numeric_cols = data_action_filtered.drop(['Id'], axis=1).select_dtypes(include=['number', 'bool']).columns
    correlation_matrix = data_action_filtered[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix')
    save_plot(plt, 'correlation_matrix.png')
    plt.close()

    # 13. Model Training Pipeline
    def train_and_evaluate_model(data):
        """Train and evaluate the XGBoost model"""
        
        # Feature Engineering
        data['nameDest_type'] = data['nameDest'].str[0].map({'C': 'customer', 'B': 'merchant'}).fillna('other')
        data = pd.get_dummies(data, columns=['nameDest_type', 'action'], dtype="int64")

        # Prepare features and target
        X = data.drop(['isFraud', 'Id', 'nameOrig', 'nameDest', 'amount_bin'], axis=1, errors='ignore')
        y = data['isFraud'].values

        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.25, random_state=0)
        
        # Calculate class weight
        fraud_ratio_train = y_train.mean()
        scale_pos_weight = (1 - fraud_ratio_train) / fraud_ratio_train

        # Define cross-validation
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)

        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 5, 6],
            'learning_rate': [0.1, 0.3],
            'subsample': [0.8, 1.0],
            'method': ['hist']
        }

        # Initialize model
        xgb_clf = XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='aucpr'
        )

        # Grid search
        grid_search = GridSearchCV(
            estimator=xgb_clf,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=cv,
            verbose=2,
            n_jobs=-1
        )

        # Fit the model
        grid_search.fit(X_train, y_train)
        
        print("Best parameters found:", grid_search.best_params_)
        print("Best AUC score during CV:", grid_search.best_score_)

        # Get best model
        best_model = grid_search.best_estimator_

        # Make predictions
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]

        # Print metrics
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        print("ROC AUC:", roc_auc_score(y_val, y_pred_proba))

        # Plot confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Fraud', 'Fraud'],
                   yticklabels=['Not Fraud', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        save_plot(plt, 'confusion_matrix.png')
        plt.close()

        # Feature importance
        plot_importance(best_model, max_num_features=10)
        plt.title('Feature Importance')
        save_plot(plt, 'feature_importance.png')
        plt.close()

        return best_model

    # Train the model
    print("\nTraining the model...")
    best_model = train_and_evaluate_model(data)

    # Save the processed data
    print("\nSaving processed data...")
    data_processed, y = prepare_data_for_prediction(raw_data)
    data_processed.to_csv('data/processed_data.csv', index=False)

    print("\nEvaluation on the full dataset:")
    evaluate_model(data_processed, y, best_model)

    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()
