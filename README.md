# PaySim-Fraud-Detection-XGBoost

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)

An XGBoost-based fraud detection model to identify money laundering in mobile transactions using the PaySim synthetic dataset.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
- [Usage](#usage)
  - [Jupyter Notebook](#jupyter-notebook)
  - [Python Script](#python-script)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Fraud detection in financial transactions is crucial for preventing financial losses and maintaining trust in financial systems. This project leverages the power of XGBoost, a scalable and efficient gradient boosting framework, to build a robust model capable of identifying fraudulent transactions effectively.

Using the PaySim synthetic dataset, which simulates mobile transactions, this project encompasses the entire machine learning pipeline:

- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training and Hyperparameter Tuning
- Model Evaluation
- Deployment Preparation

## Dataset

The [PaySim dataset](https://github.com/krishnaik06/PaySim) is a synthetic simulation of mobile money transactions, which includes information about both legitimate and fraudulent transactions. It is designed to mimic real-world mobile banking transactions, making it ideal for training and testing fraud detection models.

**Key Features:**

- **Id:** Transaction identifier
- **step:** Hours elapsed since the start of the simulation
- **type:** Transaction type (e.g., CASH_IN, CASH_OUT, TRANSFER)
- **amount:** Transaction amount
- **nameOrig:** Customer identifier initiating the transaction
- **oldBalanceOrig:** Original balance before the transaction
- **newBalanceOrig:** New balance after the transaction
- **nameDest:** Recipient identifier
- **oldBalanceDest:** Original balance of the recipient before the transaction
- **newBalanceDest:** New balance of the recipient after the transaction
- **isFraud:** Indicator of fraudulent transaction (1 for fraud, 0 otherwise)
- **isFlaggedFraud:** Indicator if the transaction was flagged as fraud

## Features

The project includes extensive feature engineering to enhance model performance, such as:

- **Time-Based Features:** Hour of day, day of week, and cumulative time features.
- **Ratio Features:** Transaction amount ratios relative to original and new balances.
- **Cumulative Features:** Counts and cumulative amounts of transactions per user.
- **Aggregated Features:** Counts and amounts of transactions per step and day for both originators and recipients.
- **Categorical Encoding:** One-hot encoding for categorical features like transaction type and recipient type.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip

### Clone the Repository

```bash
git clone https://github.com/bartublack/PaySim-Fraud-Detection-XGBoost.git
cd PaySim-Fraud-Detection-XGBoost
```

### Install Dependencies

Install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Alternatively, you can install the dependencies manually:

```bash
pip install matplotlib==3.8.4 numpy==2.2.1 pandas==2.2.3 scikit_learn==1.3.1 seaborn==0.13.2 xgboost==2.1.3
```

## Usage

### Jupyter Notebook

The project includes a comprehensive Jupyter Notebook `PaySim_Fraud_Detection_XGBoost.ipynb` that demonstrates the entire workflow from data loading to model evaluation.

1. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

2. **Open the Notebook:**

   Navigate to `PaySim_Fraud_Detection_XGBoost.ipynb` in your browser and run the cells sequentially.

### Python Script

For a more streamlined approach, you can run the Python script `paysim_fraud_detection_xgboost.py`, which includes data preprocessing, feature engineering, model training, and evaluation.

1. **Execute the Script:**

   ```bash
   python paysim_fraud_detection_xgboost.py
   ```

## Results

After training, the model achieves high accuracy and AUC scores, effectively distinguishing between fraudulent and legitimate transactions. The project also generates several visualization plots saved in the `images` directory, including:

- Feature Importance Charts
- Confusion Matrices
- Decision Tree Structures

These visuals aid in understanding model performance and feature relevance.

## Project Structure

```
PaySim-Fraud-Detection-XGBoost/
│
├── data/
│   └── train.csv # Raw dataset
│
├── images/
│   ├── base_feature_importance.png
│   ├── confusion_matrix.png
│   ├── feature_engineered_confusion_matrix.png
│   ├── final_feature_importance.png
│   ├── decision_tree_structure.png
│   └── cross_validation_confusion_matrix.png
│
├── paysim_fraud_detection_xgboost.py # Main script for data processing and model training
├── PaySim_Fraud_Detection_XGBoost.ipynb # Jupyter Notebook
├── requirements.txt # Python dependencies
├── README.md
├── LICENSE
└── .gitignore
```

