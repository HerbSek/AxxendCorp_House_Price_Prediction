# AxxendCorp House Price Prediction Project

## Overview

This project is an AI/ML assessment for AxxendCorp that demonstrates a complete machine learning workflow for predicting house prices. The project uses the Kaggle House Prices dataset to build and evaluate multiple regression models, with a focus on data preprocessing, feature engineering, and model optimization.

## Project Goal

Build a machine learning model to predict house prices based on various features such as size, location, and number of rooms. The project demonstrates comprehensive data preprocessing, model training, evaluation, and reporting of insights.

## Features

- **Complete ML Pipeline**: From data loading to model deployment
- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Data Preprocessing**: Handling missing values, outlier detection, and feature scaling
- **Feature Engineering**: Creating new meaningful features from existing data
- **Multiple Model Comparison**: Linear Regression, Decision Tree, and Random Forest
- **Hyperparameter Tuning**: Grid search optimization for best performance
- **Model Evaluation**: RMSE, MAE, and R² score metrics
- **Feature Importance Analysis**: Understanding which features most affect predictions

## Dataset

The project uses the [Kaggle House Prices Dataset](https://www.kaggle.com/datasets/lespin/house-prices-dataset) which contains:

- **1460 samples** with 81 features
- **Target variable**: SalePrice (house sale price in USD)
- **Feature types**: Numerical, ordinal categorical, and nominal categorical features
- **Data files**:
  - `fill_null_data.csv` - Dataset with missing values filled
  - `cleaned_data.csv` - Fully preprocessed dataset ready for modeling

## Technology Stack

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning algorithms and tools
- **kagglehub** - Dataset downloading
- **joblib** - Model serialization

## Project Structure

```
AxxendCorp_House_Price_Prediction/
├── AxxendCorp_EDA.ipynb          # Exploratory Data Analysis
├── AxxendCorp_Model_Training.ipynb # Model training and evaluation
├── AxxendCorp_House_Price_Report.pdf # Comprehensive project report
├── requirements.txt               # Python dependencies
├── cleaned_data.csv              # Preprocessed dataset
├── fill_null_data.csv            # Dataset with missing values filled
├── random_forest_model.pkl       # Trained Random Forest model
├── scaler.pkl                    # StandardScaler for feature scaling
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AxxendCorp_House_Price_Prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Jupyter** (if not already installed)
   ```bash
   pip install jupyter
   ```

## Usage

### Running the Analysis

1. **Exploratory Data Analysis**
   ```bash
   jupyter notebook AxxendCorp_EDA.ipynb
   ```
   This notebook covers:
   - Data loading and initial exploration
   - Missing value analysis and treatment
   - Feature categorization (ordinal vs nominal)
   - Correlation analysis
   - Outlier detection and handling
   - Feature engineering

2. **Model Training and Evaluation**
   ```bash
   jupyter notebook AxxendCorp_Model_Training.ipynb
   ```
   This notebook covers:
   - Data preprocessing and train-test split
   - Feature scaling
   - Model training (Linear Regression, Decision Tree, Random Forest)
   - Model evaluation and comparison
   - Hyperparameter tuning with GridSearchCV
   - Feature importance analysis
   - Model serialization


### Final Optimized Model
- **Algorithm**: Random Forest Regressor
- **Best Parameters**: Determined via GridSearchCV
- **Cross-validation**: 5-fold CV
- **Final Metrics**: RMSE, MAE, R² Score on test set

## Key Insights

### Top 5 Most Important Features
1. **OverallQual** - Overall material and finish quality
2. **GrLivArea** - Above grade living area square footage
3. **GarageCars** - Size of garage in car capacity
4. **GarageArea** - Size of garage in square feet
5. **TotalBsmtSF** - Total basement area square footage

### Data Preprocessing Steps
1. **Missing Value Treatment**:
   - Numerical features filled with median values
   - Categorical features filled with appropriate default values

2. **Feature Engineering**:
   - `HouseAge`: Age of the house in years
   - `RemodAge`: Years since last remodeling
   - `SinceRemod`: Years between construction and remodeling
   - `TotalRooms`: Total rooms above ground

3. **Feature Selection**:
   - Removed features with low correlation to SalePrice
   - Applied one-hot encoding to nominal categorical features
   - Used ordinal encoding for ordinal categorical features

## Results

The Random Forest model was selected as the final model due to its superior performance compared to Linear Regression and Decision Tree models. The model demonstrates good predictive capability with reasonable error metrics and provides valuable insights into the factors that most influence house prices.