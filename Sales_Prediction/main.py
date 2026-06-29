"""
CodSoft Data Science Internship - Task 4: Sales Prediction Using Python
Author: CodSoft Intern
Description: Predicts sales based on advertising budgets for TV, Radio, and Newspaper
             using Linear Regression. Handles dynamic file paths, performs data cleaning,
             generates visualizations, and evaluates model performance.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set theme for modern, clean visual aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Define absolute paths relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
DATA_PATH = os.path.join(DATA_DIR, 'advertising.csv')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_or_generate_data():
    """
    Loads dataset from local 'data/advertising.csv'. 
    Falls back to downloading from GitHub, or generating synthetic data if offline.
    """
    print("=" * 60)
    print("STEP 1: LOADING DATASET")
    print("=" * 60)
    
    url = "https://raw.githubusercontent.com/utjimmyx/regression/master/advertising.csv"
    
    if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0:
        print(f"Loading dataset from existing local file: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        return df

    try:
        print(f"Local file missing or empty. Attempting to download from: {url}")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(DATA_PATH, 'wb') as f:
                f.write(response.content)
            print(f"Dataset downloaded successfully and saved to: {DATA_PATH}")
            df = pd.read_csv(DATA_PATH)
            return df
        else:
            raise Exception(f"HTTP Status: {response.status_code}")
    except Exception as e:
        print(f"\n[WARNING] Could not retrieve dataset online: {e}")
        print("Generating a statistically accurate synthetic advertising dataset...")
        
        np.random.seed(42)
        n_samples = 200
        tv = np.random.uniform(0.7, 296.4, n_samples)
        radio = np.random.uniform(0.0, 49.6, n_samples)
        newspaper = np.random.uniform(0.3, 114.0, n_samples)
        
        # Real Advertising dataset regression: Sales = 4.3 + 0.054*TV + 0.107*Radio + 0.003*Newspaper + noise
        noise = np.random.normal(0, 1.5, n_samples)
        sales = 4.3 + (0.054 * tv) + (0.107 * radio) + (0.003 * newspaper) + noise
        sales = np.clip(sales, 1.6, None)
        
        df = pd.DataFrame({
            'TV': tv,
            'Radio': radio,
            'Newspaper': newspaper,
            'Sales': sales
        }).round(1)
        
        df.to_csv(DATA_PATH, index=False)
        print(f"Fallback dataset saved to: {DATA_PATH}")
        return df

def clean_and_preprocess(df):
    """
    Cleans column names (standardizing to TV, Radio, Newspaper, Sales),
    drops extra index/ID columns, and handles nulls/duplicates.
    """
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING & PREPROCESSING")
    print("=" * 60)
    
    # Normalize column names case-insensitively
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower == 'tv':
            rename_map[col] = 'TV'
        elif col_lower == 'radio':
            rename_map[col] = 'Radio'
        elif col_lower == 'newspaper':
            rename_map[col] = 'Newspaper'
        elif col_lower == 'sales':
            rename_map[col] = 'Sales'
            
    df = df.rename(columns=rename_map)
    
    # Drop non-standard columns (e.g. Unnamed: 0, X1, ID)
    standard_cols = ['TV', 'Radio', 'Newspaper', 'Sales']
    available_standard = [col for col in standard_cols if col in df.columns]
    extra_cols = [col for col in df.columns if col not in standard_cols]
    
    if extra_cols:
        print(f"Dropping index/unnecessary columns: {extra_cols}")
        df = df[available_standard]
        
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\nChecking for Missing (Null) Values:")
    nulls = df.isnull().sum()
    print(nulls)
    if nulls.sum() > 0:
        df = df.dropna()
        print("Dropped rows containing missing values.")
    else:
        print("No missing values found.")
        
    print("\nChecking for Duplicate Rows:")
    dups = df.duplicated().sum()
    print(f"Found {dups} duplicate rows.")
    if dups > 0:
        df = df.drop_duplicates()
        print("Duplicate rows removed.")
        
    print("\nSummary Statistics of Cleaned Dataset:")
    print(df.describe())
    
    return df

def generate_plots(df):
    """
    Saves exploratory data analysis plots in the plots/ directory.
    """
    print("\n" + "=" * 60)
    print("STEP 3: GENERATING EXPLORATORY DATA VISUALIZATIONS")
    print("=" * 60)
    
    colors = ['#4f46e5', '#0ea5e9', '#10b981']
    
    # 1. Sales Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Sales'], kde=True, color=colors[0], bins=15, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Sales Volume', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Sales (in thousands of units)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    dist_path = os.path.join(PLOTS_DIR, 'sales_distribution.png')
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"Saved: {os.path.basename(dist_path)}")

    # 2. Scatter Plots with Trendlines
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    features = ['TV', 'Radio', 'Newspaper']
    for i, feature in enumerate(features):
        sns.regplot(
            data=df, x=feature, y='Sales', ax=axes[i],
            scatter_kws={'alpha': 0.6, 'color': colors[i]},
            line_kws={'color': '#dc2626', 'linewidth': 2}
        )
        axes[i].set_title(f'{feature} Spend vs Sales', fontsize=13, fontweight='bold')
        axes[i].set_xlabel(f'{feature} Advertising Spend ($k)', fontsize=11)
        if i == 0:
            axes[i].set_ylabel('Sales (thousand units)', fontsize=11)
            
    plt.suptitle('Advertising Budget Channels vs Sales Revenue', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    scatter_path = os.path.join(PLOTS_DIR, 'advertising_vs_sales_scatter.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.basename(scatter_path)}")

    # 3. Correlation Heatmap
    plt.figure(figsize=(7, 6))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".3f", cmap="Blues",
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    heatmap_path = os.path.join(PLOTS_DIR, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved: {os.path.basename(heatmap_path)}")

def train_and_evaluate(df):
    """
    Fits a Linear Regression model, calculates metrics, saves comparison
    and diagnostics plots.
    """
    print("\n" + "=" * 60)
    print("STEP 4: MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    # Split features and target
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("\n" + "-" * 40)
    print("        Model Evaluation Metrics        ")
    print("-" * 40)
    print(f"R-squared (R2) Score:  {r2:.5f}  (Explains {r2*100:.2f}% of test variance)")
    print(f"Mean Absolute Error:   {mae:.5f}")
    print(f"Root Mean Sq. Error:   {rmse:.5f}")
    print("-" * 40)
    
    # Coefficients & Intercept
    intercept = model.intercept_
    coefs = model.coef_
    
    print("\nModel Parametric Coefficients:")
    print(f"  Intercept (Beta 0):       {intercept:.5f}")
    print(f"  TV Coefficient (Beta 1):  {coefs[0]:.5f}")
    print(f"  Radio Coeff. (Beta 2):    {coefs[1]:.5f}")
    print(f"  Newspaper Coeff. (Beta 3): {coefs[2]:.5f}")
    
    print(f"\nLinear Regression Formula:")
    print(f"  Sales = {intercept:.4f} + ({coefs[0]:.4f} * TV) + ({coefs[1]:.4f} * Radio) + ({coefs[2]:.4f} * Newspaper)")
    
    # Output sample predictions
    compare_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': y_test - y_pred
    })
    print("\nSample Predictions (First 5 Test Data Rows):")
    print(compare_df.head().to_string())
    
    # 4. Actual vs Predicted Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, color='#0ea5e9', alpha=0.7, edgecolors='k')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=2, label='Perfect Fit')
    plt.title('Actual vs Predicted Sales Volume', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Actual Sales', fontsize=12)
    plt.ylabel('Predicted Sales', fontsize=12)
    plt.legend()
    plt.tight_layout()
    act_path = os.path.join(PLOTS_DIR, 'actual_vs_predicted.png')
    plt.savefig(act_path, dpi=300)
    plt.close()
    print(f"\nSaved: {os.path.basename(act_path)}")

    # 5. Residuals Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred, residuals, color='#10b981', alpha=0.7, edgecolors='k')
    plt.axhline(y=0, color='#dc2626', linestyle='--', linewidth=2)
    plt.title('Residuals vs Predicted Sales', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Sales', fontsize=12)
    plt.ylabel('Residual (Error)', fontsize=12)
    plt.tight_layout()
    resid_path = os.path.join(PLOTS_DIR, 'residual_plot.png')
    plt.savefig(resid_path, dpi=300)
    plt.close()
    print(f"Saved: {os.path.basename(resid_path)}")

if __name__ == "__main__":
    df = load_or_generate_data()
    df_cleaned = clean_and_preprocess(df)
    generate_plots(df_cleaned)
    train_and_evaluate(df_cleaned)
    print("\nSales Prediction modeling finished successfully!")
