import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set seaborn style for beautiful visualizations
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Define directories
DATA_DIR = 'data'
PLOTS_DIR = 'plots'
DATASET_PATH = os.path.join(DATA_DIR, 'titanic.csv')
DATASET_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def setup_directories():
    """Create necessary directories if they don't exist."""
    for directory in [DATA_DIR, PLOTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def download_dataset():
    """Download Titanic dataset if not locally available."""
    if not os.path.exists(DATASET_PATH):
        print(f"Downloading Titanic dataset from {DATASET_URL}...")
        try:
            response = requests.get(DATASET_URL)
            response.raise_for_status()
            with open(DATASET_PATH, 'wb') as f:
                f.write(response.content)
            print(f"Dataset downloaded successfully and saved to: {DATASET_PATH}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise e
    else:
        print(f"Dataset found locally at: {DATASET_PATH}")

def load_data():
    """Load the dataset into a pandas DataFrame."""
    df = pd.read_csv(DATASET_PATH)
    print("\n--- Dataset Summary ---")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nColumns and Data Types:")
    print(df.dtypes)
    return df

def perform_eda_and_visualizations(df):
    """Generate and save visualizations for EDA (Exploratory Data Analysis)."""
    print("\n--- Generating Visualizations ---")
    
    # 1. Survival Rate by Gender
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Sex', hue='Survived', palette='Set2')
    plt.title('Survival Count by Gender', fontsize=14, fontweight='bold')
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(['Deceased', 'Survived'], loc='upper right')
    plot_path_1 = os.path.join(PLOTS_DIR, 'survival_by_gender.png')
    plt.savefig(plot_path_1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path_1}")

    # 2. Survival Rate by Passenger Class (Pclass)
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Pclass', hue='Survived', palette='viridis')
    plt.title('Survival Count by Ticket Class (Pclass)', fontsize=14, fontweight='bold')
    plt.xlabel('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(['Deceased', 'Survived'], loc='upper right')
    plot_path_2 = os.path.join(PLOTS_DIR, 'survival_by_class.png')
    plt.savefig(plot_path_2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path_2}")

    # 3. Age Distribution of Survivors vs Non-Survivors
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', hue='Survived', kde=True, multiple='stack', palette='coolwarm', bins=30)
    plt.title('Age Distribution by Survival Status', fontsize=14, fontweight='bold')
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(['Survived', 'Deceased'], loc='upper right')  # Stack order is Deceased first then Survived
    plot_path_3 = os.path.join(PLOTS_DIR, 'age_distribution.png')
    plt.savefig(plot_path_3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path_3}")

    # 4. Correlation Heatmap (Numerical features)
    plt.figure(figsize=(10, 8))
    # Select numerical columns for correlation mapping
    numerical_df = df.select_dtypes(include=[np.number]).copy()
    # Let's map Sex to binary in a temp df for correlation visualization
    numerical_df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})
    corr_matrix = numerical_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Features', fontsize=14, fontweight='bold')
    plot_path_4 = os.path.join(PLOTS_DIR, 'correlation_heatmap.png')
    plt.savefig(plot_path_4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path_4}")

def preprocess_data(df):
    """Clean and preprocess the dataset for machine learning."""
    print("\n--- Data Preprocessing ---")
    
    # Check for missing values
    print("Missing values count before imputation:")
    print(df.isnull().sum())
    
    # Impute missing Age with the median age
    age_median = df['Age'].median()
    df['Age'] = df['Age'].fillna(age_median)
    print(f"Imputed missing 'Age' values with median: {age_median}")
    
    # Impute missing Embarked with the mode (most common port)
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    print(f"Imputed missing 'Embarked' values with mode: '{embarked_mode}'")
    
    # Drop Cabin column because it has too many missing values (>75% missing)
    # Also drop PassengerId, Name, and Ticket as they do not provide predictive value
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    print(f"Dropped non-predictive columns: {columns_to_drop}")
    
    # Convert categorical variables into numerical
    # Map Sex: male -> 0, female -> 1
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    print("Converted 'Sex' column: male -> 0, female -> 1")
    
    # One-hot encode the Embarked column (C, Q, S)
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    print("One-hot encoded 'Embarked' column (created dummy columns, dropping the first).")
    
    print("\nMissing values count after imputation:")
    print(df.isnull().sum())
    
    print("\nProcessed Dataset Sample:")
    print(df.head())
    
    return df

def train_and_evaluate(df):
    """Split the data, train Logistic Regression, and evaluate the model."""
    print("\n--- Model Training & Evaluation ---")
    
    # Split data into Features (X) and Target (y)
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Split into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing set size: X={X_test.shape}, y={y_test.shape}")
    
    # Scale the features (essential for regularized Logistic Regression convergence)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Scaled features using StandardScaler.")
    
    # Initialize and Train Logistic Regression Model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    print("Logistic Regression model trained successfully.")
    
    # Make predictions on test set
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot and Save Confusion Matrix Visualization
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Deceased', 'Survived'], 
                yticklabels=['Deceased', 'Survived'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plot_path_cm = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(plot_path_cm, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot: {plot_path_cm}")
    
    # Generate Classification Report
    report = classification_report(y_test, y_pred, target_names=['Deceased', 'Survived'])
    print("\nClassification Report:")
    print(report)
    
    # Display Sample Predictions vs Actual Values
    sample_comparison = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Survival Probability': y_pred_proba
    })
    print("\nSample Predictions (First 10 records of Test Set):")
    print(sample_comparison.head(10))
    
    # Feature Importance (Coefficients)
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)
    print("\nFeature Coefficients (Model Interpretation):")
    print(coefficients.to_string(index=False))

def main():
    print("==========================================================")
    print("      TITANIC SURVIVAL PREDICTION SYSTEM (CodSoft Task 1)  ")
    print("==========================================================")
    
    setup_directories()
    download_dataset()
    raw_df = load_data()
    
    # Generate exploratory analysis visualizations using raw dataset
    perform_eda_and_visualizations(raw_df)
    
    # Preprocess the dataset
    processed_df = preprocess_data(raw_df)
    
    # Train and evaluate the model
    train_and_evaluate(processed_df)
    
    print("\n==========================================================")
    print("Project run complete. Visualizations saved in 'plots/' directory.")
    print("==========================================================")

if __name__ == '__main__':
    main()
