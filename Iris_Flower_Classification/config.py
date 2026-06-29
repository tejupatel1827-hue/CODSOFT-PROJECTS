"""
Configuration file containing directories, paths, and model parameters.
"""

import os

# Root directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Output Directories
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
EVAL_FIGURES_DIR = os.path.join(FIGURES_DIR, "evaluation")

# File Paths
MODEL_PATH = os.path.join(MODELS_DIR, "iris_random_forest.joblib")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")

# Hyperparameter search grid for RandomForestClassifier
GRID_SEARCH_PARAMS = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5],
    'criterion': ['gini', 'entropy']
}

# General configurations
RANDOM_STATE = 42
TEST_SIZE = 0.20
CROSS_VALIDATION_FOLDS = 5
