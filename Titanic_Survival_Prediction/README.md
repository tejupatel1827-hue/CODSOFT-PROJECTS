# Titanic Survival Prediction (CodSoft Task 1)

This repository contains a complete Python-based machine learning project to predict the survival of passengers on the Titanic. This is the implementation for **Task 1: Titanic Survival Prediction** of the **CodSoft Data Science Internship**.

## 📌 Project Objective
The objective of this project is to build a classification model that predicts whether a passenger survived the Titanic shipwreck based on their personal attributes (such as age, gender, ticket class, fare, and family size). We use the classic Titanic dataset and implement **Logistic Regression** as the classification algorithm.

## 🛠️ Technologies Used
- **Language:** Python 3.x
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn
- **Package Management:** pip

## 📂 Project Structure
```text
titanic_survival_prediction/
│
├── data/
│   └── titanic.csv                  # Dataset (automatically downloaded)
│
├── plots/                           # Generated visualizations
│   ├── survival_by_gender.png       # Gender vs Survival count
│   ├── survival_by_class.png        # Class vs Survival count
│   ├── age_distribution.png         # Age distribution by survival status
│   ├── correlation_heatmap.png      # Feature correlation matrix
│   └── confusion_matrix.png         # Model confusion matrix
│
├── main.py                          # Complete python execution script
├── requirements.txt                 # Python dependencies
└── README.md                        # Documentation
```

## ⚙️ Data Preprocessing & Cleaning Workflow
1. **Handling Missing Values:**
   - **Age:** Replaced missing age values with the median age of the dataset (28 years).
   - **Embarked:** Replaced missing embarkation ports with the mode ('S').
   - **Cabin:** Dropped because more than 75% of the data in this column was missing.
2. **Feature Selection:**
   - Removed non-predictive columns (`PassengerId`, `Name`, `Ticket`, `Cabin`) that don't contribute to passenger survival.
3. **Categorical Encoding:**
   - **Sex:** Converted categorical `Sex` column into binary numerical values (male -> 0, female -> 1).
   - **Embarked:** Encoded categorical ports using one-hot encoding, creating dummy variables (`Embarked_Q`, `Embarked_S`) and dropping the first category to avoid multicollinearity.
4. **Feature Scaling:**
   - Standardized all independent features using `StandardScaler` to ensure the Logistic Regression model converges properly and feature weights are directly comparable.

## 📈 Model Performance & Evaluation

The Logistic Regression model was trained on **80%** of the data and tested on the remaining **20%** (stratified split to maintain class balance).

### Key Results
- **Model Accuracy:** **80.45%**

### Confusion Matrix
```text
[[98 12]  --> [True Deceased, False Survived]
 [23 46]] --> [False Deceased, True Survived]
```
- **True Negatives (Deceased correctly predicted):** 98
- **False Positives (Deceased predicted as Survived):** 12
- **False Negatives (Survived predicted as Deceased):** 23
- **True Positives (Survived correctly predicted):** 46

### Classification Report
```text
              precision    recall  f1-score   support

    Deceased       0.81      0.89      0.85       110
    Survived       0.79      0.67      0.72        69

    accuracy                           0.80       179
   macro avg       0.80      0.78      0.79       179
weighted avg       0.80      0.80      0.80       179
```

### Model Coefficients & Interpretation
The coefficients representing the relative impact of each scaled feature on survival probability:

| Feature | Coefficient | Interpretation |
| :--- | :--- | :--- |
| **Sex** | 1.270 | Positive coefficient. Females (coded as 1) had significantly higher chances of survival. |
| **Fare** | 0.100 | Slightly positive. Passengers who paid higher fares had a marginally better chance of survival. |
| **Embarked_Q** | 0.081 | Slightly positive effect of embarking at Queenstown compared to Cherbourg. |
| **Parch** | -0.067 | Slightly negative. Having parents/children aboard slightly reduced survival log-odds. |
| **Embarked_S** | -0.172 | Negative. Embarking at Southampton was associated with lower survival odds compared to Cherbourg. |
| **SibSp** | -0.263 | Negative. Having more siblings/spouses aboard reduced survival probability. |
| **Age** | -0.503 | Negative. Older passengers had lower chances of survival (women and children first policy). |
| **Pclass** | -0.929 | Strong negative coefficient. Higher numerical ticket class (e.g., 3rd class vs 1st class) significantly decreased survival probability. |

## 🚀 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd titanic_survival_prediction
   ```

2. **Install requirements:**
   Make sure you have Python installed. Install the dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the script:**
   Run the model pipeline:
   ```bash
   python main.py
   ```
   The script will:
   - Check if `data/titanic.csv` exists and download it from the official source if missing.
   - Print dataset summaries.
   - Generate and save the exploratory visualizations to the `plots/` directory.
   - Clean the data and impute missing values.
   - Scale features, train a Logistic Regression model, print metrics, and display sample predictions.

## 🎯 Conclusion
The Logistic Regression model achieved a solid classification accuracy of **80.45%**.
- **Gender (Sex)** and **Passenger Class (Pclass)** were the most significant predictors of survival. Being female strongly increased survival odds, whereas traveling in 3rd class (higher numerical value of class) and being older significantly reduced the probability of survival. This aligns perfectly with historical accounts of the shipwreck ("women and children first" and priority boarding for upper class passengers).
