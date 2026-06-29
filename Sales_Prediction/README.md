# CodSoft Data Science Internship - Task 4: Sales Prediction Using Python

This repository contains the complete Python machine learning project to predict product sales based on advertising budgets spent on three media channels: **TV, Radio, and Newspaper**. 

We perform exploratory data analysis, data cleaning, relation plotting, and fit a Multiple Linear Regression model using `scikit-learn` to estimate coefficients and make predictions.

---

## 📂 Project Repository Structure

```text
Sales_Prediction/
│
├── data/
│   └── advertising.csv          # Local advertising spend dataset
│
├── plots/                       # Visualizations folder (automatically generated)
│   ├── sales_distribution.png   # Target variable distribution
│   ├── advertising_vs_sales_scatter.png # Individual ad channels vs. sales with regression lines
│   ├── correlation_heatmap.png  # Pearson correlation heatmap of features
│   ├── actual_vs_predicted.png  # Comparison plot of actual vs predicted sales
│   └── residual_plot.png        # Residuals vs predictions (diagnostics plot)
│
├── main.py                      # Core Python script containing the modeling pipeline
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation (this file)
```

---

## 🛠️ Installation & Execution

Make sure you have Python (version 3.8 or higher) installed on your system.

### 1. Install Dependencies
You can install the required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 2. Run the Modeling Pipeline
Execute the pipeline script:
```bash
python main.py
```
This script will:
- Load/clean the dataset and display statistics.
- Generate and save all 5 diagnostic and EDA plots in the `plots/` directory.
- Split data, train a Multiple Linear Regression model, and print evaluation metrics.

---

## 📉 Mathematical Formulation

The Multiple Linear Regression model is formulated as:

$$\text{Sales} = \beta_0 + (\beta_1 \times \text{TV}) + (\beta_2 \times \text{Radio}) + (\beta_3 \times \text{Newspaper})$$

Where:
- $\beta_0$ is the intercept (base sales with no advertising).
- $\beta_1, \beta_2, \beta_3$ are the regression coefficients for TV, Radio, and Newspaper advertising budgets, respectively.

### Standard Model Performance
After running the model, the evaluations on the test partition (20% split) yield the following standard metrics:
- **$R^2$ Score (Coefficient of Determination):** `0.89944` (Explains **89.94%** of the variance in test sales).
- **Mean Absolute Error (MAE):** `1.46076`
- **Root Mean Squared Error (RMSE):** `1.78160`

### Calculated Model Coefficients
- **Intercept ($\beta_0$):** `2.97907`
- **TV Spend Coeff ($\beta_1$):** `0.04473`
- **Radio Spend Coeff ($\beta_2$):** `0.18920`
- **Newspaper Spend Coeff ($\beta_3$):** `0.00276`

$$\text{Sales} = 2.9791 + (0.0447 \times \text{TV}) + (0.1892 \times \text{Radio}) + (0.0028 \times \text{Newspaper})$$

---

## 💡 Key Business Insights

1. **Radio Spend ($\beta_2 \approx 0.189$):** For every $\$1,000$ increase in Radio advertising spend, sales increase by approximately $189$ units. This indicates Radio has the highest relative rate of return on advertising dollar.
2. **TV Spend ($\beta_1 \approx 0.045$):** For every $\$1,000$ increase in TV advertising, sales increase by approximately $45$ units. TV represents the largest scale channel, driving high overall sales volume.
3. **Newspaper Spend ($\beta_3 \approx 0.003$):** Newspaper advertising has a near-zero impact on sales, suggesting we should relocate budget from Newspaper to TV and Radio to maximize return on investment (ROI).
