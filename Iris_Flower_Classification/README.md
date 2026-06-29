# Iris Flower Classification

This project classifies Iris flowers into three species (Setosa, Versicolor, and Virginica) based on their sepal and petal measurements. It is built as part of the CodSoft Data Science Internship (Task 3).

## Objective
To train a machine learning model that predicts the species of an Iris flower based on its measurements.

## Dataset
We use the built-in Iris dataset from Scikit-Learn. It contains 150 samples with the following features:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## Technologies Used
- Python 3
- Pandas & NumPy
- Matplotlib & Seaborn (for visualizations)
- Scikit-Learn (for training the Random Forest classifier)

## Project Workflow
1. Load the dataset and convert it into a Pandas DataFrame.
2. Print the first 5 rows, dataset info, and summary statistics.
3. Save visualizations (`countplot.png`, `pairplot.png`, `heatmap.png`) in the `images/` folder.
4. Split the data into 80% training and 20% testing sets.
5. Train a Random Forest Classifier.
6. Print the model accuracy and classification report.
7. Test the model with a custom flower sample.

## How to Run

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:
   ```bash
   python iris_classification.py
   ```

## Results
The Random Forest model yields a test accuracy of around **90%** to **100%**. All generated plots are saved inside the `images/` folder.
