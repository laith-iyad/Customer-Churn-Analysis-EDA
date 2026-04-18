# Customer Churn Analysis — EDA

This project is an Exploratory Data Analysis (EDA) on a customer churn dataset. The goal is to understand what kind of customers are more likely to churn and what features are most related to churn behavior. Everything is done in Python using Pandas, Matplotlib, and Seaborn.

---

## What the project does

The analysis is split into 6 main tasks:

### Task 1 — Data Loading and Initial Inspection
Loads `customer_data.csv` and prints the first few rows, data types, and basic statistics to get a first look at the data.

### Task 2 — Handling Missing Data
Checks for missing values across all columns. Turns out 643 rows had at least one missing value. Missing values in `Age`, `Income`, `Tenure`, and `SupportCalls` are filled with the **median** of each column (median is used instead of mean to avoid the influence of outliers).

### Task 3 — Handling Outliers
For the same four numerical columns, outliers are detected using the **2 standard deviations rule** — any value more than 2 std away from the mean is considered an outlier and removed. The number of detected outliers is printed for each column.

### Task 4 — Feature Scaling
Applies **Z-score normalization** (standardization) to `Age`, `Income`, `Tenure`, and `SupportCalls` — subtracts the mean and divides by the std. This puts all features on the same scale.

### Task 5 — EDA & Correlation Analysis
This is the main part. It includes:
- Histograms with KDE for the four numerical features
- Count plots for categorical features (`Gender`, `ProductType`, `ChurnStatus`)
- Box plots of each numerical feature vs `ChurnStatus` to see how distributions differ between churned and non-churned customers
- Count plots of `Gender` and `ProductType` vs `ChurnStatus`
- A **correlation heatmap** between all numerical features and the target (`ChurnStatus`)
- Correlations with ChurnStatus sorted by absolute value, printed to terminal

### Task 6 — Advanced Visualizations
A few more detailed plots:
- **Pair plot** of all numerical features colored by ChurnStatus
- **Average Income and Tenure by Product Type** (grouped bar chart)
- **Churn Rate by Gender and Product Type**
- **Churn Rate by Income Quartile** — customers split into 4 income groups, churn rate calculated for each
- **Churn Distribution by Income Quartile** — heatmap showing stayed vs churned percentages
- Same two plots repeated for **Tenure Quartile**

---

## Dataset

The dataset is `customer_data.csv` and it contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| Age | Numerical | Customer age |
| Income | Numerical | Customer income |
| Tenure | Numerical | How long they've been a customer |
| SupportCalls | Numerical | Number of support calls made |
| Gender | Categorical | Male / Female |
| ProductType | Categorical | Type of product subscribed to |
| ChurnStatus | Target | 0 = Stayed, 1 = Churned |

---

## How to run

### Install dependencies
```bash
pip install pandas matplotlib seaborn
```

### Run the script
```bash
python Assignment1.py
```

Make sure `customer_data.csv` is in the same directory as the script. All plots will pop up one by one as the script runs.

---

## Requirements

- Python 3.7+
- pandas
- matplotlib
- seaborn
