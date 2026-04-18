import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns




##############################################
#   Task #1:    Data Loading and Initial Inspection:

df = pd.read_csv('customer_data.csv')
print(df.head())
print(df.info())
print(df.describe())


###############################################
#   Task #2:    Handling Missing Data:

print (df.isnull().sum())
print(df.isnull().any(axis=1).sum())  # rows that have at least one missing value, count = 643

for col in ['Age', 'Income', 'Tenure', 'SupportCalls']:
    df[col].fillna(df[col].median(), inplace=True)

#print (df.isnull().sum())
#print(df.info())



###############################################


#   Task #3:    Handling Outliers:

for col in ['Age', 'Income', 'Tenure', 'SupportCalls']:
    upper = df[col].mean() + 2 * df[col].std()
    lower = df[col].mean() - 2 * df[col].std()
    outliers = df[(df[col] > upper) | (df[col] < lower)]
    print(f"\nOutliers detected in {col}: {len(outliers)}")

    df = df[(df[col] <= upper) & (df[col] >= lower)]



###############################################
#   Task #4:    Feature Scaling:

columns = ['Age', 'Income', 'Tenure', 'SupportCalls']

df[columns] = df[columns].apply(lambda x: (x - x.mean()) / x.std())
print(df)


###############################################
#   Task #5:    Exploratory Data Analysis (EDA): 

# **********************1*********************
# Numerical distributions

num_cols = ['Age', 'Income', 'Tenure', 'SupportCalls']
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()
    

# Categorical distributions
cat_cols = ['Gender', 'ProductType', 'ChurnStatus']
for col in cat_cols:
    plt.figure(figsize=(5,4))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.show()

# **********************2*********************
# Numerical vs Target (ChurnStatus)
num_cols = ['Age', 'Income', 'Tenure', 'SupportCalls']

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='ChurnStatus', y=col, data=df)
    plt.title(f'{col} vs ChurnStatus')
    plt.show()
# Categorical vs Target
cat_cols = ['Gender', 'ProductType']

for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, hue='ChurnStatus', data=df)
    plt.title(f'{col} vs ChurnStatus')
    plt.legend(title='ChurnStatus', labels=['Stayed (0)', 'Churned (1)'])
    plt.show()

print(df)




###############################################
#   Task #5:    Correlation Analysis:

# 1) Build a correlation matrix for numeric features (incl. target)
corr_features = ['Age', 'Income', 'Tenure', 'SupportCalls', 'ChurnStatus']
corr_matrix = df[corr_features].corr()

# 2) Visualize the correlation matrix as a heatmap
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
            linewidths=0.5, linecolor='white')
plt.title('Correlation Matrix (Numeric Features + ChurnStatus)')
plt.tight_layout()
plt.show()

# 3) Print correlations with the target, sorted by strength
target_corr = corr_matrix['ChurnStatus'].drop('ChurnStatus').sort_values(ascending=False, key=lambda s: s.abs())
print("\nCorrelation with ChurnStatus (sorted by absolute value):")
print(target_corr.to_frame('Correlation'))


###############################################
#   Task #6:    Data Visualizations:

# 1. Correlation Heatmap
plt.figure(figsize=(8,6))
corr = df[['Age', 'Income', 'Tenure', 'SupportCalls', 'ChurnStatus']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 2. Pair Plot for Numerical Features
sns.pairplot(df[['Age', 'Income', 'Tenure', 'SupportCalls', 'ChurnStatus']], hue='ChurnStatus', diag_kind='kde')
plt.suptitle('Pair Plot of Numerical Features vs ChurnStatus', y=1.02)
plt.show()

# 3. Average Income and Tenure by Product Type
plt.figure(figsize=(7,5))
avg_data = df.groupby('ProductType')[['Income','Tenure']].mean().reset_index()
avg_data = pd.melt(avg_data, id_vars='ProductType', value_vars=['Income','Tenure'], var_name='Feature', value_name='Average')
sns.barplot(x='ProductType', y='Average', hue='Feature', data=avg_data)
plt.title('Average Income and Tenure by Product Type')
plt.show()

# 4. Churn Rate by Gender and Product Type
plt.figure(figsize=(7,5))
churn_rate = df.groupby(['Gender','ProductType'])['ChurnStatus'].mean().reset_index()
sns.barplot(x='Gender', y='ChurnStatus', hue='ProductType', data=churn_rate)
plt.title('Churn Rate by Gender and Product Type')
plt.ylabel('Churn Rate')
plt.show()



###############################################
#   Task #6:    Data Visualizations (Income & Tenure vs Churn)

# -------- Income vs ChurnStatus --------
plt.figure(figsize=(7,5))
income_q = pd.qcut(df['Income'], q=4,
                   labels=['Q1 (Low)','Q2','Q3','Q4 (High)'],
                   duplicates='drop')
churn_by_income_q = df.groupby(income_q)['ChurnStatus'].mean().reset_index()
churn_by_income_q.rename(columns={'ChurnStatus':'ChurnRate', 'Income':'IncomeQuartile'}, inplace=True)
sns.barplot(x='Income', y='ChurnRate', data=churn_by_income_q.rename(columns={'IncomeQuartile':'Income'}))
plt.title('Churn Rate by Income Quartile')
plt.xlabel('Income Quartile')
plt.ylabel('Average Churn Rate')
plt.show()

plt.figure(figsize=(7,5))
income_churn_tab = pd.crosstab(income_q, df['ChurnStatus'], normalize='index').reindex(columns=[0,1]).round(3)
sns.heatmap(income_churn_tab, annot=True, cmap='coolwarm', fmt='.3f')
plt.title('Churn Distribution by Income Quartile')
plt.xlabel('ChurnStatus (0=Stayed, 1=Churned)')
plt.ylabel('Income Quartile')
plt.show()


# -------- Tenure vs ChurnStatus --------
plt.figure(figsize=(7,5))
tenure_q = pd.qcut(df['Tenure'], q=4,
                   labels=['Q1 (Short)','Q2','Q3','Q4 (Long)'],
                   duplicates='drop')
churn_by_tenure_q = df.groupby(tenure_q)['ChurnStatus'].mean().reset_index()
churn_by_tenure_q.rename(columns={'ChurnStatus':'ChurnRate', 'Tenure':'TenureQuartile'}, inplace=True)
sns.barplot(x='Tenure', y='ChurnRate', data=churn_by_tenure_q.rename(columns={'TenureQuartile':'Tenure'}))
plt.title('Churn Rate by Tenure Quartile')
plt.xlabel('Tenure Quartile')
plt.ylabel('Average Churn Rate')
plt.show()

plt.figure(figsize=(7,5))
tenure_churn_tab = pd.crosstab(tenure_q, df['ChurnStatus'], normalize='index').reindex(columns=[0,1]).round(3)
sns.heatmap(tenure_churn_tab, annot=True, cmap='coolwarm', fmt='.3f')
plt.title('Churn Distribution by Tenure Quartile')
plt.xlabel('ChurnStatus (0=Stayed, 1=Churned)')
plt.ylabel('Tenure Quartile')
plt.show()
