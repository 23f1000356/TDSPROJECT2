!pip install pandas matplotlib --quiet


import pandas as pd

from google.colab import files
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_csv('goodreads.csv')


print("Dataset Preview:\n", df.head())
print("\nDataset Information:\n")
df.info()

print("\nSummary Statistics (Numerical Columns):\n", df.describe())


missing_values = df.isnull().sum()
print("\nMissing Values (Count per Column):\n", missing_values)

missing_percentage = (df.isnull().mean() * 100).round(2)
print("\nMissing Values (Percentage per Column):\n", missing_percentage)


#outlier analysis
from google.colab import files
uploaded = files.upload()  
file_name = list(uploaded.keys())[0]  
df = pd.read_csv('goodreads.csv')


numeric_data = df.select_dtypes(include=[np.number])


print("\n--- Outliers using Z-Score Method ---")
z_scores = np.abs(zscore(numeric_data.dropna()))
threshold = 3
outliers_zscore = (z_scores > threshold).sum(axis=0)
print("Number of outliers detected per column:\n", outliers_zscore)

print("\n--- Outliers using IQR Method ---")
Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = ((numeric_data < lower_bound) | (numeric_data > upper_bound)).sum()
print("Number of outliers detected per column:\n", outliers_iqr)

print("\n--- Boxplot Visualization ---")
for column in numeric_data.columns:
    sns.boxplot(x=numeric_data[column])
    plt.title(f'Boxplot for {column}')
    plt.show()


total_outliers = pd.DataFrame({
    "Z-Score Outliers": outliers_zscore,
    "IQR Outliers": outliers_iqr
})
print("\nSummary of Outliers Detected:\n", total_outliers)

z_scores_df = pd.DataFrame(z_scores, columns=numeric_data.columns, index=numeric_data.index)

print("\n--- Anomalous Rows Detected Using Z-Score ---")
anomalous_rows = numeric_data[(z_scores_df > threshold).any(axis=1)]
print(f"Total Anomalous Rows Detected: {len(anomalous_rows)}")
if not anomalous_rows.empty:
    display(anomalous_rows)

