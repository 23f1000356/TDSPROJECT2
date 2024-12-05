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

