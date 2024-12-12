import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the dataset
file_path = "goodreads.csv"
df = pd.read_csv(file_path)

# Convert 'original_publication_year' to numeric
df['original_publication_year'] = pd.to_numeric(df['original_publication_year'], errors='coerce')

# Summary Statistics
summary_stats = df.describe()
print("Summary Statistics:\n", summary_stats)

# Counting Missing Values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Correlation Matrix
numerical_columns = df.select_dtypes(include=["number"]).dropna()
correlation_matrix = numerical_columns.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.show()

# Outlier Detection
for col in numerical_columns.columns:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.savefig(f"boxplot_{col}.png")
    plt.close()

# Clustering
data_for_clustering = numerical_columns.dropna()
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = np.nan
df.loc[data_for_clustering.index, 'Cluster'] = kmeans.fit_predict(data_for_clustering)

# Plot KMeans Clustering
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data_for_clustering['average_rating'], y=data_for_clustering['ratings_count'], hue=df['Cluster'], palette='Set1', s=100, alpha=0.7)
plt.title("KMeans Clustering (3 Clusters)")
plt.xlabel('Average Rating')
plt.ylabel('Ratings Count')
plt.legend(title="Cluster")
plt.savefig("clustering.png")
plt.show()

# Hierarchical Clustering
linked = linkage(data_for_clustering, method='ward')
plt.figure(figsize=(12, 8))
dendrogram(linked, labels=data_for_clustering.index, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram")
plt.savefig("hierarchical_clustering.png")
plt.show()

# Time Series Analysis
if 'original_publication_year' in df.columns:
    time_series = df.groupby('original_publication_year')[['average_rating', 'ratings_count']].mean()
    time_series.dropna().plot(figsize=(12, 6), title="Trends Over Time")
    plt.savefig("time_series_trends.png")
    plt.show()




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("media.csv", encoding="latin1")

# Convert 'date' to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Summary Statistics
summary_stats = df.describe()
print("Summary Statistics:\n", summary_stats)

# Count Missing Values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Correlation Matrix
numerical_columns = df.select_dtypes(include=["number"])
correlation_matrix = numerical_columns.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.show()

# Outlier Detection
for col in numerical_columns:
    plt.figure()
    sns.boxplot(df[col])
    plt.title(f"Boxplot of {col}")
    plt.savefig(f"boxplot_{col}.png")
    plt.close()

# Clustering
data_for_clustering = numerical_columns.dropna()
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = np.nan
df.loc[data_for_clustering.index, 'Cluster'] = kmeans.fit_predict(data_for_clustering)

# Hierarchical Clustering
linked = linkage(data_for_clustering, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=data_for_clustering.index, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram")
plt.savefig("hierarchical_clustering.png")
plt.show()

# Time Series Analysis
if 'date' in df.columns:
    # Ensure 'date' is a datetime type
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Select only numeric columns for aggregation
    numeric_columns = df.select_dtypes(include=['number']).columns
    time_series = df.groupby(df['date'].dt.to_period('M'))[numeric_columns].mean()

    # Plot a specific column (e.g., 'overall') over time
    if 'overall' in numeric_columns:
        time_series.plot(y='overall', title="Overall Rating Over Time")
        plt.savefig("time_series_overall.png")
        plt.show()


