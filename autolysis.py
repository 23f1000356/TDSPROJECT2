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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore
import networkx as nx

# Load your dataset
data = pd.read_csv('happiness.csv')

# --- Outlier and Anomaly Detection ---
def detect_outliers(data, columns):
    for col in columns:
        z_scores = zscore(data[col])
        outliers = data[np.abs(z_scores) > 3]
        print(f"Outliers in {col}:\n", outliers)

# Specify columns for outlier detection
outlier_columns = ['column1', 'column2']
detect_outliers(data, outlier_columns)

# --- Correlation and Regression Analysis ---
# Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Regression Analysis
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']
reg_model = LinearRegression()
reg_model.fit(X, y)
print("Regression Coefficients:", reg_model.coef_)

# Feature Importance
rf_model = RandomForestRegressor()
rf_model.fit(X, y)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Feature Importance:\n", feature_importance)

# --- Time Series Analysis ---
# Ensure the dataset has a datetime index
data['date_column'] = pd.to_datetime(data['date_column'])
data.set_index('date_column', inplace=True)

# Decompose Time Series
decomposition = seasonal_decompose(data['time_series_column'], model='additive')
decomposition.plot()
plt.show()

# --- Cluster Analysis ---
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['feature1', 'feature2', 'feature3']])
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize Clusters
sns.scatterplot(x='feature1', y='feature2', hue='Cluster', data=data, palette='viridis')
plt.title("Cluster Analysis")
plt.show()

# --- Geographic Analysis ---
# Assuming the dataset contains latitude and longitude
sns.scatterplot(x='longitude', y='latitude', hue='target', data=data, palette='coolwarm')
plt.title("Geographic Analysis")
plt.show()

# --- Network Analysis ---
# Create a NetworkX graph (assuming 'source' and 'target' columns in the dataset)
G = nx.from_pandas_edgelist(data, source='source', target='target', edge_attr=True)
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray')
plt.title("Network Analysis")
plt.show()


