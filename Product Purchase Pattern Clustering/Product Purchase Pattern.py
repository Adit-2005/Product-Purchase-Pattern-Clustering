import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the dataset
file_path = r"D:\Product Purchase Pattern Clustering\Dataset\E-commerce Website Logs.csv"
df = pd.read_csv(file_path, low_memory=False)

# Clean 'age' column and drop unnecessary columns
df["age"] = pd.to_numeric(df["age"], errors='coerce')
df["age"] = df["age"].fillna(df["age"].median())
df.drop(columns=["accessed_date", "ip", "network_protocol"], inplace=True)
df = df.dropna()

# Encode categorical variables numerically
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = pd.factorize(df[col])[0]

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Elbow Method
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method with annotation
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.xticks(K_range)
optimal_k = 4
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f"Elbow at K={optimal_k}")
plt.annotate("Optimal K", xy=(optimal_k, wcss[optimal_k-1]), xytext=(optimal_k+1, wcss[optimal_k-1]+5000),
             arrowprops=dict(facecolor='black', arrowstyle="->"))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# K-Means clustering with optimal K
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Visualize clusters in 2D
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=clusters, palette='Set2', s=60)
plt.title("K-Means Clustering Visualization (PCA Reduced)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title='Cluster')
plt.annotate("Distinct cluster regions", xy=(pca_result[0, 0], pca_result[0, 1]),
             xytext=(pca_result[0, 0]+5, pca_result[0, 1]+5),
             arrowprops=dict(facecolor='black', arrowstyle="->"))
plt.grid(True)
plt.tight_layout()
plt.show()
