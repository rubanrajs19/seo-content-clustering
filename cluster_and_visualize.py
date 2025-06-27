import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt

# Load embedded content
df = pd.read_pickle("embedded_content.pkl")
df = df[df['embedding'].notnull()].reset_index(drop=True)

# Convert to numpy matrix
X = np.vstack(df['embedding'].values)

# Perform clustering (you can adjust n_clusters)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Reduce dimensions for visualization using UMAP
reducer = umap.UMAP(random_state=42)
embedding_2d = reducer.fit_transform(X)

df['x'] = embedding_2d[:, 0]
df['y'] = embedding_2d[:, 1]

# Plot clusters
plt.figure(figsize=(10, 6))
for cluster in range(n_clusters):
    clustered = df[df['cluster'] == cluster]
    plt.scatter(clustered['x'], clustered['y'], label=f"Cluster {cluster}", alpha=0.7)

plt.title("Content Clustering (UMAP + KMeans)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("content_clusters.png")
plt.show()

# Save clustered data to CSV
df[['Address', 'BodyText', 'cluster', 'x', 'y']].to_csv("clustered_content.csv", index=False)
print("✅ Clustering done. Results saved to 'clustered_content.csv' and 'content_clusters.png'")
