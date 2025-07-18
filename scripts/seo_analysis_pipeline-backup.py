import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- Step 1: Load content ----------
print("Loading content.csv...")
df = pd.read_csv("content.csv")

# ---------- Step 2: Generate embeddings ----------
print("Generating embeddings using Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

embeddings = []
for text in tqdm(df['BodyText'].fillna('')):
    try:
        emb = get_embedding(text)
        embeddings.append(emb)
    except Exception as e:
        print("Embedding error:", e)
        embeddings.append(None)

df['embedding'] = embeddings
df = df[df['embedding'].notnull()].reset_index(drop=True)

# ---------- Step 3: Normalize embeddings ----------
print("Normalizing embeddings...")
def normalize_embeddings(embeddings):
    normalized = []
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        if norm == 0:
            normalized.append(emb)
        else:
            normalized.append(emb / norm)
    return normalized

df['normalized_embedding'] = normalize_embeddings(df['embedding'].tolist())
embedding_matrix = np.vstack(df['normalized_embedding'].values)

# ---------- Step 4: Cosine similarity ----------
print("Computing cosine similarity matrix...")
similarity_matrix = cosine_similarity(embedding_matrix)
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=df['Address'],
    columns=df['Address']
)
similarity_df.to_csv("similarity_matrix.csv")
print("Saved: similarity_matrix.csv")

# ---------- Step 5: KMeans clustering ----------
print("Running KMeans clustering...")
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(embedding_matrix)

# ---------- Step 6: UMAP visualization ----------
print("Reducing dimensions with UMAP and plotting clusters...")
reducer = umap.UMAP(random_state=42)
embedding_2d = reducer.fit_transform(embedding_matrix)
df['x'] = embedding_2d[:, 0]
df['y'] = embedding_2d[:, 1]

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
print("Saved: content_clusters.png")

# ---------- Step 7: Save all outputs ----------
df.drop(columns=['normalized_embedding'], inplace=True)
df.to_csv("clustered_content.csv", index=False)
df.to_pickle("embedded_content.pkl")
print("Pipeline complete! Files created:")
print("- embedded_content.pkl")
print("- similarity_matrix.csv")
print("- clustered_content.csv")
print("- content_clusters.png")
