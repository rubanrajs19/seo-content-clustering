import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import os
import openai
from tqdm import tqdm
from local_ollama_embed import get_mistral_embedding  # Must exist and return a float list

# ---------- Step 1: Load content ----------
print("Loading content.csv...")
df = pd.read_csv("../content.csv", encoding='ISO-8859-1')

# Validate required columns
required_cols = ["BodyText", "Address"]
for col in required_cols:
    if col not in df.columns:
        print(f"Missing required column: {col}")
        exit(1)

# ---------- Step 2: Load OpenAI Key ----------
openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
if openai_key:
    openai.api_key = openai_key
    print("OpenAI API key detected.")
else:
    print(" OpenAI API key not set. Will fallback to Ollama or Hugging Face.")

# ---------- Step 3: Define embedding function ----------
def get_embedding(text):
    # Try OpenAI first
    if openai_key:
        try:
            print("Using OpenAI Embeddings...")
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"OpenAI failed. Falling back to Ollama. Reason: {e}")

    # Try Ollama + Mistral
    emb = get_mistral_embedding(text)
    if emb is not None:
        print("Using Ollama Embeddings (sfr-embedding-mistral)...")
        return emb

    # Fallback: Hugging Face
    # print("Using Hugging Face MiniLM Embeddings...")
    # tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    # model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    # inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # with torch.no_grad():
    #     outputs = model(**inputs)
    # return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# ---------- Step 4: Generate embeddings ----------
print("Generating embeddings...")
embeddings = []
EXPECTED_DIM = 4096 
for text in tqdm(df['BodyText'].fillna('')):
    try:
        emb = get_embedding(text)
        if isinstance(emb, list) and len(emb) == EXPECTED_DIM:
            embeddings.append(emb)
        else:
            print("⚠️ Invalid embedding shape. Falling back to zero vector.")
            embeddings.append([0.0] * EXPECTED_DIM)
    except Exception as e:
        print("Embedding error:", e)
        embeddings.append([0.0] * EXPECTED_DIM)

# df['embedding'] = embeddings
df['embedding'] = [list(e) if isinstance(e, (np.ndarray, list)) else [0.0] * 4096 for e in embeddings]
df = df[df['embedding'].notnull()].reset_index(drop=True)
df.to_pickle("embedded_content.pkl")

if df.empty:
    print("No embeddings generated. Exiting.")
    exit(1)

print(f"{len(df)} valid embeddings generated.")

# ---------- Step 5: Normalize embeddings ----------
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

# ---------- Step 6: Cosine similarity ----------
print("Computing cosine similarity matrix...")
similarity_matrix = cosine_similarity(embedding_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=df['Address'], columns=df['Address'])
similarity_df.to_csv("similarity_matrix.csv")
print("Saved: similarity_matrix.csv")

# ---------- Step 7: KMeans clustering ----------
print("Running KMeans clustering...")
n_clusters = 4  # You can make this configurable
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(embedding_matrix)

# ---------- Step 8: UMAP Visualization ----------
print("Reducing dimensions with UMAP...")
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

# ---------- Step 9: Save outputs ----------
df.drop(columns=['normalized_embedding'], inplace=True)
df.to_csv("clustered_content.csv", index=False)
df.to_pickle("embedded_content.pkl")
print("Pipeline complete! Files created:")
print("- embedded_content.pkl")
print("- similarity_matrix.csv")
print("- clustered_content.csv")
print("- content_clusters.png")
