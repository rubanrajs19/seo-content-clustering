import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

EXPECTED_DIM = 4096  # match your embedding size

# ---------- Load Data Directly ----------
print("🔄 Loading site content and keyword embeddings...")
site_df = pd.read_pickle("embedded_content.pkl")
prospect_df = pd.read_csv("keyword_embeddings.csv")

# ---------- Parse Embeddings ----------
def parse_embedding(emb):
    try:
        if isinstance(emb, list):
            return np.array(emb) if len(emb) == EXPECTED_DIM else None
        elif isinstance(emb, str):
            emb = [float(x.strip()) for x in emb.strip("[]").split(",")]
            return np.array(emb) if len(emb) == EXPECTED_DIM else None
    except:
        return None

prospect_df['embedding'] = prospect_df['embedding'].apply(parse_embedding)
site_df['embedding'] = site_df['embedding'].apply(lambda x: np.array(x) if isinstance(x, list) and len(x) == EXPECTED_DIM else None)

# ---------- Filter Valid ----------
prospect_df.dropna(subset=['embedding'], inplace=True)
site_df.dropna(subset=['embedding'], inplace=True)

if site_df.empty or prospect_df.empty:
    print("❌ One or both datasets are empty after filtering. Check embedded_content.pkl and keyword_embeddings.csv.")
    exit()

# ---------- Normalize ----------
def normalize(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

site_mat = normalize(np.vstack(site_df['embedding']))
prospect_mat = normalize(np.vstack(prospect_df['embedding']))
site_df['normalized'] = list(site_mat)
prospect_df['normalized'] = list(prospect_mat)

# ---------- Compute Best Matches ----------
print("🔍 Matching each keyword prospect to best live URL...")
top_matches, top_scores = [], []

for vec in prospect_df['normalized']:
    sims = cosine_similarity([vec], site_mat)[0]
    best_idx = np.argmax(sims)
    top_matches.append(site_df.iloc[best_idx]['Address'])
    top_scores.append(sims[best_idx])

# ---------- Save Output ----------
prospect_df['BestMatchURL'] = top_matches
prospect_df['CosineSimilarity'] = top_scores

prospect_df[['Keyword', 'URL', 'BestMatchURL', 'CosineSimilarity']].to_csv("link-prospect-relevance.csv", index=False)
print("✅ Saved: link-prospect-relevance.csv")
