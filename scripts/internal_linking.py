import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load Data ----------
print("🔄 Loading embeddings...")
keyword_df = pd.read_csv("keyword_embeddings.csv")
site_df = pd.read_pickle("embedded_content.pkl")

# ---------- Parse Embeddings ----------
def parse_embedding(emb_str):
    try:
        return np.array([float(x) for x in emb_str.strip("[]").split(",")])
    except:
        return None

keyword_df['embedding'] = keyword_df['embedding'].apply(parse_embedding)
site_df['embedding'] = site_df['embedding'].apply(lambda x: np.array(x) if isinstance(x, list) else None)

keyword_df.dropna(subset=['embedding'], inplace=True)
site_df.dropna(subset=['embedding'], inplace=True)

# ---------- Normalize ----------
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

keyword_df['normalized'] = list(normalize(np.vstack(keyword_df['embedding'])))
site_df['normalized'] = list(normalize(np.vstack(site_df['embedding'])))

# ---------- Find Top N Neighbors ----------
print("🔗 Generating internal link suggestions...")
N = 5
internal_links = []

for idx, row in keyword_df.iterrows():
    kw_vec = np.array(row['normalized'])
    current_url = row['URL']

    # Compute cosine similarity with all site pages
    sims = cosine_similarity([kw_vec], np.vstack(site_df['normalized'].values))[0]

    # Build a temporary dataframe for ranking
    temp_df = site_df.copy()
    temp_df['similarity'] = sims

    # Exclude self
    temp_df = temp_df[temp_df['Address'] != current_url]

    # Sort by similarity and pick top N
    top_matches = temp_df.sort_values(by='similarity', ascending=False).head(N)
    internal_links.append(top_matches['Address'].tolist())

# ---------- Save ----------
keyword_df['InternalLinkSuggestions'] = internal_links
keyword_df[['Keyword', 'URL', 'InternalLinkSuggestions']].to_excel("keyword-internal-link-mapping.xlsx", index=False)

print("✅ Saved: keyword-internal-link-mapping.xlsx")
