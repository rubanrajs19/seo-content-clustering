import pandas as pd
import numpy as np

# ---------- Load Data ----------
print("Loading embedded_content.pkl and keyword_embeddings.csv...")
site_df = pd.read_pickle("embedded_content.pkl")
keyword_df = pd.read_csv("keyword_embeddings.csv")

# ---------- Parse stringified embeddings ----------
def parse_embedding(emb_string):
    try:
        return np.array([float(val) for val in emb_string.strip("[]").split(",")])
    except Exception as e:
        print("Failed to parse embedding:", e)
        return None

keyword_df['embedding'] = keyword_df['embedding'].apply(parse_embedding)
site_df['embedding'] = site_df['embedding'].apply(lambda x: np.array(x) if isinstance(x, list) else None)

# Drop rows with bad embeddings
keyword_df = keyword_df[keyword_df['embedding'].notnull()].reset_index(drop=True)
site_df = site_df[site_df['embedding'].notnull()].reset_index(drop=True)

# ---------- Normalize embeddings ----------
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).tolist()

keyword_df['normalized'] = normalize_embeddings(np.vstack(keyword_df['embedding']))
site_df['normalized'] = normalize_embeddings(np.vstack(site_df['embedding']))

# ---------- Compute cosine similarity ----------
def cosine_similarity(v1, v2):
    return np.dot(v1, v2)

similarities = []
relevances = []

for i, row in keyword_df.iterrows():
    url = row['URL']
    kw_vec = np.array(row['normalized'])

    match = site_df[site_df['Address'] == url]
    if not match.empty:
        page_vec = np.array(match.iloc[0]['normalized'])
        sim = cosine_similarity(kw_vec, page_vec)
        rel = sim * 100
    else:
        sim = None
        rel = None

    similarities.append(sim)
    relevances.append(rel)

keyword_df['similarity'] = similarities
keyword_df['relevance'] = relevances

# ---------- Save ----------
keyword_df.to_csv("keyword_relevance_output.csv", index=False)
print("Saved: keyword_relevance_output.csv")
