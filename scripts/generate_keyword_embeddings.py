import pandas as pd
from local_ollama_embed import get_mistral_embedding
import numpy as np
import csv

# Load keywords + URLs
df = pd.read_csv("keywords.csv")  # expects: Keyword, URL
embeddings = []

# Generate embeddings
for text in df['Keyword'].fillna(''):
    emb = get_mistral_embedding(text)
    if emb is not None:
        embeddings.append(emb)
    else:
        embeddings.append([0] * 4096)

# Convert embeddings to string before saving
df['embedding'] = [str(e) for e in embeddings]  # ✅ Make it a safe string

# Save with quoting to prevent Excel parsing issues
df.to_csv("keyword_embeddings.csv", index=False, quoting=csv.QUOTE_ALL)
print("✅ keyword_embeddings.csv saved.")
