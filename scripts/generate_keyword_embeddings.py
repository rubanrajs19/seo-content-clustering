# generate_keyword_embeddings.py
import pandas as pd
from local_ollama_embed import get_mistral_embedding  # or OpenAI if preferred
import numpy as np

df = pd.read_csv("keywords.csv")  # columns: Keyword, URL
embeddings = []

for text in df['Keyword'].fillna(''):
    emb = get_mistral_embedding(text)  # or OpenAI
    if emb is not None:
        embeddings.append(emb)
    else:
        embeddings.append([0]*384)  # fallback zero-vector

df['embedding'] = embeddings
df.to_csv("keyword_embeddings.csv", index=False)
print("✅ keyword_embeddings.csv saved.")
