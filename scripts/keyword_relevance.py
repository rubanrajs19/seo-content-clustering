import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
from tqdm import tqdm
import os
import openai
from local_ollama_embed import get_mistral_embedding

# Load content with embeddings
print("Loading embedded_content.pkl...")
df = pd.read_pickle("embedded_content.pkl")

# Load GSC keyword data
print("Loading gsc_data.csv...")
gsc_df = pd.read_csv("gsc_data.csv")
gsc_df = gsc_df.dropna(subset=["Query", "Address"])
gsc_df['Query'] = gsc_df['Query'].astype(str)

# Load embedding model (fallback to MiniLM)
print("Loading fallback embedding model (MiniLM)...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
if openai_key:
    openai.api_key = openai_key
    print("OpenAI API key detected.")

# Define embedding logic
def get_embedding(text):
    if openai_key:
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"OpenAI error, using Ollama: {e}")
    emb = get_mistral_embedding(text)
    if emb:
        return emb
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Compute keyword relevance
print("Computing keyword relevance...")
relevance_data = []

for _, row in tqdm(gsc_df.iterrows(), total=len(gsc_df)):
    keyword = row['Query']
    page = row['Address']
    clicks = row['Clicks']

    if page not in df['Address'].values:
        continue

    page_embedding = df[df['Address'] == page]['embedding'].values[0]
    keyword_embedding = get_embedding(keyword)

    if page_embedding is not None and keyword_embedding is not None:
        page_vec = np.array(page_embedding).reshape(1, -1)
        keyword_vec = np.array(keyword_embedding).reshape(1, -1)
        similarity = cosine_similarity(page_vec, keyword_vec)[0][0]
        relevance_data.append({
            "Address": page,
            "Query": keyword,
            "Clicks": clicks,
            "RelevanceScore": similarity
        })

# Output
result_df = pd.DataFrame(relevance_data)
result_df.to_csv("keyword_relevance.csv", index=False)
print("Saved: keyword_relevance.csv")
