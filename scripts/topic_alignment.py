import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Step 1: Load existing embeddings ----------
print("Loading embedded_content.pkl...")
df = pd.read_pickle("embedded_content.pkl")

# ---------- Step 2: Load strategic topics from CSV ----------
print("Loading strategic topics from strategic_topics.csv...")
topic_df = pd.read_csv("strategic_topics.csv")

# Group into a dictionary: {topic: [keyword1, keyword2, ...]}
strategic_topics = topic_df.groupby('topic')['keyword'].apply(list).to_dict()

# ---------- Step 3: Load the embedding model ----------
print("Loading Hugging Face embedding model...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to embed text
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# ---------- Step 4: Generate topic centroids ----------
print("Generating centroids for each topic...")
topic_centroids = {}
for topic, keywords in strategic_topics.items():
    keyword_embeddings = [get_embedding(k) for k in keywords]
    topic_centroids[topic] = np.mean(keyword_embeddings, axis=0)

# ---------- Step 5: Normalize content embeddings ----------
print("Normalizing page embeddings...")
def normalize_embeddings(embeddings):
    return [e / np.linalg.norm(e) if np.linalg.norm(e) != 0 else e for e in embeddings]

df['normalized_embedding'] = normalize_embeddings(df['embedding'].tolist())

# ---------- Step 6: Calculate relevance to each topic ----------
print("Calculating cosine similarity to each topic...")
for topic, centroid in topic_centroids.items():
    scores = cosine_similarity(np.vstack(df['normalized_embedding'].values), centroid.reshape(1, -1))
    df[f'relevance_to_{topic}'] = scores

# ---------- Step 7: Save output ----------
df.to_pickle("topic_aligned.pkl")
df.to_csv("topic_relevance_scores.csv", index=False)

print("Topic alignment complete!")
print("  - Saved: topic_aligned.pkl")
print("  - Saved: topic_relevance_scores.csv")
