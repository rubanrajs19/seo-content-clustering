import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Load the sample content file
df = pd.read_csv("content.csv")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to generate embedding for a single text
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Generate embeddings for each page's BodyText
print("Generating embeddings...")
embeddings = []
for text in tqdm(df['BodyText'].fillna('')):
    try:
        emb = get_embedding(text)
        embeddings.append(emb)
    except Exception as e:
        print("Error:", e)
        embeddings.append(None)

# Add embeddings to the dataframe
df['embedding'] = embeddings

# Save to new file
df.to_pickle("embedded_content.pkl")
print("✅ Embeddings generated and saved to 'embedded_content.pkl'")
