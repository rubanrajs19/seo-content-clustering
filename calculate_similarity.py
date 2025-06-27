import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embedded content
df = pd.read_pickle("embedded_content.pkl")

# Filter rows with valid embeddings
df = df[df['embedding'].notnull()].reset_index(drop=True)

# Convert list of vectors to a 2D numpy array
embedding_matrix = np.vstack(df['embedding'].values)

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embedding_matrix)

# Create a DataFrame from the matrix
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=df['Address'],
    columns=df['Address']
)

# Save to CSV
similarity_df.to_csv("similarity_matrix.csv")

print("✅ Cosine similarity matrix saved as 'similarity_matrix.csv'")
