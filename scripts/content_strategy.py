import pandas as pd
import numpy as np

# ---------- Step 1: Load required data ----------
print("Loading topic-aligned embeddings and GSC performance data...")
df = pd.read_pickle("topic_aligned.pkl")        # From topic_alignment.py
gsc = pd.read_csv("gsc_data.csv")               # From Google Search Console export

# ---------- Step 2: Merge on page URL ----------
print("Merging performance metrics...")
merged = df.merge(gsc, left_on="Address", right_on="Page", how="left")

# ---------- Step 3: Calculate topic relevance score ----------
print("Calculating maximum relevance per page...")
topic_cols = [col for col in merged.columns if col.startswith("relevance_to_")]
merged["relevance_score"] = merged[topic_cols].max(axis=1)

# Fill missing Clicks with 0 (e.g., if GSC didn't report anything)
merged["Clicks"] = merged["Clicks"].fillna(0)

# ---------- Step 4: Content Strategy Classification ----------
RELEVANCE_THRESHOLD = 0.75
CLICKS_THRESHOLD = 50

conditions = [
    (merged["relevance_score"] >= RELEVANCE_THRESHOLD) & (merged["Clicks"] >= CLICKS_THRESHOLD),
    (merged["relevance_score"] >= RELEVANCE_THRESHOLD) & (merged["Clicks"] < CLICKS_THRESHOLD),
    (merged["relevance_score"] < RELEVANCE_THRESHOLD) & (merged["Clicks"] < CLICKS_THRESHOLD),
]
choices = ["Keep", "Update", "Prune"]
merged["strategy"] = np.select(conditions, choices, default="Review")

# ---------- Step 5: Save Final Strategy Output ----------
merged.to_csv("final_content_strategy.csv", index=False)

print("Content strategy analysis complete!")
print("Output saved to: final_content_strategy.csv")
