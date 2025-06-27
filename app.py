import streamlit as st
import pandas as pd
import os

from modules.runner import run_script
from modules.uploader import upload_csv_files
from modules.visualizations import plot_strategy_distribution, plot_clicks_vs_relevance
from modules.openai_config import configure_openai_api

st.set_page_config(page_title="SEO Content Strategy Dashboard", layout="wide")
st.title("🔍 SEO Content Strategy Dashboard")

# ---------- Configure OpenAI ----------
configure_openai_api()

# ---------- Run Buttons ----------
st.sidebar.header("⚙️ Run Scripts")
if st.sidebar.button("▶️ Run Full Pipeline"):
    run_script("Full Pipeline", [
        "scripts/seo_analysis_pipeline.py",
        "scripts/topic_alignment.py",
        "scripts/content_strategy.py"
    ])

if st.sidebar.button("🔄 Update Strategy Only"):
    run_script("Topic Alignment + Strategy", [
        "scripts/topic_alignment.py",
        "scripts/content_strategy.py"
    ])

# ---------- File Uploaders ----------
upload_csv_files()

# ---------- Load Final Strategy Output ----------
csv_path = "final_content_strategy.csv"
image_path = "content_clusters.png"

if not os.path.exists(csv_path):
    st.warning("⚠️ 'final_content_strategy.csv' not found. Please run the pipeline first.")
    st.stop()

df = pd.read_csv(csv_path)

# ---------- Filters ----------
st.sidebar.header("🔎 Filter Content")

strategy_options = df['strategy'].unique().tolist()
selected_strategies = st.sidebar.multiselect("Strategy", strategy_options, default=strategy_options)

# Clicks Filter Handling
if 'Clicks' in df.columns and pd.to_numeric(df['Clicks'], errors='coerce').dropna().max() > 0:
    max_clicks = int(pd.to_numeric(df['Clicks'], errors='coerce').max())
    min_clicks = st.sidebar.slider("Minimum Clicks", 0, max_clicks, 0)
else:
    st.sidebar.warning("⚠️ No valid click data available.")
    min_clicks = 0

# Relevance Score Filter
if 'relevance_score' in df.columns:
    min_relevance = st.sidebar.slider("Minimum Relevance Score", 0.0, 1.0, 0.0, step=0.01)
else:
    st.sidebar.warning("⚠️ No relevance scores found. Check input files.")
    min_relevance = 0.0

# ---------- Apply Filters ----------
filtered_df = df[
    (df['strategy'].isin(selected_strategies)) &
    (df.get('Clicks', 0) >= min_clicks) &
    (df.get('relevance_score', 0) >= min_relevance)
]

# ---------- Data Table ----------
st.subheader("📄 Filtered Content")
st.dataframe(
    filtered_df[['Address', 'Clicks', 'relevance_score', 'strategy']].sort_values(by='relevance_score', ascending=False),
    use_container_width=True
)

# ---------- Visualizations ----------
if os.path.exists(image_path):
    st.subheader("📊 UMAP Cluster Plot")
    st.image(image_path, use_column_width=True)

plot_strategy_distribution(df)
plot_clicks_vs_relevance(df)

# ---------- Download ----------
st.download_button(
    label="📥 Download Filtered CSV",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_strategy_output.csv",
    mime="text/csv"
)
