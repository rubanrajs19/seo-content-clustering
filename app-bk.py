import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import subprocess


st.set_page_config(page_title="SEO Content Strategy Dashboard", layout="wide")
st.title("🔍 SEO Content Strategy Dashboard")

# ---------- Sidebar: Run Pipeline Buttons ----------
st.sidebar.header("⚙️ Run Scripts")

def run_script(label, scripts):
    with st.spinner(f"Running: {label}"):
        try:
            for script in scripts:
                result = subprocess.run(
                    [sys.executable, script],
                    capture_output=True,
                    text=True,
                    check=True
                )
                st.text(f"✅ {script} output:")
                st.code(result.stdout)
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Error running {script}:")
            st.code(e.stdout)
            st.code(e.stderr)

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

# ---------- Load Final Strategy Output ----------
csv_path = "final_content_strategy.csv"
image_path = "content_clusters.png"

if not os.path.exists(csv_path):
    st.warning("⚠️ 'final_content_strategy.csv' not found. Please run the pipeline first.")
    st.stop()

df = pd.read_csv(csv_path)

# ---------- Sidebar Filters ----------
st.sidebar.header("🔎 Filter Content")

strategy_options = df['strategy'].unique().tolist()
selected_strategies = st.sidebar.multiselect("Strategy", strategy_options, default=strategy_options)

min_clicks = st.sidebar.slider("Minimum Clicks", min_value=0, max_value=int(df['Clicks'].max()), value=0)
min_relevance = st.sidebar.slider("Minimum Relevance Score", 0.0, 1.0, 0.0, step=0.01)

# ---------- Filter Data ----------
filtered_df = df[
    (df['strategy'].isin(selected_strategies)) &
    (df['Clicks'] >= min_clicks) &
    (df['relevance_score'] >= min_relevance)
]

# ---------- Main Table ----------
st.subheader("📄 Filtered Content")
st.dataframe(
    filtered_df[['Address', 'Clicks', 'relevance_score', 'strategy']].sort_values(by='relevance_score', ascending=False),
    use_container_width=True
)

# ---------- Visualization ----------
st.subheader("📊 UMAP Cluster Plot")
if os.path.exists(image_path):
    st.image(image_path, use_column_width=True)
else:
    st.info("No cluster plot found. Run the full pipeline to generate 'content_clusters.png'.")

# ---------- CSV Export ----------
st.download_button(
    label="📥 Download Filtered CSV",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_strategy_output.csv",
    mime="text/csv"
)
