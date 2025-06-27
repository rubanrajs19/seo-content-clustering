import streamlit as st

def upload_csv_files():
    st.sidebar.header("📂 Upload CSV Files")

    uploaded_files = {
        "content.csv": st.sidebar.file_uploader("Upload content.csv", type="csv"),
        "gsc_data.csv": st.sidebar.file_uploader("Upload gsc_data.csv", type="csv"),
        "strategic_topics.csv": st.sidebar.file_uploader("Upload strategic_topics.csv", type="csv"),
    }

    for filename, uploaded in uploaded_files.items():
        if uploaded:
            with open(filename, "wb") as f:
                f.write(uploaded.read())
            st.sidebar.success(f"✅ {filename} uploaded")
