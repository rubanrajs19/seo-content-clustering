import streamlit as st
import os

def configure_openai_api():
    st.sidebar.header("🔐 OpenAI API (Optional)")
    user_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

    if user_key:
        os.environ["OPENAI_API_KEY"] = user_key
        st.sidebar.success("✅ OpenAI API Key is set for this session.")
    else:
        st.sidebar.info("💡 No key entered — local embedding model will be used.")
