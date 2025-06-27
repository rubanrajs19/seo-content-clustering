import streamlit as st
import subprocess
import sys

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
                st.success(f"✅ {script} executed successfully")
                st.code(result.stdout)
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Error running {script}")
            st.code(e.stdout)
            st.code(e.stderr)
