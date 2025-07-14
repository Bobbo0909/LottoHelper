# app.py
import streamlit as st
import pandas as pd # Assuming you might use pandas for general data display
import os


# --- Page Configuration (General App Settings) ---
st.set_page_config(
    page_title="Lottery Prediction Helper",
    page_icon="🎰",
    layout="centered", # Can be "wide" or "centered"
    initial_sidebar_state="expanded"
)

# --- Path Definitions (Assuming these are still relevant for your app structure) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
CLEANED_DIR = os.path.join(DATA_DIR, 'cleaned')

# Ensure necessary directories exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)

# --- Main App Content ---
st.title("Welcome to the Lottery Prediction Helper! 🎰")

st.write("""
This application helps you analyze past lottery results and generate potential predictions using Machine Learning.

Use the sidebar to navigate:
- **📊 Data Overview:** View historical lottery results and update data.
- **🤖 ML Predictor:** Generate new lottery number predictions.
""")

st.markdown("---")

st.header("How to Use:")
st.markdown("""
1.  **Go to '📊 Data Overview'** in the sidebar.
2.  **Download New Data:** Click the 'Update Lottery Data' button to fetch the latest results from the web. This is crucial for the ML models to have up-to-date information.
3.  **Go to '🤖 ML Predictor'** in the sidebar.
4.  **Select a Lottery Type:** Choose 'Daily Lotto', 'Lotto', or 'Powerball'.
5.  **Generate Predictions:** Click the 'Generate Predictions' button to get a set of predicted numbers.
6.  **Analyze & Download:** View the analysis for each prediction and download them as a CSV file.
""")

st.markdown("---")

st.info("💡 **Tip:** The quality of predictions relies on having up-to-date and clean historical data. Make sure to update the data regularly!")

# Optional: Add a section for feedback or contact
st.sidebar.markdown("### About")
st.sidebar.info("This app is a project by Robert.")