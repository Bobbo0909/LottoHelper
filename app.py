# app.py
import streamlit as st
import pandas as pd
import os

# Import the new scraper and cleaner modules
import src.scraper as scraper
import src.cleaner as cleaner
# Import the new data_loader module and its LOTTERY_CONFIGS
from src.data_loader import load_cleaned_data_for_ml, get_lottery_config, LOTTERY_CONFIGS

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CLEANED_DIR = os.path.join(DATA_DIR, 'cleaned')

# Ensure cleaned directory exists for load functions
os.makedirs(CLEANED_DIR, exist_ok=True)

# Default lottery for initial display
default_lottery_name = 'Lotto'


# --- Helper Functions for Hot/Cold Analysis ---

def calculate_hot_cold_numbers(df, top_n=10):
    """
    Calculates the frequency of all main numbers (excluding bonus) and identifies
    the hottest and coldest numbers based on overall frequency.
    """

    # 1. Extract all main numbers from the 'Numbers_List' column
    # We flatten the list of lists into a single list of all numbers drawn
    all_main_numbers = [num for sublist in df['Numbers_List'] for num in sublist]

    if not all_main_numbers:
        return pd.DataFrame(), pd.DataFrame()

    # 2. Calculate frequencies
    frequency_series = pd.Series(all_main_numbers).value_counts()

    # 3. Create a DataFrame for frequency
    freq_df = frequency_series.reset_index()
    freq_df.columns = ['Number', 'Frequency']

    # 4. Identify Hot (most frequent) and Cold (least frequent) numbers

    # Sort by frequency descending for Hot Numbers
    hot_numbers = freq_df.sort_values(by='Frequency', ascending=False).head(top_n).reset_index(drop=True)

    # Sort by frequency ascending for Cold Numbers
    cold_numbers = freq_df.sort_values(by='Frequency', ascending=True).head(top_n).reset_index(drop=True)

    return hot_numbers, cold_numbers


# --- Main App Logic ---

st.set_page_config(layout="wide", page_title="Lottery Helper")

st.title("Lottery Helper & Predictor")
st.markdown("Analyze past lottery data and get prediction insights.")

# --- Lottery Selection and Data Management on Home Screen ---

# Lottery Selection
st.header("Select Lottery Type")
selected_lottery = st.radio(
    "Choose a Lottery:",
    list(LOTTERY_CONFIGS.keys()),
    index=list(LOTTERY_CONFIGS.keys()).index(default_lottery_name),
    horizontal=True
)

# Data Management/Update Button
st.header("Data Management")

if st.button("Update All Lottery Data", help="Fetches new lottery results and cleans them for analysis."):
    st.info("Updating data. This may take a while...")

    # Run scraper and cleaner (as implemented in the previous step)
    scraper_success = scraper.scrape_all_lotteries_incremental_callable()
    if scraper_success:
        st.success("Raw data scraping completed. Now cleaning...")
        cleaner_success = cleaner.clean_all_lottery_data_callable()
        if cleaner_success:
            st.success("Data cleaning completed successfully!")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("Data cleaning failed or no data was cleaned.")
    else:
        st.info("No new raw data was scraped. Data might be up-to-date, or an error occurred during scraping.")
        cleaner_success = cleaner.clean_all_lottery_data_callable()
        if cleaner_success:
            st.success("Existing raw data cleaned successfully!")
            st.cache_data.clear()
            st.rerun()
        else:
            st.warning("No new data to clean, or cleaning failed for existing data.")

st.markdown("---")  # Visual separator

# --- Data Loading and Analysis ---

# Get the config for the selected lottery
selected_config = get_lottery_config(selected_lottery)
if selected_config is None:
    st.error(f"Configuration for '{selected_lottery}' not found. Please check LOTTERY_CONFIGS in data_loader.py.")
    st.stop()

# Get the file name for the selected lottery from its config
selected_file_name = selected_config['cleaned_file']

# Load data for the selected lottery using the new data_loader
df_selected_lotto = load_cleaned_data_for_ml(selected_lottery, selected_file_name)

if df_selected_lotto.empty:
    st.warning(f"No data available for {selected_lottery}. Please update data by clicking 'Update All Lottery Data'.")
    # Stop displaying the analysis sections if no data is loaded
else:
    # --- Data Overview Section ---
    st.write(f"### {selected_lottery} Data Overview")
    st.write(f"Number of draws: {len(df_selected_lotto)}")
    st.write(
        f"Data from {df_selected_lotto['Date'].min().strftime('%Y-%m-%d')} to {df_selected_lotto['Date'].max().strftime('%Y-%m-%d')}")
    st.dataframe(df_selected_lotto.head())

    # --- Hot and Cold Numbers Section ---
    st.header("Hot and Cold Numbers Analysis")

    # Calculate hot and cold numbers
    hot_numbers_df, cold_numbers_df = calculate_hot_cold_numbers(df_selected_lotto, top_n=10)

    # Display the results using Streamlit columns for side-by-side view
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"🔥 Hot Numbers (Top {len(hot_numbers_df)})")
        if not hot_numbers_df.empty:
            st.dataframe(hot_numbers_df)
        else:
            st.info("Unable to calculate hot numbers.")

    with col2:
        st.subheader(f"❄️ Cold Numbers (Bottom {len(cold_numbers_df)})")
        if not cold_numbers_df.empty:
            st.dataframe(cold_numbers_df)
        else:
            st.info("Unable to calculate cold numbers.")

