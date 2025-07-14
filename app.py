import streamlit as st
import pandas as pd
import os
from datetime import datetime
from collections import Counter
import itertools  # For combinations

# Import the combined data update function from your new data_manager script
from src.data_manager import update_all_data

# --- Configuration ---
# Determine the project root to correctly locate data directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CLEANED_DIR = os.path.join(DATA_DIR, 'cleaned')


# --- Helper Functions with Caching ---

@st.cache_data
def load_cleaned_data(lottery_type):
    """
    Loads the cleaned lottery data for a given lottery type from a Parquet file.
    Returns a DataFrame or None if the file is not found/empty.
    """
    file_path = os.path.join(CLEANED_DIR, f'{lottery_type}_results_all_years.parquet')
    if not os.path.exists(file_path):
        # Changed from st.error to st.warning as initial data might not exist
        # and the user needs to click the update button.
        st.warning(f"Cleaned data file not found for {lottery_type} at: {file_path}. Please update data.")
        return None
    try:
        df = pd.read_parquet(file_path)
        # Ensure 'Date' column is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        # Convert 'Numbers' string to list of integers for analysis
        df['Numbers_List'] = df['Numbers'].apply(lambda x: sorted([int(n.strip()) for n in x.split(',')]))
        return df
    except Exception as e:
        st.error(f"Error loading or processing cleaned data for {lottery_type}: {e}")
        return None


@st.cache_data
def get_hot_cold_numbers(df, top_n=10, recent_draws=None):
    """
    Calculates hot and cold numbers based on overall frequency or recent draws.
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # If recent_draws is specified, use only the head of the DataFrame
    df_subset = df.head(recent_draws) if recent_draws else df

    all_numbers = []
    for num_list in df_subset['Numbers_List']:
        all_numbers.extend(num_list)

    # Add bonus numbers if applicable. This ensures bonus numbers are also counted
    # in overall frequency.
    if 'Bonus' in df_subset.columns:
        all_numbers.extend(df_subset['Bonus'].dropna().astype(int).tolist())

    number_counts = Counter(all_numbers)
    freq_df = pd.DataFrame(number_counts.items(), columns=['Number', 'Frequency'])
    freq_df = freq_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)

    hot_numbers = freq_df.head(top_n)
    cold_numbers = freq_df.tail(top_n).sort_values(by='Frequency', ascending=True).reset_index(drop=True)

    return hot_numbers, cold_numbers


@st.cache_data
def get_number_combinations_frequencies(df, combination_size=2, top_n=20):
    """
    Calculates frequencies of number combinations (pairs or triplets).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    all_combinations = []
    for num_list in df['Numbers_List']:
        # Ensure numbers are sorted within the list before creating combinations.
        # This makes (1, 5) and (5, 1) the same combination.
        sorted_numbers = sorted(num_list)
        all_combinations.extend(list(itertools.combinations(sorted_numbers, combination_size)))

    combination_counts = Counter(all_combinations)
    freq_df = pd.DataFrame(combination_counts.items(), columns=['Combination', 'Frequency'])
    # Convert tuple combinations to a readable string format
    freq_df['Combination'] = freq_df['Combination'].apply(lambda x: ', '.join(map(str, x)))
    freq_df = freq_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)

    return freq_df.head(top_n)


# --- Application Layout ---

st.set_page_config(
    page_title="SA Lottery Analysis",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🇿🇦 South African Lottery Analysis")

# --- Sidebar for Lottery Type Selection ---
st.sidebar.header("Lottery Selection")
lottery_types = ['daily_lotto', 'lotto', 'powerball']
selected_lottery = st.sidebar.selectbox(
    "Choose a Lottery Type:",
    lottery_types,
    index=lottery_types.index('lotto')  # Default to Lotto
)

st.sidebar.markdown("---")

# --- Data Update Button in Sidebar ---
st.sidebar.header("Data Management")
# Button to trigger scraping and cleaning
if st.sidebar.button("📊 Update Lottery Data"):
    with st.spinner(
            "Updating data... This may take a few minutes depending on new draws and your internet speed. Check your terminal for detailed logs."):
        update_success = update_all_data()  # Call the combined function
    if update_success:
        st.sidebar.success("Data updated successfully!")
        # Force a rerun of the app to reload data from the updated files
        st.rerun()
    else:
        st.sidebar.error("Data update failed. Check logs in the `logs/` directory for details.")

st.sidebar.markdown("---")
st.sidebar.info(
    "Data is updated by clicking the 'Update Lottery Data' button, which runs the scraper and cleaner."
)

# --- Load Data ---
# This function is called every time Streamlit reruns.
# Thanks to @st.cache_data, it only re-reads the file if it has changed on disk.
df_lottery = load_cleaned_data(selected_lottery)

if df_lottery is None:
    st.warning(
        f"No data available for {selected_lottery.replace('_', ' ').title()}. Please click 'Update Lottery Data' to fetch it.")
else:
    st.subheader(f"Analyzing: {selected_lottery.replace('_', ' ').title()}")
    st.write(f"Total draws loaded: **{len(df_lottery)}**")
    st.write(
        f"Data from **{df_lottery['Date'].min().strftime('%Y-%m-%d')}** to **{df_lottery['Date'].max().strftime('%Y-%m-%d')}**")

    # --- Hot and Cold Numbers Section ---
    st.markdown("---")
    st.header("🔥 Hot & ❄️ Cold Numbers")
    st.write("Numbers that have appeared most (Hot) and least (Cold) frequently.")

    hot_cold_scope = st.radio(
        "Analyze frequencies based on:",
        ('All Historical Draws', 'Most Recent Draws'),
        key='hot_cold_scope'
    )

    num_top_bottom = st.slider("Number of Hot/Cold numbers to display:", 5, 20, 10, key='num_hot_cold_slider')

    hot_numbers_df, cold_numbers_df = pd.DataFrame(), pd.DataFrame()

    if hot_cold_scope == 'All Historical Draws':
        hot_numbers_df, cold_numbers_df = get_hot_cold_numbers(df_lottery, num_top_bottom)
    else:
        # Ensure that the value for recent_draws_count doesn't exceed available data
        max_recent_draws = len(df_lottery)
        recent_draws_count = st.number_input(
            "Consider how many most recent draws for Hot/Cold numbers?",
            min_value=1,  # Changed min_value to 1 to avoid division by zero or empty data
            max_value=max_recent_draws,
            value=min(200, max_recent_draws),  # Default to 200 or max available
            step=10,  # More granular step
            key='recent_draws_input'
        )
        if recent_draws_count > 0:
            hot_numbers_df, cold_numbers_df = get_hot_cold_numbers(df_lottery, num_top_bottom,
                                                                   recent_draws=recent_draws_count)
        else:
            st.info("Please enter a valid number of recent draws.")

    col_hot, col_cold = st.columns(2)

    with col_hot:
        st.subheader(f"Hot Numbers (Top {num_top_bottom})")
        if not hot_numbers_df.empty:
            st.dataframe(hot_numbers_df, use_container_width=True)
        else:
            st.info("Not enough data to determine hot numbers.")

    with col_cold:
        st.subheader(f"Cold Numbers (Bottom {num_top_bottom})")
        if not cold_numbers_df.empty:
            st.dataframe(cold_numbers_df, use_container_width=True)
        else:
            st.info("Not enough data to determine cold numbers.")

    # --- Number Combinations Analysis Section ---
    st.markdown("---")
    st.header("🔢 Number Combinations Analysis")
    st.write("Explore the frequencies of pairs and triplets of numbers.")

    combination_type = st.radio(
        "Select Combination Type:",
        ('Pairs (2 Numbers)', 'Triplets (3 Numbers)'),
        key='combination_type_radio'
    )

    num_top_combinations = st.slider(
        "Number of top combinations to display:",
        10, 50, 20, key='num_comb_slider'
    )

    if combination_type == 'Pairs (2 Numbers)':
        combinations_df = get_number_combinations_frequencies(df_lottery, combination_size=2,
                                                              top_n=num_top_combinations)
        st.subheader(f"Top {num_top_combinations} Most Frequent Pairs")
    else:  # Triplets (3 Numbers)
        combinations_df = get_number_combinations_frequencies(df_lottery, combination_size=3,
                                                              top_n=num_top_combinations)
        st.subheader(f"Top {num_top_combinations} Most Frequent Triplets")

    if not combinations_df.empty:
        st.dataframe(combinations_df, use_container_width=True)
    else:
        st.info("Not enough data to analyze combinations. Ensure your lottery type has enough numbers per draw.")

    # --- Raw Data/Overview (simplified) ---
    st.markdown("---")
    st.header("Recent Draws Overview")
    st.write("A quick look at the most recent lottery results.")
    # Show bonus column only if it exists for the selected lottery type
    display_cols = ['Date', 'Draw', 'Numbers']
    if 'Bonus' in df_lottery.columns:
        display_cols.append('Bonus')
    st.dataframe(df_lottery[display_cols].head(10), use_container_width=True)