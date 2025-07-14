import streamlit as st
import pandas as pd
from datetime import datetime, date
import os # Import os for path operations

# Import data loading functions and LOTTERY_CONFIGS from data_loader
from src.data_loader import load_cleaned_data_for_ml, get_lottery_config, LOTTERY_CONFIGS
# Import MLPredictor only if it's actually used in this file (it is for config helper)
from src.ml_predictor import MLPredictor

# Assuming these imports are correctly configured based on your project structure
from src.pattern_analyzer import LottoPatternAnalyzer
from src.prediction_validator import LottoPredictionValidator


# --- Page Setup ---
st.title("📊 Pattern-Based Predictor")
st.write("Generate and validate lottery numbers based on historical patterns and statistics.")

# --- Lottery Type Selection ---
# Use LOTTERY_CONFIGS directly from data_loader for consistency
lottery_names = list(LOTTERY_CONFIGS.keys())
default_lottery_name = 'Lotto' # Set a default for this page
if default_lottery_name not in lottery_names:
    default_lottery_name = lottery_names[0] if lottery_names else None

selected_lottery_display_name = st.selectbox(
    "Select Lottery Type:",
    lottery_names,
    index=lottery_names.index(default_lottery_name) if default_lottery_name else 0
)

# Get the actual configuration for the selected lottery
lottery_config = get_lottery_config(selected_lottery_display_name)

# Ensure config is found before proceeding
if lottery_config is None:
    st.error(f"Configuration for '{selected_lottery_display_name}' not found. Please check LOTTERY_CONFIGS in data_loader.py.")
    st.stop() # Stop execution if config is missing

# Use the correct file name from the config
file_name = lottery_config['cleaned_file']


# --- Data Loading ---
# We use the cached data loader function from data_loader.py
# The load_cleaned_data_for_ml function itself is already cached, so no need for nested @st.cache_data here.
def get_data_for_pattern_analyzer(lottery_type_name, file):
    # This directly calls the cached function from data_loader
    return load_cleaned_data_for_ml(lottery_type_name, file)


df_full = get_data_for_pattern_analyzer(selected_lottery_display_name, file_name)

# Flag to control rendering of the rest of the page. Initialize assuming data is available.
data_is_available = True

if df_full.empty:
    st.warning(
        f"No historical data available for {selected_lottery_display_name}. Please ensure data is updated via the 'Data Overview' page or the Home screen.")
    data_is_available = False
else:
    # Ensure 'Date' column is in datetime format before getting min/max for date filter
    df_full['Date'] = pd.to_datetime(df_full['Date'])
    min_date_available = df_full['Date'].min().date()
    max_date_available = df_full['Date'].max().date()

    # --- Date Range Filter (Sidebar) ---
    st.sidebar.header("Filter Historical Data")
    start_date = st.sidebar.date_input("Start Date", value=min_date_available, min_value=min_date_available,
                                       max_value=max_date_available)
    end_date = st.sidebar.date_input("End Date", value=max_date_available, min_value=min_date_available,
                                     max_value=max_date_available)

    # Perform date filtering
    df_to_analyze = pd.DataFrame()  # Initialize empty DataFrame
    if start_date > end_date:
        st.sidebar.error("Error: End date must be after start date.")
        data_is_available = False  # Mark as not available due to invalid date range
    else:
        # Initialize LottoPatternAnalyzer with the full data first, then filter
        # It's better to pass the full df and let the analyzer handle filtering internally if it has that method
        # Or, filter the df here and pass the filtered df. Let's filter here for clarity.
        df_filtered_by_date = df_full[(df_full['Date'].dt.date >= start_date) & (df_full['Date'].dt.date <= end_date)].copy()
        df_to_analyze = df_filtered_by_date # Use the filtered DataFrame

        if df_to_analyze.empty:
            st.warning("No data found for the selected date range. Please adjust the dates.")
            data_is_available = False  # Mark as not available due to empty filtered data

# Only proceed with analysis and prediction if data is valid and available after filters
if data_is_available:
    # --- Historical Pattern Analysis ---
    st.markdown("---")
    st.subheader("Historical Pattern Analysis")
    # Initialize LottoPatternAnalyzer with the (potentially) filtered data
    # Ensure LottoPatternAnalyzer can handle the df_to_analyze and lottery_config
    pattern_analyzer = LottoPatternAnalyzer(df_to_analyze, lottery_config)
    pattern_analyzer.show_pattern_analysis()

    # --- Generate and Validate Predictions ---
    st.markdown("---")
    st.subheader("Generate Pattern-Based Predictions")

    # Initialize LottoPredictionValidator with the (potentially) filtered data
    prediction_validator = LottoPredictionValidator(df_to_analyze, lottery_config)

    num_predictions = st.slider("Number of Predictions to Generate", 1, 10, 5)

    if st.button("Generate Predictions"):
        if df_to_analyze.empty:
            st.warning("Cannot generate predictions: No historical data available in the selected range.")
        else:
            with st.spinner("Generating and validating predictions..."):
                predictions_with_scores = prediction_validator.generate_predictions(num_predictions)

                if predictions_with_scores:
                    validation_results = prediction_validator.validate_against_history(predictions_with_scores)
                    prediction_validator.show_validation_results(predictions_with_scores, validation_results)

                    # Prepare data for CSV download
                    download_data = []
                    for pred, val in zip(predictions_with_scores, validation_results):
                        download_data.append({
                            "Lottery_Type": selected_lottery_display_name, # Use the display name
                            # FIX: Format the combination to be a clean string for CSV
                            "Combination": ', '.join(map(str, pred[0])),
                            "Score": pred[1],
                            "Max_Matches_in_History": val['max_matches'],
                            "Avg_Matches_in_History": val['avg_matches'],
                            "Date_Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

                    download_df = pd.DataFrame(download_data)
                    csv = download_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name=f"{selected_lottery_display_name.replace(' ', '_').lower()}_pattern_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download the generated pattern-based predictions and their validation results."
                    )

                else:
                    st.warning("No predictions could be generated. Please check the data and configurations.")
else:
    # If data_is_available is False, display a general info message
    st.info(
        "Please ensure historical data is loaded and the selected date range is valid to see analysis and generate predictions.")