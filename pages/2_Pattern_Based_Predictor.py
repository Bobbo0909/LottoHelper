import streamlit as st
import pandas as pd
from datetime import datetime, date

# Assuming these imports are correctly configured based on your project structure
from src.pattern_analyzer import LottoPatternAnalyzer
from src.prediction_validator import LottoPredictionValidator
from src.ml_predictor import load_cleaned_data_for_ml, MLPredictor

# --- Page Setup ---
st.title("📊 Pattern-Based Predictor")
st.write("Generate and validate lottery numbers based on historical patterns and statistics.")

# --- Lottery Type Selection ---
# We use MLPredictor to access lottery configurations consistently
predictor_config_helper = MLPredictor()
LOTTERY_OPTIONS = {
    "Daily Lotto (1-36, 5 picks)": "Daily Lotto",
    "Lotto (1-52, 6 picks + Bonus)": "Lotto",
    "Powerball (1-50, 5 picks + Powerball 1-20)": "Powerball"
}

selected_option = st.selectbox(
    "Select Lottery Type:",
    list(LOTTERY_OPTIONS.keys())
)

lottery_type_display_name = LOTTERY_OPTIONS[selected_option]
lottery_config = predictor_config_helper.lottery_configs[lottery_type_display_name]
file_name = lottery_config['file']


# --- Data Loading ---
# We use the cached data loader function from ml_predictor.py
@st.cache_data(show_spinner="Loading and preprocessing historical data...")
def get_data_for_pattern_analyzer(lottery_type_name, file):
    return load_cleaned_data_for_ml(lottery_type_name, file)


df_full = get_data_for_pattern_analyzer(lottery_type_display_name, file_name)

# Flag to control rendering of the rest of the page. Initialize assuming data is available.
data_is_available = True

if df_full.empty:
    st.warning(
        f"No historical data available for {lottery_type_display_name}. Please ensure data is updated via the 'Data Overview' page or the Home screen.")
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
        # Use LottoPatternAnalyzer's update_date_range for filtering
        temp_analyzer_for_filter = LottoPatternAnalyzer(df_full, lottery_config)
        filtered_analyzer = temp_analyzer_for_filter.update_date_range(start_date, end_date)
        df_to_analyze = filtered_analyzer.df  # Get the filtered DataFrame from the new analyzer instance

        if df_to_analyze.empty:
            st.warning("No data found for the selected date range. Please adjust the dates.")
            data_is_available = False  # Mark as not available due to empty filtered data

# Only proceed with analysis and prediction if data is valid and available after filters
if data_is_available:
    # --- Historical Pattern Analysis ---
    st.markdown("---")
    st.subheader("Historical Pattern Analysis")
    # Initialize LottoPatternAnalyzer with the (potentially) filtered data
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
                            "Lottery_Type": lottery_type_display_name,
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
                        file_name=f"{lottery_type_display_name}_pattern_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download the generated pattern-based predictions and their validation results."
                    )

                else:
                    st.warning("No predictions could be generated. Please check the data and configurations.")
else:
    # If data_is_available is False, display a general info message
    st.info(
        "Please ensure historical data is loaded and the selected date range is valid to see analysis and generate predictions.")