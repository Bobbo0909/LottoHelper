# pages/1_ML_Predictor.py
import streamlit as st
import pandas as pd
import os

# Import data loading functions from data_loader (where they now reside)
from src.data_loader import load_cleaned_data_for_ml, get_lottery_config, LOTTERY_CONFIGS
# Import only the MLPredictor class from ml_predictor
from src.ml_predictor import MLPredictor

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CLEANED_DIR = os.path.join(DATA_DIR, 'cleaned')

# --- Streamlit App Structure for this page ---

st.set_page_config(layout="wide", page_title="ML Predictor")

st.title("Lottery Prediction Powered by Machine Learning")
st.markdown("Leveraging historical data to generate potential future lottery numbers.")

# Initialize session state for predictions if not already present
if 'current_predictions' not in st.session_state:
    st.session_state['current_predictions'] = []
if 'last_selected_lottery' not in st.session_state:
    st.session_state['last_selected_lottery'] = None

# --- Lottery Selection ---
st.header("Select Lottery Type for Prediction")

# Get lottery configurations from data_loader
lottery_names = list(LOTTERY_CONFIGS.keys())
default_lottery_name = 'Lotto'  # Set a default for this page if needed
if default_lottery_name not in lottery_names:
    default_lottery_name = lottery_names[0] if lottery_names else None

selected_lottery = st.selectbox(
    "Choose a Lottery:",
    lottery_names,
    index=lottery_names.index(default_lottery_name) if default_lottery_name else 0,
    key="lottery_selector"  # Add a key for the selectbox
)

# Clear predictions if lottery type changes
if st.session_state['last_selected_lottery'] != selected_lottery:
    st.session_state['current_predictions'] = []
    st.session_state['last_selected_lottery'] = selected_lottery

if selected_lottery:
    selected_config = get_lottery_config(selected_lottery)
    selected_file_name = selected_config['cleaned_file']

    # --- Load Data ---
    st.subheader("Loading Data...")
    df_selected_lotto = load_cleaned_data_for_ml(selected_lottery, selected_file_name)

    if df_selected_lotto.empty:
        st.warning(
            f"No cleaned data available for {selected_lottery}. Please run the 'Update All Lottery Data' process on the Home page.")
    else:
        st.success(f"Loaded {len(df_selected_lotto)} draws for {selected_lottery}.")
        st.write(
            f"Data from {df_selected_lotto['Date'].min().strftime('%Y-%m-%d')} to {df_selected_lotto['Date'].max().strftime('%Y-%m-%d')}")

        # --- ML Prediction Section ---
        st.header("Generate Predictions")

        # Initialize ML predictor
        predictor = MLPredictor()

        # Training happens once when the page loads, as it's a prerequisite for prediction
        if len(df_selected_lotto) < 2:
            st.info(
                "Not enough historical data to train the prediction model (at least 2 draws needed). Please scrape more data.")
        else:
            with st.spinner("Training prediction model..."):
                model_trained = predictor.train_model(selected_lottery, df_selected_lotto)

            if model_trained:
                st.success("Prediction model trained successfully!")

                # SLIDER FOR NUMBER OF PREDICTIONS
                num_predictions_to_show = st.slider(
                    "Number of prediction sets to generate:",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key="num_preds_slider"
                )

                # GENERATE BUTTON
                if st.button("Generate Lottery Predictions", key="generate_button"):
                    # This block will only execute when the button is clicked
                    with st.spinner(f"Generating {num_predictions_to_show} predictions..."):
                        predictions_with_confidence = predictor.generate_predictions(
                            selected_lottery, df_selected_lotto, num_predictions=num_predictions_to_show
                        )

                    # Store predictions in session_state
                    st.session_state['current_predictions'] = predictions_with_confidence

                    if not predictions_with_confidence:
                        st.warning("Prediction generation failed. Please check the logs (`logs/ml_predictor.log`).")

                # Display predictions if they exist in session_state
                if st.session_state['current_predictions']:
                    st.subheader("🔮 Generated Predictions")

                    # Prepare data for display and export
                    predictions_data_for_df = []
                    for i, (numbers, confidence) in enumerate(st.session_state['current_predictions']):
                        prediction_entry = {"Prediction Set": f"Set {i + 1}", "Confidence": f"{confidence:.2%}"}

                        # Determine if bonus is present for display
                        is_lotto_or_powerball = selected_lottery in ['Lotto', 'Powerball']
                        # Check if the length matches expected (picks + 1 for bonus)
                        has_expected_bonus_length = (len(numbers) == selected_config['picks'] + 1)

                        if is_lotto_or_powerball and has_expected_bonus_length:
                            main_numbers = sorted(numbers[:-1])
                            bonus_number = numbers[-1]
                            prediction_entry["Main Numbers"] = ", ".join(map(str, main_numbers))
                            prediction_entry["Bonus Number"] = bonus_number
                        else:
                            prediction_entry["Numbers"] = ", ".join(map(str, sorted(numbers)))
                            prediction_entry["Bonus Number"] = "N/A"  # Consistent column for all lotteries

                        predictions_data_for_df.append(prediction_entry)

                    predictions_df = pd.DataFrame(predictions_data_for_df)
                    st.dataframe(predictions_df, hide_index=True)

                    # EXPORT TO CSV BUTTON
                    csv_filename = f"{selected_lottery.replace(' ', '_').lower()}_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    st.download_button(
                        label="Export Predictions to CSV",
                        data=predictions_df.to_csv(index=False),
                        file_name=csv_filename,
                        mime="text/csv",
                        help="Download the generated prediction sets as a CSV file."
                    )

                    # Show detailed analysis for the top prediction (first one in the sorted list)
                    st.markdown("---")
                    st.subheader("Detailed Analysis for Top Prediction")
                    # Ensure current_predictions is not empty before accessing index 0
                    if st.session_state['current_predictions']:
                        predictor.show_prediction_analysis(st.session_state['current_predictions'][0][0],
                                                           selected_lottery, df_selected_lotto)
                    else:
                        st.info("No top prediction available for detailed analysis.")
                else:
                    # Message displayed when no predictions are generated yet (or after clearing)
                    st.info(
                        "Set the desired number of predictions and click 'Generate Lottery Predictions' to see results.")

            else:
                st.error(
                    "Could not train prediction model. This typically happens if there isn't enough clean data, or if there are structural issues with the data. Please check the logs (`logs/ml_predictor.log`) for details on why training failed.")

        st.markdown("---")
        # EXPLANATION OF MODELS
        st.header("Understanding the Prediction Models")
        st.markdown("""
        This tool uses two common machine learning models for prediction: **Random Forest** and **Gradient Boosting**.

        **How they work (Simplified):**
        Instead of trying to predict the *exact* next lottery numbers, these models are trained to learn patterns from the sequence of past draws. They analyze features from previous draws (like sums, parities, ranges, etc.) to infer which numbers are more likely to appear in the *next* draw, or what properties the next draw might have.

        The predictions displayed are generated by combining the outputs from both models. Numbers that both models indicate as having a higher probability are favored.

        * **Random Forest (RF):** Imagine a "forest" made up of many individual decision trees. Each tree in the forest makes its own prediction. For example, one tree might predict "number 7 is likely," another "number 12 is likely." The Random Forest then combines these individual predictions (e.g., by taking a vote or an average) to produce a more stable and accurate final prediction. It's excellent at handling complex relationships in data and reducing overfitting.
        * **Gradient Boosting (GB):** This is another powerful ensemble method, but it builds trees in a sequential manner. Each new tree that is added to the model attempts to correct the errors made by the previously built trees. It "learns" from the mistakes of its predecessors, gradually improving the overall prediction accuracy. Gradient Boosting models are known for their high performance in many predictive tasks.

        By leveraging both Random Forest and Gradient Boosting, we aim to gain a more robust and insightful set of predictions by considering different learning approaches and their combined strengths.
        """)


else:
    st.info("Please select a lottery type to get predictions.")