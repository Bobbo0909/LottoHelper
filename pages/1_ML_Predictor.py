# pages/1_ML_Predictor.py
import streamlit as st
import pandas as pd
from datetime import datetime
import logging

from src.ml_predictor import MLPredictor, load_cleaned_data_for_ml

predictor = MLPredictor()

# --- Page Setup ---
st.title("🤖 ML Predictor")
st.write("Generate potential lottery numbers using Machine Learning.")

# --- Data Preparation and Model Training ---

if 'data' not in st.session_state:
    st.session_state.data = {}
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = {}

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
lottery_config = predictor.lottery_configs[lottery_type_display_name]
file_name = lottery_config['file']

st.markdown("---")
st.subheader("Model Status")

if lottery_type_display_name not in st.session_state.data:
    df = load_cleaned_data_for_ml(lottery_type_display_name, file_name)
    st.session_state.data[lottery_type_display_name] = df
else:
    df = st.session_state.data[lottery_type_display_name]

if lottery_type_display_name not in st.session_state.models_trained:
    if not df.empty:
        st.info(f"Training models for {lottery_type_display_name}...")
        try:
            success = predictor.train_model(lottery_type_display_name, df)
            st.session_state.models_trained[lottery_type_display_name] = success
            if success:
                st.success("Models trained successfully!")
            else:
                st.error("Failed to train models. Please ensure you have sufficient data.")
        except Exception as e:
            st.error(f"An error occurred during model training: {e}")
            st.session_state.models_trained[lottery_type_display_name] = False
    else:
        st.warning(f"No data available for {lottery_type_display_name}. Please update data in 'Data Overview'.")
else:
    if st.session_state.models_trained[lottery_type_display_name]:
        st.success("Models are already trained and ready for prediction.")
    else:
        st.error("Models failed to train previously.")

# --- Prediction Generation and Display ---
st.markdown("---")
st.subheader("Generate Predictions")

if st.session_state.models_trained.get(lottery_type_display_name):
    num_predictions = st.slider("Number of Predictions to Generate", 1, 10, 5)

    if st.button("Generate Predictions"):
        if df.empty:
            st.warning("Cannot generate predictions: historical data is empty.")
        else:
            with st.spinner("Generating predictions..."):
                predictions_list = predictor.generate_predictions(lottery_type_display_name, df, num_predictions)

            if predictions_list:
                st.success(f"Generated {len(predictions_list)} predictions.")

                prediction_data = []
                for i, (numbers, confidence) in enumerate(predictions_list):
                    if lottery_config['has_bonus']:
                        main_numbers_str = ", ".join(map(str, numbers[:-1]))
                        bonus_number_str = str(numbers[-1])
                        formatted_numbers = f"{main_numbers_str} | Bonus: {bonus_number_str}"
                    else:
                        formatted_numbers = ", ".join(map(str, numbers))

                    st.markdown(f"**Prediction {i + 1}:** {formatted_numbers} (Confidence: {confidence:.2%})")

                    prediction_data.append({
                        "Prediction_Index": i + 1,
                        "Numbers_List": numbers,
                        "Confidence": confidence,
                        "Lottery_Type": lottery_type_display_name,
                        "Date_Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                    predictor.show_prediction_analysis(numbers, lottery_type_display_name, df)

                predictions_df = pd.DataFrame(prediction_data)

                csv = predictions_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name=f"{lottery_type_display_name}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download the generated predictions and their confidence scores."
                )

            else:
                st.error("Could not generate predictions. Please check the logs for details.")

else:
    st.warning("Please ensure models are trained successfully before generating predictions.")