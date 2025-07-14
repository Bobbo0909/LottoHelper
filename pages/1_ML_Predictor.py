# pages/1_ML_Predictor.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.ml_predictor import MLPredictor, load_cleaned_data_for_ml
import os
from datetime import datetime  # ADDED: For current date in CSV filename

# --- Page Configuration ---
st.set_page_config(
    page_title="ML Predictor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Lottery Prediction")


# Instantiate the MLPredictor class once using st.cache_resource
@st.cache_resource
def get_predictor():
    return MLPredictor()


predictor = get_predictor()

# Select lottery type
# Define lottery types and map them to the keys used in MLPredictor's config
lottery_type_map = {
    "Daily Lotto": 'Daily Lotto',
    "Lotto": 'Lotto',
    "Powerball": 'Powerball'
}
lottery_type_display = st.selectbox(
    "Select Lottery Type",
    list(lottery_type_map.keys())
)
selected_lottery = lottery_type_map[lottery_type_display]

# Load and prepare data for the selected lottery type
# This uses the @st.cache_data function from src/ml_predictor.py
df_lottery = load_cleaned_data_for_ml(
    selected_lottery,
    predictor.lottery_configs[selected_lottery]['file']
)

if df_lottery.empty:
    st.warning(
        "No data available. Please go to the main page and update the data first using the 'Update Lottery Data' button.")
    st.stop()  # Stop further execution if no data
else:
    # Corrected variable name from df to df_lottery
    st.write(f"Loaded **{len(df_lottery)}** draws for {selected_lottery}.")
    st.write("Latest Draw Date:", df_lottery['Date'].max().strftime('%Y-%m-%d'))
    st.write("---")

    # Train models if needed
    if selected_lottery not in predictor.models:
        st.info(f"Training models for {selected_lottery}...")

        with st.spinner("Training models..."):
            if not predictor.train_model(selected_lottery, df_lottery):
                st.error(
                    "Failed to train models. Check `logs/ml_predictor.log` for details. This often happens if there isn't enough historical data or if the data is not correctly structured.")
                st.stop()  # Stop further execution if training fails

    if selected_lottery in predictor.models:
        # Show model performance
        st.subheader("Model Performance")
        for model_name, score in predictor.models[selected_lottery]['scores'].items():
            st.metric(f"{model_name.upper()} Model Score", f"{score:.2%}")

        # Explanation of models in an expander
        with st.expander("What are these models?"):
            st.write("""
            Our prediction system uses two powerful Machine Learning models:
            """)
            st.markdown("""
            **1. Random Forest (RF)**
            - **How it works:** Imagine a "forest" of many individual "decision trees." Each tree makes its own prediction. The Random Forest combines the predictions from all these trees (like a vote) to get the final, more accurate, and stable prediction. It's good at handling complex data and avoiding overfitting.
            - **Analogy:** Like asking a diverse group of experts to each give their best guess and then going with the most popular answer.

            **2. Gradient Boosting (GB)**
            - **How it works:** Gradient Boosting also builds a series of decision trees, but it does so sequentially. Each new tree tries to correct the errors made by the previous trees. It focuses on the mistakes and learns from them, gradually improving its accuracy.
            - **Analogy:** Like a team where each member learns from the previous one's mistakes, iteratively improving the overall performance on a task.

            Both models are ensemble methods, meaning they combine multiple simpler models to achieve better performance than a single model alone.
            """)

        st.write("---")

        # Generate predictions
        st.subheader("Generate Predictions")
        num_predictions = st.slider(
            "Number of predictions to generate",
            min_value=1,
            max_value=10,
            value=5
        )

        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                predictions = predictor.generate_predictions(
                    selected_lottery,
                    df_lottery,
                    num_predictions
                )

                if predictions:
                    # Display individual predictions
                    for i, (numbers, confidence) in enumerate(predictions, 1):
                        with st.expander(f"Prediction {i} - Confidence: {confidence:.2%}"):
                            config = predictor.lottery_configs[selected_lottery]

                            if config['has_bonus']:
                                main_numbers = numbers[:-1]
                                bonus = numbers[-1]
                                st.write(f"Main Numbers: {', '.join(map(str, main_numbers))}")
                                st.write(f"{'Powerball' if selected_lottery == 'Powerball' else 'Bonus'}: {bonus}")
                            else:
                                st.write(f"Numbers: {', '.join(map(str, numbers))}")

                            # Show analysis by calling the class method and passing the DataFrame
                            predictor._show_prediction_analysis(numbers, selected_lottery, df_lottery)

                    st.write("---")  # Separator before download button

                    # --- CSV Export Logic ---
                    csv_data = []
                    headers = []

                    config = predictor.lottery_configs[selected_lottery]

                    # Determine headers based on lottery type
                    for i in range(1, config['picks'] + 1):
                        headers.append(f"Number {i}")
                    if config['has_bonus']:
                        if selected_lottery == 'Powerball':
                            headers.append("Powerball")
                        else:
                            headers.append("Bonus")
                    # 'Confidence' column removed as per user request

                    for numbers, confidence in predictions:  # 'confidence' is available but not added to row
                        row = []
                        # Extract main numbers for export
                        main_numbers_for_export = numbers[:-1] if config['has_bonus'] else numbers
                        row.extend(main_numbers_for_export)

                        # Add bonus number if applicable
                        if config['has_bonus']:
                            row.append(numbers[-1])

                        csv_data.append(row)

                    df_export = pd.DataFrame(csv_data, columns=headers)

                    # Convert DataFrame to CSV string
                    csv_string = df_export.to_csv(index=False)

                    # Generate filename with current date and time
                    current_date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                    file_name = f"{selected_lottery.lower().replace(' ', '_')}_predictions_{current_date}.csv"

                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv_string,
                        file_name=file_name,
                        mime="text/csv",
                        help="Download all generated predictions to a CSV file."
                    )
                    # --- End CSV Export Logic ---

                else:
                    st.error(
                        "Failed to generate predictions. This could happen if there isn't enough historical data or an issue occurred during the prediction process.")