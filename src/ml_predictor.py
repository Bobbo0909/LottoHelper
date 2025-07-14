# src/ml_predictor.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
import plotly.express as px
from collections import Counter
import logging
from datetime import datetime
import os
import ast  # For safe evaluation of string representations of lists

# --- Configuration & Logging ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CLEANED_DIR = os.path.join(DATA_DIR, 'cleaned')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging to write to a file
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, 'ml_predictor.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# --- Cached Data Loader Function ---
@st.cache_data(show_spinner=False)
def load_cleaned_data_for_ml(lottery_type_display_name, file_name):
    """
    Loads the cleaned lottery data for a given lottery type from a Parquet file.
    Returns a DataFrame or None if the file is not found/empty.
    Caches the result.

    Args:
        lottery_type_display_name (str): The display name of the lottery (e.g., 'Daily Lotto').
        file_name (str): The filename of the parquet file (e.g., 'daily_lotto_results_all_years.parquet').
    Returns:
        pd.DataFrame: The loaded and preprocessed DataFrame.
    """
    file_path = os.path.join(CLEANED_DIR, file_name)
    if not os.path.exists(file_path):
        logging.warning(f"Cleaned data file not found at: {file_path} for {lottery_type_display_name}")
        return pd.DataFrame()  # Return empty DataFrame if file not found
    try:
        df = pd.read_parquet(file_path)

        # Ensure 'Date' column is datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])  # Drop rows where date could not be parsed

        # --- ROBUST NUMBER PARSING ---
        def parse_numbers_string_robust(x):
            """Safely parses various formats of numbers data into a sorted list of integers."""
            if pd.isna(x):
                return []
            try:
                # First, try ast.literal_eval. This handles string representations of lists '[1, 2, 3]' or even single numbers '10'
                evaluated = ast.literal_eval(str(x))  # Convert to string before literal_eval

                if isinstance(evaluated, (list, tuple)):
                    # If it's a list/tuple, convert all elements to int safely, filtering non-numeric
                    return sorted([int(str(n).strip()) for n in evaluated if str(n).strip().isdigit()])
                elif isinstance(evaluated, (int, float)):
                    # If it's a single number, return as a list
                    return sorted([int(evaluated)])
                else:
                    # Fallback for other unexpected types after literal_eval
                    logging.warning(
                        f"Unexpected type after ast.literal_eval: {type(evaluated)} for value: '{x}' in {lottery_type_display_name}")
                    return []
            except (ValueError, SyntaxError):
                # If ast.literal_eval fails (e.g., for "1, 2, 3" without brackets),
                # try splitting by comma, if it's a string
                if isinstance(x, str):
                    return sorted([int(n.strip()) for n in x.split(',') if n.strip().isdigit()])
                return []
            except Exception as e:
                logging.error(f"Failed to parse number string/value '{x}' for {lottery_type_display_name}: {e}",
                              exc_info=True)
                return []

        df['Numbers_List'] = df['Numbers'].apply(parse_numbers_string_robust)
        # --- END ROBUST PARSING ---

        # Drop rows where Numbers_List could not be parsed into a non-empty list
        # This implicitly handles rows where 'Numbers' was invalid or became empty list after parsing
        df = df[df['Numbers_List'].apply(lambda x: len(x) > 0)].copy()

        # Ensure Bonus column is int if it exists
        if 'Bonus' in df.columns:
            # Convert to nullable integer, coerce errors to NaN, then drop NaNs
            df['Bonus'] = pd.to_numeric(df['Bonus'], errors='coerce').astype('Int64')
            df = df.dropna(subset=['Bonus'])

        logging.info(
            f"Successfully loaded and preprocessed {len(df)} rows for {lottery_type_display_name} from {file_path}")
        return df
    except Exception as e:
        logging.error(
            f"Critical error loading or processing cleaned data for {lottery_type_display_name} from {file_path}: {e}",
            exc_info=True)
        return pd.DataFrame()


class MLPredictor:
    def __init__(self):
        self.lottery_configs = {
            'Daily Lotto': {
                'file': 'daily_lotto_results_all_years.parquet',
                'range': (1, 36),
                'picks': 5,
                'has_bonus': False
            },
            'Lotto': {
                'file': 'lotto_results_all_years.parquet',
                'range': (1, 52),
                'picks': 6,
                'has_bonus': True
            },
            'Powerball': {
                'file': 'powerball_results_all_years.parquet',
                'range': (1, 50),
                'picks': 5,
                'has_bonus': True,
                'bonus_range': (1, 20)
            }
        }
        self.models = {}
        self.scalers = {}
        # self.data is no longer stored here as data is passed dynamically

    def _prepare_features(self, df, lottery_type):
        """
        Prepares features (X) and targets (y) from the DataFrame for model training.
        """
        config = self.lottery_configs[lottery_type]
        features = []
        targets = []

        # We need at least two rows to create one feature-target pair
        if len(df) < 2:
            logging.warning(
                f"Not enough data to prepare features for {lottery_type}. Needs at least 2 rows, has {len(df)}.")
            return np.array([]), np.array([])

        # Iterate up to the second to last element to predict the next
        for i in range(len(df) - 1):
            current_numbers = df.iloc[i]['Numbers_List']
            next_numbers = df.iloc[i + 1]['Numbers_List']

            # Skip if any of the lists are empty (due to parsing errors or missing data)
            if not current_numbers or not next_numbers:
                continue

            feature_vector = self._extract_features(current_numbers, config)
            if feature_vector is not None:
                features.append(feature_vector)
                targets.append(next_numbers)

        if not features:  # If no valid feature-target pairs could be created
            logging.warning(f"No valid feature-target pairs could be prepared for {lottery_type}.")
            return np.array([]), np.array([])

        return np.array(features), np.array(targets)

    def _extract_features(self, numbers, config):
        """
        Extracts various features from a list of lottery numbers.
        Returns None if numbers list is invalid/empty.
        """
        if not numbers:
            logging.warning("Attempted to extract features from an empty numbers list.")
            return None

        try:
            features = []
            max_num = config['range'][1]

            # One-hot encoding of current numbers
            number_features = np.zeros(max_num)
            for n in numbers:
                if 1 <= n <= max_num:  # Ensure number is within valid range
                    number_features[n - 1] = 1
            features.extend(number_features)

            # Statistical features
            # These are guaranteed to have numbers due to the initial 'if not numbers' check
            features.extend([
                np.mean(numbers),
                np.std(numbers),
                np.median(numbers),
                max(numbers) - min(numbers)
            ])

            # Pattern features
            features.extend([
                sum(1 for n in numbers if n % 2 == 0) / len(numbers),  # Even ratio
                sum(1 for n in numbers if n <= max_num / 2) / len(numbers),  # Low ratio
                len(set(n // 10 for n in numbers)) / ((max_num // 10) + 1)  # Decade coverage
            ])

            return features

        except Exception as e:
            logging.error(f"Error in feature extraction for numbers {numbers}: {str(e)}", exc_info=True)
            return None

    def train_model(self, lottery_type, df):
        """
        Trains machine learning models for the specified lottery type.

        Args:
            lottery_type (str): The type of lottery (e.g., 'Daily Lotto').
            df (pd.DataFrame): The DataFrame containing the lottery data.
        Returns:
            bool: True if models were trained successfully, False otherwise.
        """
        if df.empty:
            logging.error(f"No data available for training {lottery_type}.")
            return False

        try:
            X, y = self._prepare_features(df, lottery_type)

            # Check if feature preparation yielded enough data
            if X.shape[0] < 2 or X.shape[1] == 0:
                logging.error(
                    f"Insufficient or invalid data for training for {lottery_type} after feature preparation. X shape: {X.shape}")
                return False

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[lottery_type] = scaler

            # Define models to train
            models = {
                'rf': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'gb': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            }

            trained_models = {}
            model_scores = {}

            for name, model in models.items():
                try:
                    # MultiOutputClassifier needs n_jobs to be -1 for actual parallelization
                    multi_model = MultiOutputClassifier(estimator=model, n_jobs=-1)
                    multi_model.fit(X_scaled, y)
                    score = multi_model.score(X_scaled, y)  # Score on training data
                    trained_models[name] = multi_model
                    model_scores[name] = score
                    logging.info(f"Trained {name} model for {lottery_type}. Score: {score:.2%}")
                except Exception as e:
                    logging.error(f"Error training {name} model for {lottery_type}: {str(e)}", exc_info=True)
                    continue

            if trained_models:
                self.models[lottery_type] = {
                    'models': trained_models,
                    'scores': model_scores
                }
                return True
            else:
                logging.error(f"Failed to train any models for {lottery_type}.")
                return False

        except Exception as e:
            logging.error(f"Unexpected error during model training for {lottery_type}: {str(e)}", exc_info=True)
            return False

    def generate_predictions(self, lottery_type, df, num_predictions=5):
        """
        Generates lottery number predictions using the trained models.

        Args:
            lottery_type (str): The type of lottery.
            df (pd.DataFrame): The DataFrame containing the lottery data.
            num_predictions (int): Number of prediction sets to generate.
        Returns:
            list: A list of tuples, each containing a predicted set of numbers and its confidence.
        """
        if lottery_type not in self.models:
            logging.error(f"Models not trained for {lottery_type}.")
            return []
        if df.empty:
            logging.error(f"No data available to get latest draw for prediction for {lottery_type}.")
            return []

        config = self.lottery_configs[lottery_type]
        predictions = []

        try:
            # Get latest draw from the provided DataFrame
            # Ensure the latest_numbers list is not empty
            latest_numbers_list = df.iloc[-1]['Numbers_List']
            if not latest_numbers_list:
                logging.error("Latest draw numbers list is empty, cannot generate features for prediction.")
                return []

            features = self._extract_features(latest_numbers_list, config)

            if features is None:
                raise ValueError("Failed to extract features from latest draw.")

            # Reshape for single sample prediction
            features_scaled = self.scalers[lottery_type].transform([features])

            all_predictions_from_models = []
            for model_name, model in self.models[lottery_type]['models'].items():
                pred = model.predict(features_scaled)[0]  # Get the first (and only) prediction
                all_predictions_from_models.extend(pred)

            # Use Counter to get frequencies of numbers predicted by all models
            number_probabilities_counts = Counter(all_predictions_from_models)
            total_predictions_count = len(all_predictions_from_models)

            # Generate multiple unique prediction sets
            generated_sets = set()
            attempts = 0
            # Limit attempts to avoid infinite loops if it's hard to find unique sets
            max_attempts = num_predictions * 10
            while len(predictions) < num_predictions and attempts < max_attempts:
                single_prediction = self._generate_single_prediction(
                    number_probabilities_counts,
                    total_predictions_count,
                    config
                )
                if single_prediction is not None:
                    # Convert list to tuple for hashing in a set (lists are unhashable)
                    prediction_tuple = tuple(single_prediction)
                    if prediction_tuple not in generated_sets:
                        generated_sets.add(prediction_tuple)
                        confidence = self._calculate_prediction_confidence(
                            single_prediction,
                            number_probabilities_counts,
                            total_predictions_count
                        )
                        predictions.append((single_prediction, confidence))
                attempts += 1

            if not predictions and num_predictions > 0:
                logging.warning(
                    f"Could not generate {num_predictions} unique predictions for {lottery_type} after {attempts} attempts.")
            return sorted(predictions, key=lambda x: x[1], reverse=True)

        except Exception as e:
            logging.error(f"Error generating predictions for {lottery_type}: {str(e)}", exc_info=True)
            return []

    def _generate_single_prediction(self, probabilities_counts, total_predictions_count, config):
        """
        Generates a single set of lottery numbers based on probabilities.
        """
        try:
            # Get the full range of possible numbers for the lottery
            all_possible_numbers = list(range(config['range'][0], config['range'][1] + 1))

            # Convert counts from ML predictions to probabilities
            probs_from_ml = {
                num: count / total_predictions_count
                for num, count in probabilities_counts.items()
            }

            # Combine ML-based probabilities with a small base probability for unpredicted numbers
            final_probs = {}
            # Determine a base probability for numbers not directly predicted by models
            min_non_zero_prob_from_ml = min(p for p in probs_from_ml.values() if p > 0) if any(
                p > 0 for p in probs_from_ml.values()) else 0.001

            for num in all_possible_numbers:
                final_probs[num] = probs_from_ml.get(num, min_non_zero_prob_from_ml / 2)  # Give a small chance

            # Normalize probabilities to sum to 1
            total_final_prob = sum(final_probs.values())
            if total_final_prob == 0:
                logging.warning("All final probabilities sum to zero. Using uniform distribution for prediction.")
                final_probs = {num: 1 / len(all_possible_numbers) for num in all_possible_numbers}
                total_final_prob = 1
            else:
                final_probs = {num: p / total_final_prob for num, p in final_probs.items()}

            # Create sorted list of numbers and their corresponding probabilities for np.random.choice
            numbers_for_choice = sorted(final_probs.keys())
            probabilities_for_choice = [final_probs[num] for num in numbers_for_choice]

            # Ensure we have enough numbers in the pool to pick from
            if len(numbers_for_choice) < config['picks']:
                logging.warning(
                    f"Not enough numbers in choice pool ({len(numbers_for_choice)}) to pick {config['picks']} main numbers for {config['file']}.")
                return None

            # Generate main numbers
            numbers = sorted(np.random.choice(
                numbers_for_choice,
                size=config['picks'],
                replace=False,  # Ensure unique numbers in the pick
                p=probabilities_for_choice
            ).tolist())

            # Generate bonus number if applicable
            if config['has_bonus']:
                bonus = None
                if 'bonus_range' in config and config['bonus_range']:  # Check if bonus_range exists and is not empty
                    bonus_numbers_range = list(range(config['bonus_range'][0], config['bonus_range'][1] + 1))
                    bonus = np.random.choice(bonus_numbers_range)
                else:
                    # Pick bonus from remaining numbers in main range (not picked for main numbers)
                    remaining_numbers = list(set(all_possible_numbers) - set(numbers))
                    if remaining_numbers:
                        bonus = np.random.choice(remaining_numbers)
                    else:
                        logging.warning(
                            f"No unique remaining numbers for bonus ball within main range for {config['file']}. Picking randomly from full range.")
                        # Fallback: pick randomly from full possible range if no unique remaining numbers
                        bonus = np.random.choice(all_possible_numbers)

                if bonus is not None:
                    numbers.append(bonus)  # Add bonus to the end of the list

            return numbers

        except Exception as e:
            logging.error(f"Error in single prediction generation for config {config}: {str(e)}", exc_info=True)
            return None

    def _calculate_prediction_confidence(self, prediction, probabilities_counts, total_predictions_count):
        """
        Calculates a confidence score for a generated prediction based on individual number probabilities.
        """
        try:
            number_confidences = []

            # Determine which lottery type this prediction belongs to by its length
            determined_lottery_type = None
            for lt, cfg in self.lottery_configs.items():
                if len(prediction) == cfg['picks'] or (cfg['has_bonus'] and len(prediction) == cfg['picks'] + 1):
                    determined_lottery_type = lt
                    break

            if not determined_lottery_type:
                logging.warning(
                    f"Could not determine lottery type from prediction length {len(prediction)} for confidence calculation.")
                return 0.0

            config = self.lottery_configs[determined_lottery_type]

            # Extract only the main numbers for confidence calculation
            main_numbers_in_prediction = prediction[:-1] if config['has_bonus'] and len(prediction) > config[
                'picks'] else prediction

            for num in main_numbers_in_prediction:
                prob = probabilities_counts.get(num, 0) / total_predictions_count
                number_confidences.append(prob)

            if not number_confidences:  # Should not happen if prediction has main numbers
                return 0.0

            return np.mean(number_confidences)

        except Exception as e:
            logging.error(f"Error calculating confidence for prediction {prediction}: {str(e)}", exc_info=True)
            return 0.0

    def _show_prediction_analysis(self, numbers, lottery_type, df):
        """
        Displays analysis of the predicted numbers.

        Args:
            numbers (list): The list of predicted numbers (including bonus if applicable).
            lottery_type (str): The type of lottery (e.g., 'Daily Lotto').
            df (pd.DataFrame): The DataFrame containing the lottery data for historical analysis.
        """

        st.write("---")
        st.subheader("Prediction Analysis")

        config = self.lottery_configs[lottery_type]
        # Separate main numbers from bonus for analysis
        main_numbers = numbers[:-1] if config['has_bonus'] else numbers

        # Calculate statistics
        stats = {
            "Sum": sum(main_numbers),
            "Mean": np.mean(main_numbers),
            "Even Numbers": sum(1 for n in main_numbers if n % 2 == 0),
            "Odd Numbers": sum(1 for n in main_numbers if n % 2 != 0),
            "Low Numbers": sum(1 for n in main_numbers if n <= config['range'][1] / 2),
            "High Numbers": sum(1 for n in main_numbers if n > config['range'][1] / 2)
        }

        # Display statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("Sum", stats["Sum"])
        col1.metric("Mean", f"{stats['Mean']:.1f}")
        col2.metric("Even/Odd", f"{stats['Even Numbers']}/{stats['Odd Numbers']}")
        col3.metric("Low/High", f"{stats['Low Numbers']}/{stats['High Numbers']}")

        # Show historical performance
        st.write("#### Historical Performance")
        # Pass the DataFrame to the historical performance method
        self._show_historical_performance(numbers, lottery_type, df)

    def _show_historical_performance(self, numbers, lottery_type, df):
        """
        Calculates and plots the historical match distribution for the prediction.

        Args:
            numbers (list): The list of predicted numbers (including bonus if applicable).
            lottery_type (str): The type of lottery.
            df (pd.DataFrame): The DataFrame containing the lottery data.
        """
        config = self.lottery_configs[lottery_type]
        # Ensure we only use the main numbers for matching
        number_set = set(numbers[:-1] if config['has_bonus'] else numbers)

        # Calculate matches with historical draws
        matches = []
        # Use the 'Numbers_List' column created during data loading
        if not df.empty:
            for historical_numbers_list in df['Numbers_List']:
                # Ensure historical_numbers_list is actually a list for set conversion
                if isinstance(historical_numbers_list, list):
                    historical_set = set(historical_numbers_list)
                    matches.append(len(number_set.intersection(historical_set)))
                else:
                    logging.warning(
                        f"Skipping non-list 'Numbers_List' entry in historical data: {historical_numbers_list}")

        # Create match distribution
        # Ensure match_dist is created even if matches is empty
        if matches:
            match_dist = pd.Series(matches).value_counts().sort_index()
        else:
            match_dist = pd.Series(dtype=int)  # Empty series if no matches could be calculated

        # Plot distribution
        if not match_dist.empty:
            fig = px.bar(
                x=match_dist.index,
                y=match_dist.values,
                labels={'x': 'Number of Matches', 'y': 'Frequency'},
                title=f'Historical Match Distribution (Total Draws: {len(df)})'
            )
            st.plotly_chart(fig)

            # Show summary statistics
            st.write(f"Maximum matches in history: {max(matches)}")
            st.write(f"Average matches in history: {np.mean(matches):.2f}")
        else:
            st.info(
                "No historical matches could be calculated for this prediction due to insufficient or invalid data, or no matching numbers found.")