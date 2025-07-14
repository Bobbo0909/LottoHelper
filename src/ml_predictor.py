# src/ml_predictor.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
import plotly.express as px
from collections import Counter
import logging
import os

# import ast # No longer needed here as parsing moves to data_loader

# --- Configuration & Logging ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CLEANED_DIR = os.path.join(DATA_DIR, 'cleaned')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, 'ml_predictor.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# --- REMOVE THE DUPLICATE load_cleaned_data_for_ml FUNCTION FROM HERE ---
# @st.cache_data(show_spinner=False)
# def load_cleaned_data_for_ml(...):
#    ... (delete this entire function as it's now in data_loader.py) ...


class MLPredictor:
    def __init__(self):
        # Lottery configurations - these define prediction ranges, not file paths directly
        self.lottery_configs = {
            'Daily Lotto': {
                'range': (1, 36),
                'picks': 5,
                'has_bonus': False
            },
            'Lotto': {
                'range': (1, 52),
                'picks': 6,
                'has_bonus': True,
                'bonus_range': (1, 52)  # Lotto bonus is within main range
            },
            'Powerball': {
                'range': (1, 50),
                'picks': 5,
                'has_bonus': True,
                'bonus_range': (1, 20)  # Powerball bonus has a separate range
            }
        }
        self.models = {}
        self.scalers = {}

    def _prepare_features(self, df, lottery_type):
        """
        Prepares features (X) and targets (y) from the DataFrame for model training.
        Assumes 'Numbers_List' column already exists and contains lists of integers.
        """
        config = self.lottery_configs[lottery_type]
        features_list = []
        targets_list = []

        if df.empty or len(df) < 2:
            logging.warning(
                f"Not enough data to prepare features for {lottery_type}. Needs at least 2 rows, has {len(df)}.")
            return np.array([]), np.array([])

        # Ensure data is sorted by Date for correct sequence
        df_sorted = df.sort_values('Date', ascending=True).reset_index(drop=True)

        for i in range(len(df_sorted) - 1):
            current_numbers = df_sorted.iloc[i]['Numbers_List']
            next_numbers = df_sorted.iloc[i + 1]['Numbers_List']

            # Ensure both current and next numbers are valid lists of expected length
            if not isinstance(current_numbers, list) or len(current_numbers) != config['picks']:
                logging.warning(f"Skipping row {i} due to invalid current_numbers: {current_numbers}")
                continue
            if not isinstance(next_numbers, list) or len(next_numbers) != config['picks']:
                logging.warning(f"Skipping row {i} due to invalid next_numbers: {next_numbers}")
                continue

            feature_vector = self._extract_features(current_numbers, config)
            if feature_vector is not None:
                features_list.append(feature_vector)
                targets_list.append(next_numbers)

        if not features_list:
            logging.error(f"No valid feature-target pairs could be prepared for {lottery_type}. "
                          f"Check Numbers_List completeness and validity.")
            return np.array([]), np.array([])

        return np.array(features_list), np.array(targets_list)

    @staticmethod
    def _extract_features(numbers, config):
        """
        Extracts various features from a list of lottery numbers.
        Returns None if numbers list is invalid/empty.
        """
        if not numbers or not isinstance(numbers, list) or len(numbers) != config['picks']:
            logging.warning(
                f"Attempted to extract features from invalid numbers list: {numbers}. Expected {config['picks']} numbers.")
            return None

        try:
            features = []
            max_num = config['range'][1]

            # One-hot encoding for numbers
            number_features = np.zeros(max_num)
            for n in numbers:
                if 1 <= n <= max_num:
                    number_features[n - 1] = 1
            features.extend(number_features.tolist())  # Convert to list to extend

            # Statistical features
            features.extend([
                np.mean(numbers),
                np.std(numbers),
                np.median(numbers),
                (max(numbers) - min(numbers)) if len(numbers) > 1 else 0  # Range
            ])

            # Parity and range features
            features.extend([
                sum(1 for n in numbers if n % 2 == 0) / config['picks'],  # Even ratio
                sum(1 for n in numbers if n <= max_num / 2) / config['picks'],  # Low ratio
                len(set(n // 10 for n in numbers)) / ((max_num // 10) + 1)  # Distribution across tens
            ])

            return features

        except Exception as err:
            logging.error(f"Error in feature extraction for numbers {numbers}: {str(err)}", exc_info=True)
            return None

    def train_model(self, lottery_type, df):
        """
        Trains machine learning models for the specified lottery type.

        Args:
            lottery_type (str): The type of lottery (e.g., 'Daily Lotto').
            df (pd.DataFrame): The DataFrame containing the lottery data, with 'Numbers_List' column.
        Returns:
            bool: True if models were trained successfully, False otherwise.
        """
        if df.empty:
            logging.error(f"train_model: No data available for training {lottery_type}.")
            return False

        try:
            features_data, targets_data = self._prepare_features(df, lottery_type)

            if features_data.shape[0] < 2 or features_data.shape[1] == 0:
                logging.error(
                    f"train_model: Insufficient or invalid data for training for {lottery_type} after feature preparation. "
                    f"Features shape: {features_data.shape}. Requires at least 2 samples with features.")
                return False

            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_data)
            self.scalers[lottery_type] = scaler

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
                    multi_model = MultiOutputClassifier(estimator=model, n_jobs=-1)
                    multi_model.fit(features_scaled, targets_data)
                    score = multi_model.score(features_scaled, targets_data)
                    trained_models[name] = multi_model
                    model_scores[name] = score
                    logging.info(f"train_model: Trained {name} model for {lottery_type}. Score: {score:.2%}")
                except Exception as model_train_err:
                    logging.error(
                        f"train_model: Error training {name} model for {lottery_type}: {str(model_train_err)}",
                        exc_info=True)
                    continue

            if trained_models:
                self.models[lottery_type] = {
                    'models': trained_models,
                    'scores': model_scores
                }
                return True
            else:
                logging.error(f"train_model: Failed to train any models for {lottery_type}.")
                return False

        except Exception as err:
            logging.error(f"train_model: Unexpected error during model training for {lottery_type}: {str(err)}",
                          exc_info=True)
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
            logging.error(f"generate_predictions: Models not trained for {lottery_type}.")
            return []
        if df.empty:
            logging.error(
                f"generate_predictions: No data available to get latest draw for prediction for {lottery_type}.")
            return []

        config = self.lottery_configs[lottery_type]
        predictions = []

        try:
            # Ensure df is sorted by date to get the absolute latest draw
            df_sorted = df.sort_values('Date', ascending=True).reset_index(drop=True)
            latest_numbers_list = df_sorted.iloc[-1]['Numbers_List']

            if not isinstance(latest_numbers_list, list) or len(latest_numbers_list) != config['picks']:
                logging.error(
                    f"generate_predictions: Latest draw numbers list is invalid or empty ({latest_numbers_list}), cannot generate features for prediction.")
                return []

            features = self._extract_features(latest_numbers_list, config)

            if features is None:
                logging.error("generate_predictions: Failed to extract features from latest draw.")
                return []

            # Ensure features_scaled is 2D (1, n_features)
            features_scaled = self.scalers[lottery_type].transform([features])

            all_predictions_from_models = []
            for model_name, model in self.models[lottery_type]['models'].items():
                prediction_result = model.predict(features_scaled)[0]
                all_predictions_from_models.extend(prediction_result.tolist())  # Ensure it's a list

            # Filter out numbers not in the valid range for the lottery
            valid_range_min, valid_range_max = config['range']
            all_predictions_from_models = [
                n for n in all_predictions_from_models if valid_range_min <= n <= valid_range_max
            ]

            if not all_predictions_from_models:
                logging.warning(
                    f"No valid numbers predicted by models for {lottery_type} within range {config['range']}.")
                return []

            number_probabilities_counts = Counter(all_predictions_from_models)
            total_predictions_count = sum(number_probabilities_counts.values())  # Sum of all counts
            if total_predictions_count == 0:  # Avoid division by zero
                logging.warning("Total predictions count from models is zero, cannot calculate probabilities.")
                return []

            generated_sets = set()
            attempts = 0
            max_attempts = num_predictions * 50  # Increase attempts to find unique sets
            while len(predictions) < num_predictions and attempts < max_attempts:
                single_prediction = self._generate_single_prediction(
                    number_probabilities_counts,
                    total_predictions_count,
                    config
                )
                if single_prediction is not None:
                    # Sort the prediction list itself before converting to tuple for consistent hashing
                    single_prediction_sorted = sorted(single_prediction)
                    prediction_tuple = tuple(single_prediction_sorted)

                    # Separate bonus for generating unique main numbers sets
                    main_numbers_tuple = tuple(single_prediction_sorted[:-1]) if config['has_bonus'] and len(
                        single_prediction_sorted) > config['picks'] else prediction_tuple

                    if main_numbers_tuple not in generated_sets:  # Check only main numbers for uniqueness
                        generated_sets.add(main_numbers_tuple)
                        confidence = self._calculate_prediction_confidence(
                            single_prediction_sorted,  # Pass sorted list to confidence
                            number_probabilities_counts,
                            total_predictions_count
                        )
                        predictions.append((single_prediction_sorted, confidence))
                attempts += 1

            if not predictions and num_predictions > 0:
                logging.warning(
                    f"generate_predictions: Could not generate {num_predictions} unique predictions for {lottery_type} after {attempts} attempts.")
            return sorted(predictions, key=lambda x: x[1], reverse=True)

        except Exception as err:
            logging.error(f"generate_predictions: Error generating predictions for {lottery_type}: {str(err)}",
                          exc_info=True)
            return []

    @staticmethod
    def _generate_single_prediction(probabilities_counts, total_predictions_count, config):
        """
        Generates a single set of lottery numbers based on probabilities.
        """
        try:
            all_possible_numbers = list(range(config['range'][0], config['range'][1] + 1))

            probs_from_ml = {
                num: count / total_predictions_count
                for num, count in probabilities_counts.items()
            }

            final_probs = {}
            # Ensure a small base probability for all numbers to allow them to be picked
            # This helps avoid zero probability issues for numbers that models rarely predict but are valid
            min_non_zero_prob_from_ml = min(p for p in probs_from_ml.values() if p > 0) if any(
                p > 0 for p in probs_from_ml.values()) else 0.001
            base_prob = min_non_zero_prob_from_ml / 10  # A smaller base to favor predicted numbers

            for num in all_possible_numbers:
                final_probs[num] = probs_from_ml.get(num, base_prob)  # Assign base_prob to unpredicted numbers

            total_final_prob = sum(final_probs.values())
            if total_final_prob == 0:
                logging.warning(
                    "_generate_single_prediction: All final probabilities sum to zero. Using uniform distribution for prediction.")
                final_probs = {num: 1 for num in all_possible_numbers}  # Assign uniform weight
                total_final_prob = sum(final_probs.values())  # Re-sum for normalization

            # Normalize probabilities
            probabilities_for_choice = [final_probs[num] / total_final_prob for num in all_possible_numbers]
            numbers_for_choice = all_possible_numbers  # Ensure same order as probabilities

            if sum(probabilities_for_choice) < 0.99 or sum(probabilities_for_choice) > 1.01:  # Check sum close to 1
                logging.warning(f"Normalized probabilities sum is not 1: {sum(probabilities_for_choice)}. Adjusting.")
                probabilities_for_choice = [p / sum(probabilities_for_choice) for p in
                                            probabilities_for_choice]  # Re-normalize if off

            if len(numbers_for_choice) < config['picks']:
                logging.error(
                    f"_generate_single_prediction: Not enough numbers in choice pool ({len(numbers_for_choice)}) to pick {config['picks']} main numbers for {config.get('file', lottery_type)}.")
                return None

            # Use np.random.choice with calculated probabilities
            numbers = sorted(np.random.choice(
                numbers_for_choice,
                size=config['picks'],
                replace=False,
                p=probabilities_for_choice
            ).tolist())

            if config['has_bonus']:
                bonus_number = None
                # Check for specific bonus range first
                if 'bonus_range' in config and config['bonus_range']:
                    bonus_numbers_range = list(range(config['bonus_range'][0], config['bonus_range'][1] + 1))
                    # Pick bonus from its specific range, not overlapping with main numbers
                    # Ensure bonus number is not already in main numbers if ranges overlap
                    potential_bonus_numbers = [n for n in bonus_numbers_range if n not in numbers]
                    if potential_bonus_numbers:
                        bonus_number = np.random.choice(potential_bonus_numbers)
                    else:  # If all bonus range numbers are already in main numbers (unlikely)
                        bonus_number = np.random.choice(bonus_numbers_range)  # Fallback to picking any
                else:  # If no specific bonus range, pick from remaining main range numbers
                    remaining_numbers = list(set(all_possible_numbers) - set(numbers))
                    if remaining_numbers:
                        bonus_number = np.random.choice(remaining_numbers)
                    else:
                        logging.warning(
                            f"_generate_single_prediction: No unique remaining numbers for bonus ball within main range for {lottery_type}. Picking randomly from full range.")
                        bonus_number = np.random.choice(all_possible_numbers)  # Fallback

                if bonus_number is not None:
                    numbers.append(bonus_number)

            return numbers

        except Exception as err:
            logging.error(
                f"_generate_single_prediction: Error in single prediction generation for config {config}: {str(err)}",
                exc_info=True)
            return None

    def _calculate_prediction_confidence(self, prediction, probabilities_counts, total_predictions_count):
        """
        Calculates a confidence score for a generated prediction based on individual number probabilities.
        """
        try:
            if not prediction:
                return 0.0

            # Determine lottery type based on prediction length and configs
            determined_lottery_type_name = None
            for lt_name, cfg in self.lottery_configs.items():
                expected_length = cfg['picks']
                if cfg['has_bonus']:
                    expected_length += 1
                if len(prediction) == expected_length:
                    determined_lottery_type_name = lt_name
                    break

            if not determined_lottery_type_name:
                logging.warning(
                    f"Could not determine lottery type from prediction length {len(prediction)} for confidence calculation. Prediction: {prediction}")
                return 0.0

            config = self.lottery_configs[determined_lottery_type_name]

            # Split main and bonus numbers if applicable
            main_numbers_in_prediction = prediction
            if config['has_bonus'] and len(prediction) > config['picks']:
                # The bonus number is always the last one if we appended it
                main_numbers_in_prediction = prediction[:-1]
            elif config['has_bonus'] and len(prediction) == config['picks']:
                # This case means bonus might not have been appended, handle if it's supposed to be there.
                # For confidence, we only care about the 'picks' main numbers.
                pass

            if not main_numbers_in_prediction:
                return 0.0

            number_confidences = []
            for num in main_numbers_in_prediction:
                # Ensure the number is within the range of probabilities_counts keys
                if num in probabilities_counts:
                    prob = probabilities_counts.get(num, 0) / total_predictions_count
                    number_confidences.append(prob)
                else:
                    # If a number in prediction wasn't in any model's raw output, give it a very low confidence
                    number_confidences.append(1e-6 / total_predictions_count)  # A very small non-zero value

            if not number_confidences:
                return 0.0

            return np.mean(number_confidences)

        except Exception as err:
            logging.error(f"Error calculating confidence for prediction {prediction}: {str(err)}", exc_info=True)
            return 0.0

    def show_prediction_analysis(self, numbers, lottery_type, df):
        """
        Displays analysis of the predicted numbers.
        """

        st.write("---")
        st.subheader("Prediction Analysis")

        config = self.lottery_configs[lottery_type]

        # Determine if bonus is included and extract main numbers
        main_numbers = []
        if config['has_bonus'] and len(numbers) > config['picks']:
            main_numbers = numbers[:-1]
            bonus_number = numbers[-1]
            st.write(f"**Bonus Number:** {bonus_number}")
        else:
            main_numbers = numbers

        if not main_numbers:
            st.warning("Cannot perform analysis: No main numbers found in prediction.")
            return

        stats = {
            "Sum": sum(main_numbers),
            "Mean": np.mean(main_numbers),
            "Even Numbers": sum(1 for n in main_numbers if n % 2 == 0),
            "Odd Numbers": sum(1 for n in main_numbers if n % 2 != 0),
            "Low Numbers": sum(1 for n in main_numbers if n <= config['range'][1] / 2),
            "High Numbers": sum(1 for n in main_numbers if n > config['range'][1] / 2)
        }

        col1, col2, col3 = st.columns(3)
        col1.metric("Sum", stats["Sum"])
        col1.metric("Mean", f"{stats['Mean']:.1f}")
        col2.metric("Even/Odd", f"{stats['Even Numbers']}/{stats['Odd Numbers']}")
        col3.metric("Low/High", f"{stats['Low Numbers']}/{stats['High Numbers']}")

        st.write("#### Historical Performance")
        self.show_historical_performance(numbers, lottery_type, df)

    def show_historical_performance(self, numbers, lottery_type, df):
        """
        Calculates and plots the historical match distribution for the prediction.
        """
        config = self.lottery_configs[lottery_type]

        # Ensure that 'numbers' passed here matches the structure expected
        # If it includes bonus, separate it for main number matching
        main_prediction_set = set(numbers[:-1] if config['has_bonus'] and len(numbers) > config['picks'] else numbers)

        if not main_prediction_set:
            st.info("No main numbers in prediction to check historical performance.")
            return

        matches = []
        if not df.empty and 'Numbers_List' in df.columns:
            for historical_numbers_list in df['Numbers_List']:
                if isinstance(historical_numbers_list, list):
                    historical_set = set(historical_numbers_list)
                    matches.append(len(main_prediction_set.intersection(historical_set)))
                else:
                    logging.warning(
                        f"Skipping non-list 'Numbers_List' entry in historical data for historical performance: {historical_numbers_list}")

        if matches:
            match_dist = pd.Series(matches).value_counts().sort_index()
        else:
            match_dist = pd.Series(dtype=int)  # Empty series

        if not match_dist.empty:
            fig = px.bar(
                x=match_dist.index,
                y=match_dist.values,
                labels={'x': 'Number of Matches', 'y': 'Frequency'},
                title=f'Historical Match Distribution (Total Draws: {len(df)})'
            )
            st.plotly_chart(fig)

            st.write(f"Maximum matches in history: {max(matches)}")
            st.write(f"Average matches in history: {np.mean(matches):.2f}")
        else:
            st.info(
                "No historical matches could be calculated for this prediction due to insufficient or invalid data, or no matching numbers found.")