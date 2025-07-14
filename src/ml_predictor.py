# src/ml_predictor.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
from collections import Counter
import logging
import os
from typing import List, Tuple, Dict, Optional, Any
import hashlib

# --- Constants ---
MIN_TRAINING_SAMPLES = 10
DEFAULT_NUM_PREDICTIONS = 5
MAX_PREDICTION_ATTEMPTS_MULTIPLIER = 50
DEFAULT_WINDOW_SIZE = 5
DEFAULT_TEMPERATURE = 1.0
TRAIN_TEST_SPLIT_RATIO = 0.2
MIN_PROBABILITY = 1e-6

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

# Lottery prediction disclaimer
LOTTERY_DISCLAIMER = """
⚠️ **Important Notice**: Lottery draws are designed to be completely random. 
This ML predictor identifies patterns in historical data, but these patterns 
do not guarantee future results. Please play responsibly.
"""


class MLPredictor:
    def __init__(self):
        # Lottery configurations - these define prediction ranges
        # Note: These should ideally be synced with data_loader.LOTTERY_CONFIGS
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
        self.model_scores = {}
        # Internal cache for @st.cache_resource
        self._model_cache = {}

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validates that the DataFrame has required columns and structure."""
        required_columns = ['Date', 'Numbers_List']
        if not all(col in df.columns for col in required_columns):
            logging.error(
                f"DataFrame missing required columns. Required: {required_columns}, Found: {df.columns.tolist()}")
            return False

        if df.empty:
            logging.error("DataFrame is empty")
            return False

        # Check if Numbers_List contains valid data
        # Ensure there's at least one non-null entry to sample
        if df['Numbers_List'].isnull().all():
            logging.error("Numbers_List column is entirely null.")
            return False

        # Find the first non-null sample
        sample_series = df['Numbers_List'].dropna()
        if not sample_series.empty:
            sample = sample_series.iloc[0]
            if not isinstance(sample, list):
                logging.error(f"Numbers_List column does not contain lists. Sample: {sample}")
                return False
        else:
            logging.warning("Numbers_List column has no non-null entries to validate type.")
            # If all are null but not empty, this might be okay if handled upstream.
            # For now, let's allow it to pass if no non-null sample to avoid false negatives.
            pass

        return True

    def _prepare_features_with_window(self, df: pd.DataFrame, lottery_type: str,
                                      window_size: int = DEFAULT_WINDOW_SIZE) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares features using a sliding window of recent draws.

        Args:
            df: DataFrame with lottery data
            lottery_type: Type of lottery
            window_size: Number of recent draws to use as features

        Returns:
            Tuple of (features, targets) arrays
        """
        config = self.lottery_configs[lottery_type]
        features_list = []
        targets_list = []

        if not self._validate_dataframe(df):
            return np.array([]), np.array([])

        # Ensure data is sorted by Date
        df_sorted = df.sort_values('Date', ascending=True).reset_index(drop=True)

        if len(df_sorted) < window_size + 1:
            logging.warning(
                f"Not enough data ({len(df_sorted)} rows) for window size {window_size}. Need at least {window_size + 1} rows. Falling back to single-draw features.")
            return self._prepare_features(df_sorted, lottery_type)  # Pass sorted df

        for i in range(len(df_sorted) - window_size):
            window_features = []
            valid_window = True

            # Extract features from each draw in the window
            for j in range(window_size):
                draw_numbers = df_sorted.iloc[i + j]['Numbers_List']
                if not isinstance(draw_numbers, list) or len(draw_numbers) != config['picks']:
                    logging.warning(
                        f"Skipping window at position {i} due to invalid draw at offset {j}: {draw_numbers}")
                    valid_window = False
                    break

                draw_features = self._extract_features(draw_numbers, config)
                if draw_features is None:
                    valid_window = False
                    break
                window_features.extend(draw_features)

            if valid_window:
                # All draws in window were valid, now get the target (next draw)
                next_numbers = df_sorted.iloc[i + window_size]['Numbers_List']
                if isinstance(next_numbers, list) and len(next_numbers) == config['picks']:
                    features_list.append(window_features)
                    targets_list.append(next_numbers)
                else:
                    logging.warning(
                        f"Skipping target at position {i + window_size} due to invalid numbers: {next_numbers}")

        if not features_list:
            logging.warning(f"No valid windows found for {lottery_type}. Falling back to single-draw features.")
            return self._prepare_features(df_sorted, lottery_type)  # Pass sorted df

        return np.array(features_list), np.array(targets_list)

    def _prepare_features(self, df: pd.DataFrame, lottery_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares features (X) and targets (y) from the DataFrame for model training.
        Assumes 'Numbers_List' column already exists and contains lists of integers.
        """
        config = self.lottery_configs[lottery_type]
        features_list = []
        targets_list = []

        if not self._validate_dataframe(df):
            return np.array([]), np.array([])

        if len(df) < MIN_TRAINING_SAMPLES:
            logging.warning(
                f"Not enough data to prepare features for {lottery_type}. Needs at least {MIN_TRAINING_SAMPLES} rows, has {len(df)}.")
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
            logging.error(f"No valid feature-target pairs could be prepared for {lottery_type}.")
            return np.array([]), np.array([])

        return np.array(features_list), np.array(targets_list)

    @staticmethod
    def _extract_features(numbers: List[int], config: Dict[str, Any]) -> Optional[List[float]]:
        """
        Extracts various features from a list of lottery numbers.
        Returns None if numbers list is invalid/empty.
        """
        if not numbers or not isinstance(numbers, list) or len(numbers) != config['picks']:
            logging.warning(f"Invalid numbers list: {numbers}. Expected {config['picks']} numbers.")
            return None

        try:
            features = []
            max_num = config['range'][1]

            # One-hot encoding for numbers
            number_features = np.zeros(max_num)
            for n in numbers:
                if 1 <= n <= max_num:
                    number_features[n - 1] = 1
            features.extend(number_features.tolist())

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

            # Additional pattern features
            features.extend([
                len(set(numbers)) / config['picks'],  # Uniqueness ratio (should be 1 for valid draws)
                np.var(numbers),  # Variance
                sum(1 for i in range(len(numbers) - 1) if numbers[i + 1] - numbers[i] == 1) / max(config['picks'] - 1,
                                                                                                  1)
                # Consecutive ratio
            ])

            return features

        except Exception as err:
            logging.error(f"Error in feature extraction for numbers {numbers}: {str(err)}", exc_info=True)
            return None

    @st.cache_resource
    def _get_cached_model(_self, lottery_type: str, data_hash: str):
        """Returns cached model if data hasn't changed, otherwise returns None."""
        cache_key = f"{lottery_type}_{data_hash}"
        if cache_key in _self._model_cache:
            logging.info(f"Retrieving cached model for {lottery_type} with hash {data_hash[:8]}...")
            return _self._model_cache[cache_key]
        logging.info(f"No cached model found for {lottery_type} with hash {data_hash[:8]}.")
        return None

    def _cache_model(self, lottery_type: str, data_hash: str, models: Dict, scalers: Any, scores: Dict):
        """Caches the trained model."""
        cache_key = f"{lottery_type}_{data_hash}"
        self._model_cache[cache_key] = {
            'models': models,
            'scalers': scalers,
            'scores': scores
        }
        logging.info(f"Model for {lottery_type} with hash {data_hash[:8]} cached successfully.")

    def train_model(self, lottery_type: str, df: pd.DataFrame, use_window: bool = True,
                    window_size: int = DEFAULT_WINDOW_SIZE) -> bool:
        """
        Trains machine learning models for the specified lottery type.

        Args:
            lottery_type: The type of lottery (e.g., 'Daily Lotto').
            df: The DataFrame containing the lottery data, with 'Numbers_List' column.
            use_window: Whether to use windowed features
            window_size: Size of the window for feature extraction

        Returns:
            bool: True if models were trained successfully, False otherwise.
        """
        if not self._validate_dataframe(df):
            logging.error(f"Validation failed for DataFrame of {lottery_type}.")
            return False

        # Create a copy of the DataFrame for hashing to avoid modifying the original df
        # Convert 'Numbers_List' column from list to tuple for hashing compatibility
        # Tuples are immutable and therefore hashable.
        df_for_hashing = df.copy()
        if 'Numbers_List' in df_for_hashing.columns:
            try:
                df_for_hashing['Numbers_List'] = df_for_hashing['Numbers_List'].apply(
                    lambda x: tuple(x) if isinstance(x, list) else x
                )
            except Exception as e:
                logging.warning(
                    f"Could not convert 'Numbers_List' to tuples for hashing: {e}. Attempting hash on original df, may fail.")
                pass

                # Create a hash of the data to check if we need to retrain
        data_hash = hashlib.md5(pd.util.hash_pandas_object(df_for_hashing, index=True).values).hexdigest()

        # Check cache
        cached = self._get_cached_model(lottery_type, data_hash)
        if cached:
            self.models[lottery_type] = cached['models']
            self.scalers[lottery_type] = cached['scalers']
            self.model_scores[lottery_type] = cached['scores']
            logging.info(f"Using cached model for {lottery_type}")
            return True

        try:
            # Prepare features with or without window
            if use_window:
                features_data, targets_data = self._prepare_features_with_window(df, lottery_type, window_size)
            else:
                features_data, targets_data = self._prepare_features(df, lottery_type)

            if features_data.shape[0] < MIN_TRAINING_SAMPLES or features_data.shape[1] == 0:
                logging.error(
                    f"Insufficient data for training {lottery_type}. Features shape: {features_data.shape}. Need at least {MIN_TRAINING_SAMPLES} samples.")
                return False

            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                features_data, targets_data,
                test_size=TRAIN_TEST_SPLIT_RATIO,
                random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[lottery_type] = scaler

            # Define models
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
                    multi_model.fit(X_train_scaled, y_train)

                    # Calculate scores on test set
                    train_score = multi_model.score(X_train_scaled, y_train)
                    test_score = multi_model.score(X_test_scaled, y_test)

                    trained_models[name] = multi_model
                    model_scores[name] = {
                        'train': train_score,
                        'test': test_score
                    }
                    logging.info(
                        f"Trained {name} model for {lottery_type}. Train: {train_score:.2%}, Test: {test_score:.2%}")
                except Exception as model_train_err:
                    # CORRECTED LINE HERE: lottery_err -> lottery_type
                    logging.error(f"Error training {name} model for {lottery_type}: {str(model_train_err)}",
                                  exc_info=True)
                    continue

            if trained_models:
                self.models[lottery_type] = {
                    'models': trained_models,
                    'scores': model_scores
                }
                self.model_scores[lottery_type] = model_scores

                # Cache the model
                self._cache_model(lottery_type, data_hash, trained_models, scaler, model_scores)

                return True
            else:
                logging.error(f"Failed to train any models for {lottery_type}.")
                return False

        except Exception as err:
            logging.error(f"Unexpected error during model training for {lottery_type}: {str(err)}", exc_info=True)
            return False

    def _apply_temperature_scaling(self, probabilities: Dict[int, float], temperature: float = DEFAULT_TEMPERATURE) -> \
    Dict[int, float]:
        """
        Apply temperature scaling to probabilities to control prediction diversity.

        Args:
            probabilities: Dictionary of number -> probability
            temperature: Temperature parameter (higher = more diverse, lower = more focused)

        Returns:
            Temperature-scaled probabilities
        """
        if temperature <= 0:  # Handle temperature=0 or negative by returning uniform distribution
            logging.warning(f"Temperature is non-positive ({temperature}), returning uniform distribution.")
            if not probabilities: return {}
            uniform_prob = 1.0 / len(probabilities)
            return {num: uniform_prob for num in probabilities}

        # Convert to numpy array for easier manipulation
        numbers = list(probabilities.keys())
        probs = np.array(list(probabilities.values()))

        # Apply temperature scaling
        # Avoid issues with log(0) by adding a small epsilon or clipping
        probs = np.clip(probs, MIN_PROBABILITY, 1.0)  # Ensure probabilities are never zero

        if temperature != 1.0:
            probs = np.power(probs, 1 / temperature)
            probs = probs / probs.sum()  # Re-normalize after scaling

        return dict(zip(numbers, probs))

    def generate_predictions(self, lottery_type: str, df: pd.DataFrame, num_predictions: int = DEFAULT_NUM_PREDICTIONS,
                             temperature: float = DEFAULT_TEMPERATURE, use_window: bool = True) -> List[
        Tuple[List[int], float]]:
        """
        Generates lottery number predictions using the trained models.

        Args:
            lottery_type: The type of lottery.
            df: The DataFrame containing the lottery data.
            num_predictions: Number of prediction sets to generate.
            temperature: Temperature for probability scaling (higher = more diverse).
            use_window: Whether to use windowed features for prediction.

        Returns:
            List of tuples, each containing a predicted set of numbers and its confidence.
        """
        if lottery_type not in self.models or not self.models[lottery_type]['models']:
            logging.error(f"Models not trained or available for {lottery_type}.")
            return []

        if not self._validate_dataframe(df):
            logging.error(f"Invalid DataFrame for prediction generation for {lottery_type}.")
            return []

        config = self.lottery_configs[lottery_type]
        predictions = []

        try:
            # Ensure df is sorted by date to get the latest draws
            df_sorted = df.sort_values('Date', ascending=True).reset_index(drop=True)

            # Prepare features for prediction using the latest data point(s)
            features = None
            if use_window and len(df_sorted) >= DEFAULT_WINDOW_SIZE:
                window_features = []
                for i in range(DEFAULT_WINDOW_SIZE):
                    draw_idx = len(df_sorted) - DEFAULT_WINDOW_SIZE + i
                    draw_numbers = df_sorted.iloc[draw_idx]['Numbers_List']
                    draw_features = self._extract_features(draw_numbers, config)
                    if draw_features is None:
                        logging.error(
                            f"Failed to extract features from draw at index {draw_idx} for windowed prediction.")
                        return []
                    window_features.extend(draw_features)
                features = window_features
            elif len(df_sorted) > 0:  # Fallback to last single draw if not enough for window
                latest_numbers_list = df_sorted.iloc[-1]['Numbers_List']
                if not isinstance(latest_numbers_list, list) or len(latest_numbers_list) != config['picks']:
                    logging.error(f"Latest draw numbers invalid for single-draw prediction: {latest_numbers_list}")
                    return []
                features = self._extract_features(latest_numbers_list, config)
            else:
                logging.error("No data available to generate features for prediction.")
                return []

            if features is None:
                logging.error("Failed to extract features from latest draw(s).")
                return []

            # Scale features
            features_scaled = self.scalers[lottery_type].transform([features])

            # Collect predictions from all models
            all_predictions_from_models = []
            for model_name, model in self.models[lottery_type]['models'].items():
                prediction_result = model.predict(features_scaled)[0]
                all_predictions_from_models.extend(prediction_result.tolist())

            # Filter valid numbers and calculate probabilities
            valid_range_min, valid_range_max = config['range']
            all_predictions_from_models = [
                n for n in all_predictions_from_models
                if valid_range_min <= n <= valid_range_max
            ]

            if not all_predictions_from_models:
                logging.warning(f"No valid numbers predicted by models for {lottery_type}")
                return []

            # Calculate base probabilities
            number_counts = Counter(all_predictions_from_models)
            total_count = sum(number_counts.values())

            # Create probability distribution for all possible numbers
            all_numbers = list(range(valid_range_min, valid_range_max + 1))

            # Initialize with a small base probability for all valid numbers
            probabilities = {num: MIN_PROBABILITY for num in all_numbers}

            # Add counts from model predictions
            for num, count in number_counts.items():
                probabilities[num] += count  # Add count directly, will normalize later

            # Normalize and apply temperature scaling
            total_sum_for_normalization = sum(probabilities.values())
            if total_sum_for_normalization > 0:
                probabilities = {num: p / total_sum_for_normalization for num, p in probabilities.items()}
            else:
                logging.warning("Total probability sum is zero before temperature scaling, cannot normalize.")
                return []

            probabilities = self._apply_temperature_scaling(probabilities, temperature)

            # Generate unique predictions
            generated_sets = set()
            attempts = 0
            max_attempts = num_predictions * MAX_PREDICTION_ATTEMPTS_MULTIPLIER

            while len(predictions) < num_predictions and attempts < max_attempts:
                single_prediction = self._generate_single_prediction(probabilities, config)

                if single_prediction is not None:
                    # Sort for consistent comparison
                    # For lotteries with bonus, ensure main numbers are sorted and bonus is at the end
                    main_part = sorted(single_prediction[:config['picks']])
                    bonus_part = [single_prediction[-1]] if config['has_bonus'] and len(single_prediction) > config[
                        'picks'] else []
                    single_prediction_for_comparison = tuple(main_part + bonus_part)

                    if single_prediction_for_comparison not in generated_sets:
                        generated_sets.add(single_prediction_for_comparison)
                        confidence = self._calculate_prediction_confidence(
                            single_prediction, number_counts, total_count, config
                        )
                        predictions.append(
                            (single_prediction, confidence))  # Store unsorted, or original generated order if desired
                attempts += 1

            if not predictions and num_predictions > 0:
                logging.warning(f"Could not generate {num_predictions} unique predictions after {attempts} attempts.")

            return sorted(predictions, key=lambda x: x[1], reverse=True)

        except Exception as err:
            logging.error(f"Error generating predictions for {lottery_type}: {str(err)}", exc_info=True)
            return []

    def _generate_single_prediction(self, probabilities: Dict[int, float], config: Dict[str, Any]) -> Optional[
        List[int]]:
        """
        Generates a single set of lottery numbers based on probabilities.

        Args:
            probabilities: Dictionary mapping numbers to their probabilities
            config: Lottery configuration

        Returns:
            List of predicted numbers including bonus if applicable
        """
        try:
            numbers = list(probabilities.keys())
            probs = list(probabilities.values())

            # Ensure probabilities sum to 1
            probs = np.array(probs)
            probs = probs / probs.sum()

            if len(numbers) < config['picks']:
                logging.error(f"Not enough numbers to pick from: {len(numbers)} < {config['picks']}")
                return None

            # Select main numbers
            main_numbers_raw = np.random.choice(
                numbers,
                size=config['picks'],
                replace=False,
                p=probs
            ).tolist()
            main_numbers = sorted(main_numbers_raw)  # Always sort main numbers

            # Add bonus number if required
            final_prediction = list(main_numbers)  # Start with sorted main numbers
            if config['has_bonus']:
                bonus_candidates = []
                if 'bonus_range' in config and config['bonus_range']:
                    # Bonus has specific range
                    bonus_min, bonus_max = config['bonus_range']
                    bonus_candidates = list(range(bonus_min, bonus_max + 1))
                else:
                    # Bonus from same range as main numbers
                    bonus_candidates = list(range(config['range'][0], config['range'][1] + 1))

                # Filter out numbers already in main selection to ensure bonus is unique
                available_bonus_candidates = [n for n in bonus_candidates if n not in main_numbers]

                if available_bonus_candidates:
                    # If candidates available, use their probabilities if they exist, otherwise uniform
                    available_probs = [probabilities.get(n, MIN_PROBABILITY) for n in available_bonus_candidates]
                    available_probs_sum = sum(available_probs)
                    if available_probs_sum > 0:
                        normalized_available_probs = [p / available_probs_sum for p in available_probs]
                        bonus_number = np.random.choice(available_bonus_candidates, p=normalized_available_probs)
                    else:  # Fallback if all probabilities for available numbers are zero
                        bonus_number = np.random.choice(available_bonus_candidates)
                else:
                    # If no unique bonus number available (highly unlikely for typical lottery rules), pick one from main range
                    # This could happen if all numbers in the bonus range are already in main_numbers
                    logging.warning(
                        "No unique bonus number candidates available. Picking from main range regardless of uniqueness.")
                    bonus_number = np.random.choice(list(range(config['range'][0], config['range'][1] + 1)))

                final_prediction.append(bonus_number)  # Append bonus number at the end

            return final_prediction

        except Exception as err:
            logging.error(f"Error generating single prediction: {str(err)}", exc_info=True)
            return None

    def _calculate_prediction_confidence(self, prediction: List[int], number_counts: Counter,
                                         total_count: int, config: Dict[str, Any]) -> float:
        """
        Calculates a confidence score for a generated prediction.

        Args:
            prediction: The predicted numbers (can include bonus)
            number_counts: Counts of numbers from model predictions
            total_count: Total number of predictions from models
            config: Lottery configuration

        Returns:
            Confidence score between 0 and 1
        """
        try:
            if not prediction or total_count == 0:
                return 0.0

            # Focus on main numbers for confidence calculation
            main_numbers = prediction[:config['picks']]

            # Calculate confidence based on how often models predicted these numbers
            confidences = []
            for num in main_numbers:
                if num in number_counts:
                    confidence = number_counts[num] / total_count
                else:
                    confidence = MIN_PROBABILITY  # Assign a very small probability if not directly predicted
                confidences.append(confidence)

            # Return average confidence
            return np.mean(confidences) if confidences else 0.0

        except Exception as err:
            logging.error(f"Error calculating confidence: {str(err)}", exc_info=True)
            return 0.0

    def show_prediction_analysis(self, numbers: List[int], lottery_type: str, df: pd.DataFrame):
        """
        Displays analysis of the predicted numbers.

        Args:
            numbers: The predicted numbers
            lottery_type: Type of lottery
            df: Historical data DataFrame
        """
        st.write("---")
        st.subheader("Prediction Analysis")

        # Show disclaimer
        st.info(LOTTERY_DISCLAIMER)

        config = self.lottery_configs[lottery_type]

        # Separate main and bonus numbers for display
        main_numbers_display = sorted(numbers[:config['picks']])
        if config['has_bonus'] and len(numbers) > config['picks']:
            bonus_number_display = numbers[-1]
            st.write(f"**Main Numbers:** {', '.join(map(str, main_numbers_display))}")
            st.write(f"**Bonus Number:** {bonus_number_display}")
        else:
            st.write(f"**Numbers:** {', '.join(map(str, main_numbers_display))}")

        # Show model performance if available
        if lottery_type in self.model_scores:
            st.write("#### Model Performance")
            scores = self.model_scores[lottery_type]

            cols = st.columns(len(scores))
            for idx, (model_name, score_data) in enumerate(scores.items()):
                with cols[idx]:
                    st.metric(
                        f"{model_name.upper()} Model",
                        f"Test: {score_data['test']:.1%}",
                        f"Train: {score_data['train']:.1%}"
                    )

        # Statistical analysis
        if main_numbers_display:
            st.write("#### Statistical Analysis")

            stats = {
                "Sum": sum(main_numbers_display),
                "Mean": np.mean(main_numbers_display),
                "Median": np.median(main_numbers_display),
                "Std Dev": np.std(main_numbers_display),
                "Range": max(main_numbers_display) - min(main_numbers_display),
                "Even Numbers": sum(1 for n in main_numbers_display if n % 2 == 0),
                "Odd Numbers": sum(1 for n in main_numbers_display if n % 2 != 0),
                "Low Numbers": sum(1 for n in main_numbers_display if n <= config['range'][1] / 2),
                "High Numbers": sum(1 for n in main_numbers_display if n > config['range'][1] / 2),
                "Consecutive Pairs": sum(1 for i in range(len(main_numbers_display) - 1) if
                                         main_numbers_display[i + 1] - main_numbers_display[i] == 1)
            }

            # Display in columns
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Sum", stats["Sum"])
                st.metric("Mean", f"{stats['Mean']:.1f}")
                st.metric("Median", f"{stats['Median']:.1f}")

            with col2:
                st.metric("Std Dev", f"{stats['Std Dev']:.1f}")
                st.metric("Range", stats["Range"])
                st.metric("Consecutive", stats["Consecutive Pairs"])

            with col3:
                st.metric("Even/Odd", f"{stats['Even Numbers']}/{stats['Odd Numbers']}")
                even_ratio = stats['Even Numbers'] / config['picks']
                st.progress(even_ratio, text=f"Even: {even_ratio:.0%}")

            with col4:
                st.metric("Low/High", f"{stats['Low Numbers']}/{stats['High Numbers']}")
                low_ratio = stats['Low Numbers'] / config['picks']
                st.progress(low_ratio, text=f"Low: {low_ratio:.0%}")

            # Number distribution visualization
            st.write("#### Number Distribution")
            self._show_number_distribution(main_numbers_display, config)

        # Historical performance
        st.write("#### Historical Performance")
        self.show_historical_performance(numbers, lottery_type, df)

    def _show_number_distribution(self, numbers: List[int], config: Dict[str, Any]):
        """Shows visual distribution of the predicted numbers."""
        try:
            # Create a visualization of number positions
            min_num, max_num = config['range']

            # Create bins for grouping numbers
            # Ensure bins are at least 1 and cover the full range
            num_bins = min(10, max(1, (max_num - min_num + 1) // 5))
            bins = np.linspace(min_num, max_num + 1, num_bins + 1)

            # Count numbers in each bin
            hist, bin_edges = np.histogram(numbers, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Create bar chart
            fig = px.bar(
                x=bin_centers,
                y=hist,
                labels={'x': 'Number Range', 'y': 'Count'},
                title='Distribution of Predicted Numbers'
            )

            # Add markers for actual numbers
            if numbers:  # Only add scatter if there are numbers to plot
                fig.add_scatter(
                    x=numbers,
                    y=[0.5] * len(numbers),  # Place markers slightly above x-axis
                    mode='markers',
                    marker=dict(size=10, symbol='diamond', color='red'),
                    name='Predicted Numbers'
                )

            fig.update_layout(
                xaxis_title="Number Range",
                yaxis_title="Count",
                showlegend=True,
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            logging.error(f"Error creating number distribution visualization: {str(e)}", exc_info=True)
            st.error("Could not create distribution visualization")

    def show_historical_performance(self, numbers: List[int], lottery_type: str, df: pd.DataFrame):
        """
        Calculates and displays the historical match distribution for the prediction.

        Args:
            numbers: Predicted numbers
            lottery_type: Type of lottery
            df: Historical data DataFrame
        """
        if not self._validate_dataframe(df):
            st.info("Unable to analyze historical performance due to invalid data.")
            return

        config = self.lottery_configs[lottery_type]

        # Separate main numbers for matching
        # Ensure numbers is not empty and has enough elements for picks
        main_prediction_set = set(numbers[:config['picks']])

        if not main_prediction_set:
            st.info("No main numbers in prediction to check historical performance.")
            return

            # Calculate matches for each historical draw
        matches = []
        match_details = []

        # Ensure df is sorted by date for logical display of best matches
        df_sorted = df.sort_values('Date', ascending=False).reset_index(drop=True)

        for idx, row in df_sorted.iterrows():  # Iterate through sorted DF
            historical_numbers = row['Numbers_List']
            if isinstance(historical_numbers, list) and len(historical_numbers) >= config['picks']:
                historical_main = set(historical_numbers[:config['picks']])
                num_matches = len(main_prediction_set.intersection(historical_main))
                matches.append(num_matches)

                # Store details for notable matches (3 or more)
                if num_matches >= 3:
                    match_details.append({
                        'Date': row['Date'],
                        'Matches': num_matches,
                        'Numbers': historical_numbers[:config['picks']],
                        'Matched': sorted(list(main_prediction_set.intersection(historical_main)))
                        # Ensure matched are sorted
                    })

        if not matches:
            st.info("No historical data available for comparison or no valid draws found in history.")
            return

            # Create match distribution
        match_dist = pd.Series(matches).value_counts().sort_index()

        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Max Matches", max(matches))
        with col2:
            st.metric("Avg Matches", f"{np.mean(matches):.2f}")
        with col3:
            st.metric("Draws Analyzed", len(matches))
        with col4:
            pct_3_or_more = (sum(1 for m in matches if m >= 3) / len(matches) * 100) if matches else 0
            st.metric("≥3 Matches", f"{pct_3_or_more:.1f}%")

            # Create distribution chart
        if not match_dist.empty:
            fig = px.bar(
                x=match_dist.index,
                y=match_dist.values,
                labels={'x': 'Number of Matches', 'y': 'Frequency'},
                title=f'Historical Match Distribution (Total Draws: {len(df)})',
                text=match_dist.values
            )

            # Color bars based on match quality
            colors = [
                'red' if x == 0 else 'orange' if x == 1 else 'yellow' if x == 2 else 'lightgreen' if x == 3 else 'green'
                for x in match_dist.index]
            fig.update_traces(marker_color=colors, textposition='outside')

            fig.update_layout(
                xaxis=dict(tickmode='linear', dtick=1),
                yaxis_title="Number of Occurrences",
                showlegend=False,
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show best historical matches
            if match_details:
                st.write("#### Best Historical Matches")

                # Already sorted by matches and date ascending in match_details when appended from df_sorted (descending date)
                # Sort descending by matches, then descending by date
                match_details_sorted = sorted(match_details, key=lambda x: (-x['Matches'], x['Date']), reverse=False)

                # Display top matches
                for detail in match_details_sorted[:5]:  # Show top 5
                    with st.expander(f"{detail['Date'].strftime('%Y-%m-%d')} - {detail['Matches']} matches"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Historical Numbers:** {', '.join(map(str, detail['Numbers']))}")
                        with col2:
                            st.write(f"**Matched Numbers:** {', '.join(map(str, detail['Matched']))}")

                            # Empirical Match Frequency table
            st.write("#### Empirical Match Frequencies")  # Renamed for clarity

            # Calculate empirical frequencies
            prob_data = []
            for k in range(config['picks'] + 1):  # Iterate from 0 matches up to max picks
                actual_freq = match_dist.get(k, 0) / len(matches) if len(matches) > 0 else 0
                prob_data.append({
                    'Matches': k,
                    'Frequency': f"{actual_freq:.2%}",  # Renamed column to "Frequency"
                    'Count': match_dist.get(k, 0)
                })

            prob_df = pd.DataFrame(prob_data)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
        else:
            st.info("No match distribution data available.")

    def get_model_info(self, lottery_type: str) -> Dict[str, Any]:
        """
        Returns information about the trained models for a lottery type.

        Args:
            lottery_type: Type of lottery

        Returns:
            Dictionary containing model information
        """
        if lottery_type not in self.models:
            return {'trained': False}

        model_info = {
            'trained': True,
            'models': list(self.models[lottery_type]['models'].keys()),
            'scores': self.model_scores.get(lottery_type, {}),
            'features': self._get_feature_names(lottery_type)
        }

        return model_info

    def _get_feature_names(self, lottery_type: str) -> List[str]:
        """
        Returns descriptive names for the features used in the model.

        Args:
            lottery_type: Type of lottery

        Returns:
            List of feature names
        """
        config = self.lottery_configs[lottery_type]
        feature_names = []

        # One-hot encoded features
        for i in range(1, config['range'][1] + 1):
            feature_names.append(f"has_number_{i}")

            # Statistical features
        feature_names.extend([
            "mean", "std_dev", "median", "range",
            "even_ratio", "low_ratio", "tens_distribution",
            "uniqueness_ratio", "variance", "consecutive_ratio"
        ])

        return feature_names


# Utility function for external use
def create_ml_predictor() -> MLPredictor:
    """Factory function to create an MLPredictor instance."""
    return MLPredictor()