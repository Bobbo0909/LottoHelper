# src/prediction_validator.py
import numpy as np
from collections import Counter
import streamlit as st
import plotly.express as px  # Not directly used but often helpful for related visualizations
import pandas as pd
from src.pattern_analyzer import LottoPatternAnalyzer  # Import the updated Pattern Analyzer


class LottoPredictionValidator:
    def __init__(self, df, lottery_config):
        """
        Initializes the validator with a DataFrame and lottery configuration.
        It uses LottoPatternAnalyzer internally to get pattern data.

        Args:
            df (pd.DataFrame): The DataFrame containing lottery data, with a 'Numbers_List' column.
            lottery_config (dict): A dictionary with lottery-specific details (e.g., 'range', 'picks', 'has_bonus').
        """
        self.df = df.copy()  # Work on a copy
        self.lottery_config = lottery_config
        self.analyzer = LottoPatternAnalyzer(self.df, self.lottery_config)
        self.patterns = self.analyzer.patterns  # Directly access the analyzed patterns

        # Ensure patterns were successfully analyzed
        if self.patterns is None:
            st.error("Could not analyze patterns. Prediction validation may be limited.")
            self.optimal_patterns = {}  # Initialize empty to prevent errors
        else:
            self.optimal_patterns = self._calculate_optimal_patterns()

    def _calculate_optimal_patterns(self):
        """
        Calculates optimal (most frequent) patterns from the analyzed data.
        """
        if self.patterns is None:
            return {}

        midpoint = self.lottery_config['range'][1] / 2

        optimal_even_odd = max(self.patterns['distribution_patterns']['even_odd'].items(),
                               key=lambda x: x[1])[0] if self.patterns['distribution_patterns']['even_odd'] else 0

        optimal_high_low = max(self.patterns['distribution_patterns']['high_low'].items(),
                               key=lambda x: x[1])[0] if self.patterns['distribution_patterns']['high_low'] else 0

        return {
            'even_odd': optimal_even_odd,
            'high_low': optimal_high_low,
            'midpoint': midpoint
        }

    def generate_predictions(self, num_predictions=5):
        """
        Generates lottery number predictions based on historical frequencies and patterns.

        Args:
            num_predictions (int): Number of prediction sets to generate.
        Returns:
            list: A list of tuples, each containing a predicted set of numbers and its score.
        """
        predictions = []

        if self.patterns is None or not self.patterns['individual_frequencies']:
            st.warning("Cannot generate predictions: No individual number frequencies available.")
            return []

        frequencies = self.patterns['individual_frequencies']
        all_possible_numbers = list(range(self.lottery_config['range'][0], self.lottery_config['range'][1] + 1))

        # Create a probability distribution for all possible numbers
        # Use .get(n, 0) to handle numbers that might not have appeared historically
        probs_values = np.array([frequencies.get(n, 0) for n in all_possible_numbers], dtype=float)

        # Add a small uniform probability to all numbers to ensure even rare numbers have a chance
        # This prevents division by zero if all historical frequencies are zero (unlikely but robust)
        min_prob = 1e-6  # A very small probability
        probs_values = probs_values + min_prob

        # Normalize probabilities
        if np.sum(probs_values) == 0:  # Fallback if somehow still all zeros
            probs = np.full(len(all_possible_numbers), 1.0 / len(all_possible_numbers))
        else:
            probs = probs_values / np.sum(probs_values)

        for _ in range(num_predictions * 3):  # Generate more than needed to pick the best
            try:
                combination_main = sorted(np.random.choice(
                    all_possible_numbers,
                    size=self.lottery_config['picks'],
                    replace=False,
                    p=probs
                ).tolist())

                combination = list(combination_main)  # Convert to list to allow appending bonus

                if self.lottery_config['has_bonus']:
                    bonus_number = None
                    if 'bonus_range' in self.lottery_config and self.lottery_config['bonus_range']:
                        bonus_numbers_range = list(range(
                            self.lottery_config['bonus_range'][0],
                            self.lottery_config['bonus_range'][1] + 1
                        ))
                        bonus_number = np.random.choice(bonus_numbers_range)
                    else:
                        # Pick bonus from remaining numbers not in main combination
                        remaining_numbers = list(set(all_possible_numbers) - set(combination_main))
                        if remaining_numbers:
                            bonus_number = np.random.choice(remaining_numbers)
                        else:  # Fallback if all numbers are picked (e.g., small range lottery)
                            bonus_number = np.random.choice(all_possible_numbers)

                    if bonus_number is not None:
                        combination.append(bonus_number)

                score = self._calculate_score(combination)
                predictions.append((combination, score))
            except ValueError as ve:
                st.error(f"Error generating combination: {ve}. Check lottery configuration (e.g., picks vs range).")
                continue
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction generation: {e}")
                continue

        # Sort by score and return top N unique predictions
        unique_predictions = []
        seen_combinations = set()
        for pred, score in sorted(predictions, key=lambda x: x[1], reverse=True):
            pred_tuple = tuple(pred)
            if pred_tuple not in seen_combinations:
                unique_predictions.append((pred, score))
                seen_combinations.add(pred_tuple)
            if len(unique_predictions) >= num_predictions:
                break

        return unique_predictions

    def _calculate_score(self, combination):
        """
        Calculates a score for a given combination based on frequency and pattern alignment.
        """
        if self.patterns is None:
            return 0.0

        # Separate main numbers and bonus if applicable
        main_numbers = combination
        if self.lottery_config['has_bonus'] and len(combination) > self.lottery_config['picks']:
            main_numbers = combination[:-1]

        # 1. Frequency-based score
        # Use .get(n, 0) to safely handle numbers that might not exist in frequencies (shouldn't happen with proper generation)
        freq_values = [self.patterns['individual_frequencies'].get(n, 0) for n in main_numbers]

        # Normalize frequency score by max frequency observed for any number
        max_overall_freq = max(self.patterns['individual_frequencies'].values()) if self.patterns[
            'individual_frequencies'] else 1
        freq_score = np.mean(freq_values) / max_overall_freq if max_overall_freq > 0 else 0

        # 2. Pattern matching score (Even/Odd)
        even_count = sum(1 for n in main_numbers if n % 2 == 0)
        # Scale difference by total picks to get a score between 0 and 1
        even_score = 1 - (abs(even_count - self.optimal_patterns.get('even_odd', even_count)) / self.lottery_config[
            'picks']) if self.lottery_config['picks'] > 0 else 0

        # 3. Pattern matching score (High/Low)
        low_count = sum(1 for n in main_numbers if n <= self.optimal_patterns.get('midpoint', 0))
        # Scale difference by total picks to get a score between 0 and 1
        low_score = 1 - (
                    abs(low_count - self.optimal_patterns.get('high_low', low_count)) / self.lottery_config['picks']) if \
        self.lottery_config['picks'] > 0 else 0

        # Weighted average of scores (weights can be adjusted)
        # Using a simple average for now as requested by the "lite" nature
        final_score = (freq_score + even_score + low_score) / 3

        return final_score

    def validate_against_history(self, predictions_with_scores, test_size=50):
        """
        Validates generated predictions against recent historical draws.

        Args:
            predictions_with_scores (list): A list of (combination, score) tuples.
            test_size (int): Number of recent draws to validate against.
        Returns:
            list: A list of dictionaries containing validation results for each prediction.
        """
        if self.df.empty:
            st.warning("No historical data available for validation.")
            return []

        recent_draws_df = self.df.tail(test_size)
        if recent_draws_df.empty:
            st.warning(f"Less than {test_size} draws available for validation.")
            recent_draws_df = self.df  # Use all available if not enough

        recent_draws_sets = recent_draws_df['Numbers_List'].apply(set).tolist()

        results = []
        for combination, score in predictions_with_scores:
            combo_set = set(
                combination[:-1] if self.lottery_config['has_bonus'] and len(combination) > self.lottery_config[
                    'picks'] else combination)
            matches = [len(combo_set.intersection(draw_set)) for draw_set in recent_draws_sets]

            results.append({
                'combination': combination,
                'score': score,
                'max_matches': max(matches) if matches else 0,
                'avg_matches': np.mean(matches) if matches else 0.0,
                'match_distribution': Counter(matches)
            })
        return results

    def explain_prediction(self, combination, score):
        """
        Generates a textual explanation for a given prediction.
        """
        if self.patterns is None:
            return ["No pattern data available for explanation."]

        explanation_lines = []

        main_numbers = combination
        if self.lottery_config['has_bonus'] and len(combination) > self.lottery_config['picks']:
            main_numbers = combination[:-1]

        # Explanation based on score
        explanation_lines.append(f"Overall Score: {score:.2%}")

        # Frequency analysis
        explanation_lines.append("\n**Number Frequencies:**")
        for num in main_numbers:
            freq = self.patterns['individual_frequencies'].get(num, 0)
            total_draws = len(self.df)
            explanation_lines.append(
                f"- Number {num}: Appeared {freq} times ({freq / total_draws:.1%} of draws)"
            )

        # Pattern analysis
        even_count = sum(1 for n in main_numbers if n % 2 == 0)
        odd_count = len(main_numbers) - even_count
        explanation_lines.append(f"\n**Even/Odd Distribution:** {even_count} Even, {odd_count} Odd")
        explanation_lines.append(f"  (Optimal Even Count: {self.optimal_patterns.get('even_odd', 'N/A')})")

        low_count = sum(1 for n in main_numbers if n <= self.optimal_patterns.get('midpoint', 0))
        high_count = len(main_numbers) - low_count
        explanation_lines.append(f"**Low/High Distribution:** {low_count} Low, {high_count} High")
        explanation_lines.append(f"  (Optimal Low Count: {self.optimal_patterns.get('high_low', 'N/A')})")

        explanation_lines.append(f"\n**Sum of Numbers:** {sum(main_numbers)}")
        explanation_lines.append(f"**Number Spread (Max - Min):** {max(main_numbers) - min(main_numbers)}")

        return explanation_lines

    def show_validation_results(self, predictions, validation_results):
        """
        Displays validation results for the generated predictions using Streamlit.
        """
        st.write("### Validation Results")

        if not predictions or not validation_results:
            st.info("No predictions or validation results to display.")
            return

        for pred_item, val_item in zip(predictions, validation_results):
            combination_display = pred_item[0]
            score_display = pred_item[1]

            # Format numbers for display, similar to ML predictor
            if self.lottery_config['has_bonus'] and len(combination_display) > self.lottery_config['picks']:
                main_numbers_str = ", ".join(map(str, combination_display[:-1]))
                bonus_number_str = str(combination_display[-1])
                formatted_numbers = f"{main_numbers_str} | Bonus: {bonus_number_str}"
            else:
                formatted_numbers = ", ".join(map(str, combination_display))

            with st.expander(f"Prediction: {formatted_numbers} (Score: {score_display:.2%})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Overall Score:** {score_display:.2%}")
                    st.markdown(f"**Max Historical Matches:** {val_item['max_matches']}")
                    st.markdown(f"**Average Historical Matches:** {val_item['avg_matches']:.2f}")

                    st.markdown("\n**Detailed Explanation:**")
                    explanation_lines = self.explain_prediction(pred_item[0], pred_item[1])
                    for line in explanation_lines:
                        st.write(line)

                with col2:
                    st.markdown("**Historical Match Distribution:**")
                    dist_df = pd.DataFrame([
                        {"Matches": k, "Count": v}
                        for k, v in val_item['match_distribution'].items()
                    ]).sort_values("Matches").set_index("Matches")

                    if not dist_df.empty:
                        fig = px.bar(dist_df, y='Count', title="Match Distribution")
                        st.plotly_chart(fig)
                    else:
                        st.info("No match distribution data available.")