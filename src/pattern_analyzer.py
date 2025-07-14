# src/pattern_analyzer.py
import pandas as pd
import numpy as np
from collections import Counter
import streamlit as st
import plotly.express as px
from datetime import datetime


class LottoPatternAnalyzer:
    def __init__(self, df, lottery_config):
        """
        Initializes the analyzer with a pre-loaded DataFrame and lottery configuration.

        Args:
            df (pd.DataFrame): The DataFrame containing lottery data, with a 'Numbers_List' column.
            lottery_config (dict): A dictionary with lottery-specific details (e.g., 'range').
        """
        # Ensure the DataFrame is a copy to avoid modifying the original data in session state
        self.df = df.copy()
        self.lottery_config = lottery_config
        self.all_numbers = list(range(self.lottery_config['range'][0], self.lottery_config['range'][1] + 1))
        self.patterns = self._analyze_patterns()

    def _analyze_patterns(self):
        """
        Calculates all specified patterns from the internal DataFrame.
        """
        if self.df.empty:
            return None

        patterns = {
            'individual_frequencies': self._analyze_number_frequencies(),
            'pair_patterns': self._analyze_pair_patterns(),
            'distribution_patterns': self._analyze_distribution_patterns()
        }
        return patterns

    def _analyze_number_frequencies(self):
        frequencies = Counter()
        for numbers in self.df['Numbers_List']:
            frequencies.update(numbers)
        return frequencies

    def _analyze_pair_patterns(self):
        pairs = Counter()
        for numbers in self.df['Numbers_List']:
            nums = numbers
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    pairs[(min(nums[i], nums[j]), max(nums[i], nums[j]))] += 1
        return pairs

    def _analyze_distribution_patterns(self):
        patterns = {
            'even_odd': Counter(),
            'high_low': Counter()
        }

        # Determine the midpoint for high/low based on the lottery's range
        high_low_midpoint = self.lottery_config['range'][1] / 2

        for numbers in self.df['Numbers_List']:
            nums = numbers

            # Even/Odd pattern
            even_count = sum(1 for n in nums if n % 2 == 0)
            patterns['even_odd'][even_count] += 1

            # High/Low pattern
            low_count = sum(1 for n in nums if n <= high_low_midpoint)
            patterns['high_low'][low_count] += 1

        return patterns

    def update_date_range(self, start_date, end_date):
        """
        Creates a new LottoPatternAnalyzer instance with data filtered by date.

        Args:
            start_date (datetime.date): The start date for filtering.
            end_date (datetime.date): The end date for filtering.

        Returns:
            LottoPatternAnalyzer: A new instance with the filtered data.
        """
        # Ensure 'Date' column is in a comparable format
        df_to_filter = self.df.copy()
        df_to_filter['Date'] = pd.to_datetime(df_to_filter['Date']).dt.date

        filtered_df = df_to_filter[
            (df_to_filter['Date'] >= start_date) &
            (df_to_filter['Date'] <= end_date)
            ].copy()

        # Return a new instance with the filtered DataFrame
        return LottoPatternAnalyzer(filtered_df, self.lottery_config)

    def show_pattern_analysis(self):
        if self.patterns is None:
            st.warning("No data available to show pattern analysis for the selected date range.")
            return

        st.write("### Number Frequencies")
        freq_df = pd.DataFrame(
            [(num, self.patterns['individual_frequencies'].get(num, 0))
             for num in self.all_numbers],
            columns=['Number', 'Frequency']
        )
        fig = px.bar(freq_df, x='Number', y='Frequency', title="Number Frequencies")
        st.plotly_chart(fig)

        st.write("### Distribution Patterns")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Even/Odd Distribution")
            even_odd_df = pd.DataFrame(
                self.patterns['distribution_patterns']['even_odd'].items(),
                columns=['Even Numbers', 'Count']
            )
            fig_even_odd = px.bar(even_odd_df, x='Even Numbers', y='Count', title="Even/Odd Distribution")
            st.plotly_chart(fig_even_odd)

        with col2:
            st.write("High/Low Distribution")
            high_low_df = pd.DataFrame(
                self.patterns['distribution_patterns']['high_low'].items(),
                columns=['Low Numbers', 'Count']
            )
            fig_high_low = px.bar(high_low_df, x='Low Numbers', y='Count', title="High/Low Distribution")
            st.plotly_chart(fig_high_low)