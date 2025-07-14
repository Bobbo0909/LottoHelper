# src/data_loader.py
import pandas as pd
import os
import logging
import ast # For safe evaluation of string representations of lists
import streamlit as st # <--- ADD THIS LINE

# --- Configuration & Logging (Existing) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CLEANED_DIR = os.path.join(DATA_DIR, 'cleaned')

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, 'data_loader.log'), # Specific log for data loader
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Lottery configurations - used by app, scraper, cleaner, and data_loader
LOTTERY_CONFIGS = {
    'Daily Lotto': { # Display name
        'cleaned_file': 'daily_lotto_results_all_years.parquet', # File name
        'range': (1, 36),
        'picks': 5,
        'has_bonus': False
    },
    'Lotto': {
        'cleaned_file': 'lotto_results_all_years.parquet',
        'range': (1, 52),
        'picks': 6,
        'has_bonus': True,
        'bonus_range': (1, 52) # Lotto bonus is within main range
    },
    'Powerball': {
        'cleaned_file': 'powerball_results_all_years.parquet',
        'range': (1, 50),
        'picks': 5,
        'has_bonus': True,
        'bonus_range': (1, 20) # Powerball bonus has a separate range
    }
}

def get_lottery_config(lottery_display_name):
    """Retrieves configuration for a given lottery display name."""
    return LOTTERY_CONFIGS.get(lottery_display_name)

# --- NEW/UPDATED Data Loading Function ---
@st.cache_data(show_spinner=False)
def load_cleaned_data_for_ml(lottery_type_display_name, file_name):
    """
    Loads the cleaned lottery data for a given lottery type from a Parquet file,
    performs necessary preprocessing (like creating Numbers_List and parsing Bonus).
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
        return pd.DataFrame() # Return empty DataFrame if file not found
    try:
        df = pd.read_parquet(file_path)

        # Ensure 'Date' column is datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']) # Drop rows where Date could not be parsed

        # --- ROBUST NUMBER PARSING (Moved from ml_predictor) ---
        def parse_numbers_string_robust(x):
            """Safely parses various formats of numbers data into a sorted list of integers."""
            if pd.isna(x):
                return []
            try:
                # Try ast.literal_eval first (for string representations of lists)
                evaluated = ast.literal_eval(str(x))
                if isinstance(evaluated, (list, tuple)):
                    return sorted([int(str(n).strip()) for n in evaluated if str(n).strip().isdigit()])
                elif isinstance(evaluated, (int, float)): # Handle if it's just a single number
                    return sorted([int(evaluated)])
                else:
                    logging.warning(
                        f"Unexpected type after ast.literal_eval for Numbers column: {type(evaluated)} for value: '{x}' in {lottery_type_display_name}")
                    return []
            except (ValueError, SyntaxError): # Fallback for comma-separated strings (e.g., "1,5,12")
                if isinstance(x, str):
                    return sorted([int(n.strip()) for n in x.split(',') if n.strip().isdigit()])
                return []
            except Exception as parse_err:
                logging.error(f"Failed to parse Numbers string/value '{x}' for {lottery_type_display_name}: {parse_err}", exc_info=True)
                return []

        df['Numbers_List'] = df['Numbers'].apply(parse_numbers_string_robust)
        # Drop rows where 'Numbers_List' is empty after parsing (means no valid main numbers)
        df = df[df['Numbers_List'].apply(lambda x: len(x) > 0)].copy()
        # --- END ROBUST PARSING ---

        # Process 'Bonus' column if it exists and is needed
        if 'Bonus' in df.columns:
            df['Bonus'] = pd.to_numeric(df['Bonus'], errors='coerce').astype('Int64') # Convert to nullable integer
            df = df.dropna(subset=['Bonus']) # Drop rows if bonus is missing after parsing

        # Ensure 'Draw' column is also consistently typed if it exists
        if 'Draw' in df.columns:
            df['Draw'] = pd.to_numeric(df['Draw'], errors='coerce').astype('Int64')
            df = df.dropna(subset=['Draw'])


        logging.info(f"Successfully loaded and preprocessed {len(df)} rows for {lottery_type_display_name} from {file_path}")
        return df
    except Exception as err:
        logging.error(f"Critical error loading or processing cleaned data for {lottery_type_display_name} from {file_path}: {err}", exc_info=True)
        return pd.DataFrame()

# No main block here, as this is a module for functions