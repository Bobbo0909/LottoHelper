# src/cleaner.py
import pandas as pd
import os
import logging
import ast  # For safe evaluation of string representations of lists

# --- Configuration ---
# Determine the project root to correctly locate data and log directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')  # Raw scraped data
CLEANED_DIR = os.path.join(DATA_DIR, 'cleaned')  # Cleaned data (will be used by Streamlit)

# Ensure necessary directories exist - This is where the "create if not exists" happens
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)  # Ensure results dir exists, as cleaner reads from it

# Configure Logging (this happens once when the module is loaded)
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, 'cleaner.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

LOTTERY_CONFIGS = {
    'daily_lotto': {
        'has_bonus': False,
    },
    'lotto': {
        'has_bonus': True,
    },
    'powerball': {
        'has_bonus': True,
    }
}


# --- Helper Functions (No changes here, these remain the same) ---

def clean_numbers_from_string(numbers_str):
    if pd.isna(numbers_str) or not isinstance(numbers_str, str):
        return None
    try:
        # Use ast.literal_eval to safely convert string representation of list to actual list
        numbers_list = ast.literal_eval(numbers_str)
        # Ensure each element is an integer and join back to a comma-separated string
        return ','.join(str(int(n.strip())) for n in numbers_list if str(n).strip().isdigit())
    except (ValueError, SyntaxError) as e:
        logging.warning(f"Could not parse numbers string '{numbers_str}': {e}")
        return None


# --- Main Cleaner Functions ---

def clean_lottery_data(lottery_type):
    config = LOTTERY_CONFIGS[lottery_type]
    raw_input_path = os.path.join(RESULTS_DIR, f'{lottery_type}_results_all_years.csv')
    cleaned_output_path_csv = os.path.join(CLEANED_DIR, f'{lottery_type}_results_all_years.csv')
    cleaned_output_path_parquet = os.path.join(CLEANED_DIR, f'{lottery_type}_results_all_years.parquet')

    if not os.path.exists(raw_input_path):
        logging.warning(f"Raw data file not found for {lottery_type} at {raw_input_path}. Skipping cleaning.")
        return False  # Indicate no cleaning was done

    try:
        logging.info(f"Loading raw data for {lottery_type} from {raw_input_path}...")
        df = pd.read_csv(raw_input_path, parse_dates=['Date'])

        # Drop rows where 'Date' could not be parsed or is missing
        df = df.dropna(subset=['Date'])
        if df.empty:
            logging.warning(
                f"No valid dates found in raw data for {lottery_type} after initial load. Skipping cleaning.")
            return False

        # Apply cleaning to 'Numbers' column
        df['Numbers'] = df['Numbers'].apply(clean_numbers_from_string)
        df = df.dropna(subset=['Numbers'])  # Drop rows where numbers couldn't be cleaned

        # Process 'Bonus' column if applicable
        if config['has_bonus'] and 'Bonus' in df.columns:
            df['Bonus'] = df['Bonus'].apply(
                lambda x: int(str(x).strip()) if pd.notnull(x) and str(x).strip().isdigit() else None
            )
            df = df.dropna(subset=['Bonus'])  # Drop rows if bonus is missing after parsing
            df['Bonus'] = df['Bonus'].astype('Int64')  # Use nullable integer type

        # Process 'Draw' column
        if 'Draw' in df.columns:
            df['Draw'] = pd.to_numeric(df['Draw'], errors='coerce').astype('Int64')
            df = df.dropna(subset=['Draw'])  # Drop rows if draw number is invalid

        # Sort the DataFrame by Date in descending order (most recent first)
        df = df.sort_values('Date', ascending=False).reset_index(drop=True)

        # Save cleaned data to CSV
        df.to_csv(cleaned_output_path_csv, index=False)
        logging.info(f"Cleaned {lottery_type} data saved to CSV: {cleaned_output_path_csv}")

        # Save cleaned data to Parquet (more efficient for later reading)
        df.to_parquet(cleaned_output_path_parquet, index=False)
        logging.info(f"Cleaned {lottery_type} data saved to Parquet: {cleaned_output_path_parquet}")
        return True  # Indicate cleaning was successful

    except Exception as e:
        logging.error(f"Error cleaning {lottery_type} data from {raw_input_path}: {str(e)}")
        return False  # Indicate cleaning failed


# This is the main function to be called from external scripts (like app.py)
def clean_all_lottery_data_callable():
    """
    Orchestrates the cleaning process for all lottery types.
    Designed to be called from other modules.
    Returns True if data for at least one lottery type was successfully cleaned.
    """
    logging.info("Starting comprehensive lottery data cleaning process from callable function...")
    any_data_cleaned = False
    for lottery_type in LOTTERY_CONFIGS.keys():
        logging.info(f"Cleaning data for {lottery_type.upper()}...")
        if clean_lottery_data(lottery_type):
            any_data_cleaned = True
    logging.info("Data cleaning completed by callable function.")
    return any_data_cleaned


# Only run if this script is executed directly (e.g., from terminal)
if __name__ == "__main__":
    print("Ensuring data and log directories exist...")
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CLEANED_DIR, exist_ok=True)
    print(f"Directories checked/created: {LOGS_DIR}, {RESULTS_DIR}, {CLEANED_DIR}")

    # Call the callable function when run directly
    clean_all_lottery_data_callable()
    print("\nCleaning process finished. Check logs/cleaner.log for details.")