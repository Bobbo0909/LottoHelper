# src/scraper.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import logging
from datetime import datetime

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')  # Raw scraped data
CLEANED_DIR = os.path.join(DATA_DIR, 'cleaned')  # Cleaned data (will be used by Streamlit)

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, 'scraper.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

LOTTERY_CONFIGS = {
    'daily_lotto': {
        'url_base': "https://www.lottery.co.za/daily-lotto/results/",
        'start_year': 2019,
        'num_numbers': 5,
        'has_bonus': False,
        'ball_class': 'daily-lotto-ball'
    },
    'lotto': {
        'url_base': "https://www.lottery.co.za/lotto/results/",
        'start_year': 2000,
        'num_numbers': 6,
        'has_bonus': True,
        'ball_class': 'lotto-ball',
        'bonus_ball_class': 'lotto-bonus-ball'
    },
    'powerball': {
        'url_base': "https://www.lottery.co.za/powerball/results/",
        'start_year': 2009,
        'num_numbers': 5,
        'has_bonus': True,
        'ball_class': 'powerball-ball',
        'bonus_ball_class': 'powerball-powerball'
    }
}

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/91.0.4472.124 Safari/537.36'
    )
}


# --- Helper Functions (remain the same) ---

def find_ball_numbers(cell, ball_class):
    balls = cell.find_all('div', class_=ball_class)
    extracted_numbers = []
    for ball in balls:
        number = ball.text.strip()
        if number.isdigit():
            extracted_numbers.append(number)
    return extracted_numbers


def find_bonus_numbers(cell, bonus_ball_class):
    bonus_balls = cell.find_all('div', class_=bonus_ball_class)
    extracted_bonus = []
    for bonus in bonus_balls:
        number = bonus.text.strip()
        if number.isdigit():
            extracted_bonus.append(number)
    return extracted_bonus


def extract_draw_number(cells):
    for cell in cells:
        # Check if cell contains any ball divs, if not, it might be the draw number
        if not cell.find_all('div', class_=['daily-lotto-ball', 'lotto-ball', 'powerball-ball', 'lotto-bonus-ball',
                                            'powerball-powerball']):
            num = cell.text.strip()
            if num.isdigit():
                return num  # Return as string, will be converted to int later
    return None


def parse_date_string(date_str):
    try:
        return datetime.strptime(date_str.strip(), "%A, %d %B %Y")
    except ValueError:
        logging.error(f"Could not parse date string: '{date_str}'")
        return None


# --- Main Scraper Functions ---

def get_last_draw_date_from_raw(lottery_type):
    raw_file_path = os.path.join(RESULTS_DIR, f'{lottery_type}_results_all_years.csv')
    if not os.path.exists(raw_file_path):
        logging.info(f"Raw data file for {lottery_type} not found at {raw_file_path}. Assuming clean start.")
        return datetime.min
    try:
        # Explicitly set dtype for 'Draw' to avoid inference issues upon initial load
        df = pd.read_csv(raw_file_path, parse_dates=['Date'], dtype={'Draw': str})
        if df.empty:
            logging.info(f"Raw data file for {lottery_type} is empty. Assuming clean start.")
            return datetime.min

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Convert 'Draw' to numeric then nullable integer, handle errors
        df['Draw'] = pd.to_numeric(df['Draw'], errors='coerce').astype('Int64')
        df = df.dropna(subset=['Date', 'Draw'])  # Drop rows if date or draw are invalid

        if df.empty:
            logging.warning(f"No valid dates or draws found in raw data file for {lottery_type}. Assuming clean start.")
            return datetime.min
        latest_date = df['Date'].max()
        logging.info(f"Last draw date for {lottery_type} in raw data: {latest_date.strftime('%Y-%m-%d')}")
        return latest_date
    except Exception as e:
        logging.error(f"Error reading raw data for {lottery_type} to get latest date: {e}")
        return datetime.min


def scrape_lottery_year(lottery_type, year, last_known_date=datetime.min):
    config = LOTTERY_CONFIGS[lottery_type]
    url = f"{config['url_base']}{year}"
    new_draws = []

    try:
        logging.info(f"Fetching URL: {url} for {lottery_type} (Year: {year})")
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        rows = soup.find_all('tr')

        if "No Results Found" in soup.get_text():
            logging.warning(f"No results found message detected for {lottery_type} in {year}. Skipping.")
            return pd.DataFrame()

        for row in rows:
            try:
                cells = row.find_all('td')
                if not cells or len(cells) < 3:
                    continue

                date_cell_str = cells[0].text.strip()
                draw_date = parse_date_string(date_cell_str)

                if draw_date is None:
                    continue

                # Optimization: If we encounter a draw date older than or equal to our last known date
                # AND it's not the current year (where new draws are still being added),
                # we can stop scraping this year's page.
                current_year_actual = datetime.now().year  # Use a local variable to avoid re-evaluating
                if draw_date <= last_known_date and year < current_year_actual:
                    logging.info(
                        f"Encountered existing draw date {draw_date.strftime('%Y-%m-%d')} for {lottery_type} {year}. Stopping scraping for this year.")
                    break  # Stop processing rows for this year

                # Only process if draw_date is newer than last_known_date OR it's the current year (always scrape current year fully)
                if draw_date > last_known_date or year == current_year_actual:
                    draw_number_str = extract_draw_number(cells)
                    if not draw_number_str:
                        logging.warning(
                            f"Could not extract draw number for date {date_cell_str} in {lottery_type} {year}")
                        continue

                    # Convert draw number to int here, as it's a unique identifier
                    try:
                        draw_number = int(draw_number_str)
                    except ValueError:
                        logging.warning(f"Invalid draw number '{draw_number_str}' for date {date_cell_str}.")
                        continue

                    main_ball_numbers = []
                    for cell in cells:
                        main_ball_numbers.extend(find_ball_numbers(cell, config['ball_class']))

                    bonus_ball_numbers = []
                    if config['has_bonus']:
                        for cell in cells:
                            bonus_ball_numbers.extend(find_bonus_numbers(cell, config['bonus_ball_class']))

                    if len(main_ball_numbers) < config['num_numbers']:
                        logging.warning(
                            f"Not enough main numbers for {date_cell_str} ({lottery_type} {year}): found {len(main_ball_numbers)}, need {config['num_numbers']}")
                        continue

                    if config['has_bonus'] and len(bonus_ball_numbers) < 1:
                        logging.warning(f"No bonus numbers found for {date_cell_str} ({lottery_type} {year}).")
                        continue

                    new_draw = {
                        'Date': draw_date.strftime("%Y-%m-%d"),  # Keep as string initially
                        'Draw': draw_number,  # Store as int
                        'Numbers': str(main_ball_numbers[:config['num_numbers']])
                    }
                    if config['has_bonus']:
                        new_draw['Bonus'] = bonus_ball_numbers[0]

                    new_draws.append(new_draw)

            except Exception as e:
                logging.error(f"Error processing row for {lottery_type} {year}: {str(e)} - Row: {row.text[:100]}...")
                continue

        if new_draws:
            df = pd.DataFrame(new_draws)
            # Convert 'Date' and 'Draw' to their final types immediately after DataFrame creation
            df['Date'] = pd.to_datetime(df['Date'])
            df['Draw'] = df['Draw'].astype('Int64')  # Use nullable integer type
            df = df.sort_values('Date', ascending=True).reset_index(drop=True)
            logging.info(f"Scraped {len(df)} new/recent draws for {lottery_type} {year}")
            return df

        logging.info(
            f"No new data found for {lottery_type} {year} beyond {last_known_date.strftime('%Y-%m-%d') if last_known_date != datetime.min else 'N/A'}")
        return pd.DataFrame()

    except requests.exceptions.Timeout:
        logging.error(f"Timeout occurred while scraping {lottery_type} for year {year}. URL: {url}")
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err} while scraping {lottery_type} for year {year}. URL: {url}")
    except Exception as e:
        logging.error(f"Error scraping {lottery_type} for year {year}: {str(e)}. URL: {url}")
    return pd.DataFrame()


def update_lottery_data(lottery_type):
    config = LOTTERY_CONFIGS[lottery_type]
    raw_file_path = os.path.join(RESULTS_DIR, f'{lottery_type}_results_all_years.csv')

    last_known_date = get_last_draw_date_from_raw(lottery_type)

    start_year_to_scrape = config['start_year']
    if last_known_date != datetime.min:
        # Start scraping from the year of the last known date, to ensure we get any missing draws
        # from that year or newer ones.
        start_year_to_scrape = last_known_date.year

    logging.info(
        f"Updating {lottery_type.upper()} data. Starting from year {start_year_to_scrape} with last known date {last_known_date.strftime('%Y-%m-%d') if last_known_date != datetime.min else 'N/A'}")

    all_new_draws_for_lottery = []
    current_year_actual = datetime.now().year  # Define once

    for year in range(start_year_to_scrape, current_year_actual + 1):
        if year > current_year_actual + 1:  # Don't scrape future years + 1
            logging.info(f"Skipping year {year} for {lottery_type} as it's too far in the future.")
            continue

        new_draws_df = scrape_lottery_year(lottery_type, year, last_known_date)
        if not new_draws_df.empty:
            all_new_draws_for_lottery.append(new_draws_df)

        # This optimization is still useful: if a past year yields no new data, assume we are caught up
        if year < current_year_actual and new_draws_df.empty and last_known_date != datetime.min:
            logging.info(
                f"No new draws found for {lottery_type} in past year {year}. Assuming data is up-to-date for previous years.")
            break  # Stop iterating through older years

        time.sleep(2)  # Be polite to the server

    if all_new_draws_for_lottery:
        combined_new_df = pd.concat(all_new_draws_for_lottery, ignore_index=True)
        # Ensure 'Date' and 'Draw' are proper types after concat, before further processing
        combined_new_df['Date'] = pd.to_datetime(combined_new_df['Date'], errors='coerce')
        combined_new_df['Draw'] = pd.to_numeric(combined_new_df['Draw'], errors='coerce').astype('Int64')
        combined_new_df = combined_new_df.dropna(subset=['Date', 'Draw'])

        existing_df = pd.DataFrame()
        if os.path.exists(raw_file_path):
            try:
                # Explicitly set dtype for 'Draw' to avoid inference issues upon initial load
                existing_df = pd.read_csv(raw_file_path, parse_dates=['Date'], dtype={'Draw': str})
                # Ensure 'Date' and 'Draw' are proper types
                existing_df['Date'] = pd.to_datetime(existing_df['Date'], errors='coerce')
                existing_df['Draw'] = pd.to_numeric(existing_df['Draw'], errors='coerce').astype('Int64')
                existing_df = existing_df.dropna(subset=['Date', 'Draw'])  # Drop rows where critical fields are invalid
            except Exception as e:
                logging.error(f"Error loading existing raw data for {lottery_type}: {e}. Starting fresh for combine.")
                existing_df = pd.DataFrame()  # Treat as if no existing data if error occurs

        # Combine, sort, and remove duplicates
        combined_df = pd.concat([existing_df, combined_new_df], ignore_index=True)

        # Sort by Date (ascending) and Draw (ascending) before dropping duplicates
        # This ensures 'keep='last'' correctly retains the newest/most complete entry
        combined_df = combined_df.sort_values(by=['Date', 'Draw'], ascending=[True, True])

        # Drop duplicates based on the unique combination of Date and Draw, keeping the last (most recent)
        initial_rows = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['Date', 'Draw'], keep='last')
        if len(combined_df) < initial_rows:
            logging.info(f"Removed {initial_rows - len(combined_df)} duplicate rows for {lottery_type}.")

        logging.info(
            f"Combined {lottery_type} data. New draws added: {len(combined_new_df)}. Total unique draws: {len(combined_df)}")

        # Sort again by date descending for typical display, then save
        combined_df = combined_df.sort_values('Date', ascending=False).reset_index(drop=True)

        # Convert Date back to string for CSV saving, and Draw back to string to avoid mixed types on next read
        combined_df['Date'] = combined_df['Date'].dt.strftime("%Y-%m-%d")
        combined_df['Draw'] = combined_df['Draw'].astype(str)  # Convert back to string for CSV output

        combined_df.to_csv(raw_file_path, index=False)
        logging.info(f"Updated raw {lottery_type} data saved to: {raw_file_path}")
        return True
    else:
        logging.info(f"No new draws found for {lottery_type}. Data is already up-to-date.")
        return False


# This is the main function to be called from external scripts (like app.py)
def scrape_all_lotteries_incremental_callable():
    """
    Orchestrates incremental scraping for all configured lottery types.
    Designed to be called from other modules.
    Returns True if any new data was scraped and saved for at least one lottery type.
    """
    logging.info("Starting incremental lottery data scraping process from callable function...")
    any_new_data_scraped = False
    for lottery_type in LOTTERY_CONFIGS.keys():
        logging.info(f"Processing {lottery_type.upper()}...")
        if update_lottery_data(lottery_type):
            any_new_data_scraped = True
        time.sleep(1)  # Small delay between lottery types
    logging.info("Incremental scraping completed by callable function.")
    return any_new_data_scraped


# Only run if this script is executed directly (e.g., from terminal)
if __name__ == "__main__":
    print("Ensuring data and log directories exist...")
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CLEANED_DIR, exist_ok=True)
    print(f"Directories checked/created: {LOGS_DIR}, {RESULTS_DIR}, {CLEANED_DIR}")

    # Call the callable function when run directly
    scrape_all_lotteries_incremental_callable()
    print("\nScraping process finished. Check logs/scraper.log for details.")