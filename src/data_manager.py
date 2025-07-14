# src/data_manager.py
import logging
import os
from src.scraper import scrape_all_lotteries_incremental_callable
from src.cleaner import clean_all_lottery_data_callable

# Ensure logs directory exists for this module too
# CORRECTED LINE BELOW: Changed PROJECTS_ROOT to PROJECT_ROOT
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure specific logging for data_manager
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, 'data_manager.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Add a console handler so output is visible in the terminal when run directly or by Streamlit
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler) # Add to root logger


def update_all_data():
    """
    Combines the scraping and cleaning process for all lottery types.
    This is the single function that `app.py` will call.
    Returns True if the entire process completed successfully, False otherwise.
    """
    logging.info("Starting combined data update (scrape + clean)...")

    try:
        # Step 1: Scrape new data incrementally
        logging.info("Initiating data scraping...")
        # The scraper function now returns True if *any* new data was scraped
        # This is a useful flag, but we'll still try to clean regardless.
        scraped_result = scrape_all_lotteries_incremental_callable()

        if not scraped_result:
            logging.info(
                "Scraping completed, but no *new* data was found for any lottery type. Proceeding to clean existing data just in case.")

        # Step 2: Clean all lottery data (this re-cleans all data, ensuring consistency)
        logging.info("Initiating data cleaning...")
        cleaned_result = clean_all_lottery_data_callable()

        if cleaned_result:
            logging.info("Combined data update completed successfully!")
            return True
        else:
            logging.warning("Data cleaning did not complete successfully. Check logs for details.")
            return False

    except Exception as e:
        logging.error(f"An unexpected error occurred during combined data update: {e}", exc_info=True)
        return False


# This block allows you to test data_manager.py directly from the terminal
if __name__ == "__main__":
    print("Running data update from data_manager.py...")
    success = update_all_data()
    if success:
        print("\nData update process finished. Check logs/data_manager.log for details.")
    else:
        print("\nData update process failed or completed with warnings. Check logs/data_manager.log for details.")