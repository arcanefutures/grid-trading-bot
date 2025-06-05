# main.py
import os
from backtesting.backtester import Backtester
from config.parameters import GRID_STRATEGY_PARAMS, INITIAL_BACKTEST_CAPITAL, COMMISSION_RATE
from config.settings import KRAKEN_MIN_ORDER_SIZES, LOG_FILE, LOG_LEVEL

# Optional: Set up basic logging (good practice)
import logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL),
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler()
                    ])

def run_single_backtest(symbol):
    """
    Runs a backtest for a single cryptocurrency symbol.
    """
    print(f"\n===== Initiating Backtest for {symbol} =====")
    logging.info(f"Initiating Backtest for {symbol}")

    # Instantiate the Backtester
    # It automatically picks up INITIAL_BACKTEST_CAPITAL, COMMISSION_RATE from parameters.py
    # and KRAKEN_MIN_ORDER_SIZES from settings.py via its __init__ defaults.
    # Removed the complex initial_capital calculation which caused KeyError.
    backtester = Backtester(data_folder='backtesting/data/raw/') 

    # Override to 500.0 USD for the single backtest run (as intended for this test)
    backtester.initial_capital = INITIAL_BACKTEST_CAPITAL 

    # Load data for the chosen symbol
    data_loaded = backtester.load_data(symbol)

    if data_loaded:
        # Run the backtest
        backtester.run_backtest(symbol)
        logging.info(f"Backtest for {symbol} completed.")
    else:
        logging.error(f"Could not load data for {symbol}. Skipping backtest.")

if __name__ == "__main__":
    # Ensure the logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # You can choose which symbol to backtest from your configured list
    symbols_to_backtest = ["LINK_USD"] # Start with one symbol for clarity
    
    # If you want to backtest all symbols iteratively:
    # symbols_to_backtest = GRID_STRATEGY_PARAMS["symbols"] 

    for symbol in symbols_to_backtest:
        run_single_backtest(symbol)

    print("\nAll selected backtests finished.")