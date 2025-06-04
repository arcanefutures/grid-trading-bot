# config/settings.py

# --- GENERAL BOT LOGGING & NOTIFICATION SETTINGS ---
LOG_LEVEL = "INFO" # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "logs/bot.log"
NOTIFY_TELEGRAM = False # Set to True to enable Telegram notifications (requires setup)
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

# --- EXCHANGE-SPECIFIC SETTINGS (for Kraken) ---
# Minimum order sizes for various base assets on Kraken (in base asset units)
# These are crucial for ensuring orders are valid.
# ALWAYS VERIFY THESE ON KRAKEN'S OFFICIAL MINIMUM ORDER SIZE PAGE!
KRAKEN_MIN_ORDER_SIZES = {
    'BTC': 0.0001,
    'ETH': 0.005,
    'SOL': 0.1,
    'ADA': 1.0,
    'DOT': 1.0,
    'AVAX': 0.1,
    'ATOM': 1.0,
    'LINK': 1.0,
    'XRP': 10.0,
    # Add more assets if needed, ensure the 'base' part of 'PAIR_USD' is the key
}

# Add any other global settings here as needed, e.g., database paths, etc.
# DB_PATH = "data/trade_history.db" # Example