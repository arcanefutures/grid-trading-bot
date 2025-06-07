# coinbase_websocket_feed.py

import websocket
import json
import logging
import os
from dotenv import load_dotenv
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
import pandas as pd
import signal
import threading
import time
from collections import deque

from core.strategy import GridStrategy
from config.parameters import PRODUCT_IDS, GRID_STRATEGY_PARAMS, INITIAL_BACKTEST_CAPITAL, COMMISSION_RATE

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
load_dotenv()
MARKET_DATA_WS_URL = "wss://advanced-trade-ws.coinbase.com"

# --- Global State ---
strategy_instances = {}
paper_portfolio = None
ws_app = None
shutdown_requested = False
paper_trade_log = []
crypto_cost_basis = {}

# --- New Architecture State ---
historical_candles = {product_id: [] for product_id in PRODUCT_IDS}
app_state = "GATHERING_HISTORY" # GATHERING_HISTORY -> PROCESSING_HISTORY -> LIVE_TRADING
order_id_lock = threading.Lock()
next_order_id_counter = 1

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal): return str(obj)
        return super(DecimalEncoder, self).default(obj)

def signal_handler(signum, frame):
    global shutdown_requested, ws_app
    logging.info(f"Signal {signum} received, requesting shutdown.")
    shutdown_requested = True
    if ws_app: ws_app.close()

def get_next_order_id():
    global next_order_id_counter
    with order_id_lock:
        order_id = next_order_id_counter
        next_order_id_counter += 1
        return order_id

def execute_paper_trade(symbol, order_type, price_decimal, amount_decimal, order_id=None):
    # This function remains largely the same, handling the portfolio and logging
    # (For brevity, its detailed logic is omitted, but it should be your full original version)
    global paper_portfolio, paper_trade_log
    base_currency = symbol.split('-')[0].upper()
    commission = (price_decimal * amount_decimal) * Decimal(str(COMMISSION_RATE))
    if order_type == 'buy':
        cost = (price_decimal * amount_decimal) + commission
        if paper_portfolio['cash'] >= cost:
            paper_portfolio['cash'] -= cost
            paper_portfolio['crypto_holdings'][base_currency] += amount_decimal
        else: return # Failed trade
    elif order_type == 'sell':
        revenue = (price_decimal * amount_decimal) - commission
        if paper_portfolio['crypto_holdings'][base_currency] >= amount_decimal:
            paper_portfolio['cash'] += revenue
            paper_portfolio['crypto_holdings'][base_currency] -= amount_decimal
        else: return # Failed trade

    logging.info(f"PAPER EXECUTION: {order_type} {amount_decimal} {symbol} @ {price_decimal}")
    paper_trade_log.append({'timestamp': datetime.now(timezone.utc).isoformat(), 'symbol': symbol, 'type': order_type, 'price': price_decimal, 'amount': amount_decimal})
    strategy_instances[symbol].notify_order_filled({'type': order_type, 'price': price_decimal})
    return True

def process_historical_data():
    """Processes all gathered candles, performs an initial check, and enables live trading."""
    global app_state
    logging.info("Processing historical data...")
    app_state = "PROCESSING_HISTORY"
    
    for product_id, candles in historical_candles.items():
        if not candles:
            logging.warning(f"No historical candles gathered for {product_id}. Skipping.")
            continue

        strategy = strategy_instances[product_id]
        # Sort candles by timestamp to ensure correct order
        sorted_candles = sorted(candles, key=lambda c: c['start'])
        
        # First, populate the entire history
        for candle in sorted_candles:
            start_dt = datetime.fromtimestamp(int(candle['start']), tz=timezone.utc)
            candle_series = pd.Series({'open': Decimal(candle['open']), 'high': Decimal(candle['high']), 'low': Decimal(candle['low']), 'close': Decimal(candle['close']), 'volume': Decimal(candle['volume'])})
            strategy.add_historical_candle(candle_series, start_dt)
        logging.info(f"Processed {len(sorted_candles)} historical candles for {product_id}.")

        # --- NEW LOGIC: Perform initial check after backfill ---
        # Get the very last candle from history to perform the first check
        last_historical_candle = sorted_candles[-1]
        last_start_dt = datetime.fromtimestamp(int(last_historical_candle['start']), tz=timezone.utc)
        last_candle_series = pd.Series({
            'open': Decimal(last_historical_candle['open']), 'high': Decimal(last_historical_candle['high']),
            'low': Decimal(last_historical_candle['low']), 'close': Decimal(last_historical_candle['close']),
            'volume': Decimal(last_historical_candle['volume'])
        })
        
        logging.info(f"[{product_id}] Performing initial state check after processing history...")
        # Manually call the main logic function once to check ADX and place the initial grid if conditions are met
        strategy.on_live_candle(last_candle_series, last_start_dt)

    logging.info("ALL HISTORICAL DATA PROCESSED. SWITCHING TO LIVE TRADING.")
    app_state = "LIVE_TRADING"

def on_open_handler(ws_app):
    logging.info("Connection opened. Subscribing...")
    ws_app.send(json.dumps({"type": "subscribe", "product_ids": PRODUCT_IDS, "channel": "candles"}))
    # Start a timer to process history after a delay
    threading.Timer(20.0, process_historical_data).start()
    logging.info("Gathering historical data for 20 seconds...")

def on_message_handler(ws_app, raw_message_str):
    if shutdown_requested: return
    
    try:
        message = json.loads(raw_message_str)
        if message.get("channel") == "candles":
            for event in message.get("events", []):
                for candle in event.get("candles", []):
                    product_id = candle.get("product_id")
                    if product_id in strategy_instances:
                        if app_state == "GATHERING_HISTORY":
                            historical_candles[product_id].append(candle)
                        elif app_state == "LIVE_TRADING":
                            strategy = strategy_instances[product_id]
                            start_dt = datetime.fromtimestamp(int(candle['start']), tz=timezone.utc)
                            candle_series = pd.Series({'open': Decimal(candle['open']), 'high': Decimal(candle['high']), 'low': Decimal(candle['low']), 'close': Decimal(candle['close']), 'volume': Decimal(candle['volume'])})
                            strategy.on_live_candle(candle_series, start_dt)

    except Exception as e:
        logging.error(f"Error in on_message_handler: {e}", exc_info=True)

def on_error_handler(ws_app, error):
    logging.error(f"WebSocket Error: {error}")

def on_close_handler(ws_app, close_status_code, close_msg):
    logging.info(f"WebSocket Closed: {close_status_code} {close_msg}")

def setup_strategy_and_portfolio():
    global paper_portfolio, strategy_instances
    paper_portfolio = {'cash': Decimal(str(INITIAL_BACKTEST_CAPITAL)), 'crypto_holdings': {}}
    for product_id in PRODUCT_IDS:
        base_currency = product_id.split('-')[0].upper()
        paper_portfolio['crypto_holdings'][base_currency] = Decimal('0')

    for product_id in PRODUCT_IDS:
        logging.info(f"Initializing strategy for {product_id}")
        params = GRID_STRATEGY_PARAMS.copy()
        params["symbol"] = product_id
        strategy = GridStrategy(params)
        strategy.set_backtester_context(
            portfolio=paper_portfolio, 
            execute_order_func=execute_paper_trade,
            get_next_order_id=get_next_order_id
        )
        strategy_instances[product_id] = strategy
    logging.info(f"{len(strategy_instances)} strategy instances created.")

def main_websocket_loop():
    global ws_app
    signal.signal(signal.SIGINT, signal_handler)
    setup_strategy_and_portfolio() 
    # The dashboard saver is removed for now to simplify, can be re-added later.
    ws_app = websocket.WebSocketApp(MARKET_DATA_WS_URL, on_open=on_open_handler, on_message=on_message_handler, on_error=on_error_handler, on_close=on_close_handler)
    logging.info("Connecting to WebSocket...")
    ws_app.run_forever(ping_interval=10, ping_timeout=5)
    logging.info("WebSocket listener finished.")

if __name__ == "__main__":
    main_websocket_loop()