import websocket 
import json
import logging
import os
from dotenv import load_dotenv
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone

import pandas as pd
from core.strategy import GridStrategy
from config.parameters import GRID_STRATEGY_PARAMS, INITIAL_BACKTEST_CAPITAL, COMMISSION_RATE
from config.settings import KRAKEN_MIN_ORDER_SIZES as COINBASE_MIN_ORDER_SIZES # TODO: Update

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
load_dotenv()
API_KEY = os.getenv("API_KEY") 
API_SECRET = os.getenv("PRIVATE_KEY")
PRODUCT_ID = "ADA-USD"
MARKET_DATA_WS_URL = "wss://advanced-trade-ws.coinbase.com"

# --- Global Variables ---
grid_strategy_instance = None
paper_portfolio = None # Will be initialized to {'cash': Decimal, 'crypto_holdings': {'BASE': Decimal}}
is_first_candle_processed = False
ws_client = None

# --- Paper Trading Simulation Logic ---
# Define this function ONCE, correctly.
def execute_paper_trade(symbol, order_type, price_decimal, amount_decimal):
    """
    Simulates the execution of an order for paper trading.
    Updates the global paper_portfolio and logs the trade.
    Expected portfolio structure: {'cash': Decimal, 'crypto_holdings': {'BASE_SYMBOL_UPPER': Decimal}}
    """
    global paper_portfolio 
    
    base_currency_from_symbol = symbol.split('-')[0].upper() # e.g., "ADA"

    logging.info(f"[PAPER TRADE ATTEMPT] {order_type.upper()} {amount_decimal:.8f} {base_currency_from_symbol} at {price_decimal:.4f}")

    # Apply simulated slippage (as in your backtester's _execute_order if desired)
    # For now, let's assume the price_decimal is the execution price for simplicity here,
    # or add slippage logic if you prefer. The strategy itself might already account for some via its
    # grid placement logic or target profit. The backtester's set_backtester_context also
    # received a simulated_slippage_percentage, but GridStrategy doesn't use it directly;
    # the backtester's _execute_order did. We can add that here if needed.
    
    # For simplicity, let's assume price_decimal is the fill price for now.
    # More advanced:
    # adjusted_price_decimal = price_decimal
    # slippage_percent = grid_strategy_instance.simulated_slippage_percentage # If accessible
    # if order_type == 'buy':
    #     adjusted_price_decimal = price_decimal * (Decimal('1') + slippage_percent)
    # elif order_type == 'sell':
    #     adjusted_price_decimal = price_decimal * (Decimal('1') - slippage_percent)
    # cost_or_revenue = adjusted_price_decimal * amount_decimal

    cost_or_revenue = price_decimal * amount_decimal # Using price_decimal directly for now
    commission = cost_or_revenue * Decimal(str(COMMISSION_RATE)) 

    if order_type == 'buy':
        total_cost = cost_or_revenue + commission
        if paper_portfolio['cash'] >= total_cost:
            paper_portfolio['cash'] -= total_cost
            paper_portfolio['crypto_holdings'][base_currency_from_symbol] = \
                paper_portfolio['crypto_holdings'].get(base_currency_from_symbol, Decimal('0')) + amount_decimal
            
            logging.info(f"[PAPER BUY EXECUTED] Bought {amount_decimal:.8f} {base_currency_from_symbol} @ {price_decimal:.4f}. Cost: {cost_or_revenue:.2f}, Comm: {commission:.2f}")
            logging.info(f"[PAPER PORTFOLIO] Cash: {paper_portfolio['cash']:.2f}, Holdings: {paper_portfolio['crypto_holdings']}")
            # The strategy will call notify_order_filled itself if this returns True
            return True 
        else:
            logging.warning(f"[PAPER BUY FAILED] Insufficient cash: Need {total_cost:.2f}, Have {paper_portfolio['cash']:.2f}")
            return False
    elif order_type == 'sell':
        current_base_holding = paper_portfolio['crypto_holdings'].get(base_currency_from_symbol, Decimal('0'))
        if current_base_holding >= amount_decimal:
            paper_portfolio['cash'] += (cost_or_revenue - commission)
            paper_portfolio['crypto_holdings'][base_currency_from_symbol] -= amount_decimal

            logging.info(f"[PAPER SELL EXECUTED] Sold {amount_decimal:.8f} {base_currency_from_symbol} @ {price_decimal:.4f}. Revenue: {cost_or_revenue:.2f}, Comm: {commission:.2f}")
            logging.info(f"[PAPER PORTFOLIO] Cash: {paper_portfolio['cash']:.2f}, Holdings: {paper_portfolio['crypto_holdings']}")
            return True
        else:
            logging.warning(f"[PAPER SELL FAILED] Insufficient {base_currency_from_symbol}: Need {amount_decimal:.8f}, Have {current_base_holding:.8f}")
            return False
    return False

# --- WebSocket Event Handlers ---
def on_open_handler(ws_app):
    # (Same as before: logs connection, sends subscription messages for candles and heartbeats)
    logging.info("Connection opened. Subscribing to channels...")
    try:
        subscribe_candles_message = {"type": "subscribe", "product_ids": [PRODUCT_ID], "channel": "candles"}
        logging.info(f"Sending subscription for 'candles': {subscribe_candles_message}")
        ws_app.send(json.dumps(subscribe_candles_message))

        subscribe_heartbeats_message = {"type": "subscribe", "channel": "heartbeats"}
        logging.info(f"Sending subscription for 'heartbeats': {subscribe_heartbeats_message}")
        ws_app.send(json.dumps(subscribe_heartbeats_message))
        logging.info("All subscription messages sent.")
    except Exception as e:
        logging.error(f"Error during on_open_handler subscriptions: {e}", exc_info=True)


def on_message_handler(ws_app, raw_message_str):
    # (Same as before: parses messages, processes candles, calls strategy.on_candle)
    global PRODUCT_ID, grid_strategy_instance, is_first_candle_processed 
    
    if not grid_strategy_instance: 
        return

    try:
        message_dict = json.loads(raw_message_str)
        channel = message_dict.get("channel")
        msg_type = message_dict.get("type")
        
        if channel == "candles":
            events = message_dict.get("events", [])
            for event in events:
                candles_data_list = event.get("candles", [])
                for candle_dict in candles_data_list:
                    if candle_dict.get("product_id") == PRODUCT_ID:
                        try:
                            start_ts_unix_str = candle_dict.get('start')
                            if start_ts_unix_str is None:
                                logging.warning(f"Candle for {PRODUCT_ID} missing 'start': {candle_dict}")
                                continue
                            
                            candle_start_dt = datetime.fromtimestamp(int(start_ts_unix_str), tz=timezone.utc)
                            open_price = Decimal(str(candle_dict.get('open')))
                            high_price = Decimal(str(candle_dict.get('high')))
                            low_price = Decimal(str(candle_dict.get('low')))
                            close_price = Decimal(str(candle_dict.get('close')))
                            volume = Decimal(str(candle_dict.get('volume')))

                            logging.info(
                                f"PROCESSED CANDLE [{PRODUCT_ID}]: "
                                f"Time: {candle_start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}, "
                                f"O:{open_price} H:{high_price} L:{low_price} C:{close_price} V:{volume}"
                            )
                            
                            current_candle_for_strategy_dict = {
                                'open': open_price, 'high': high_price, 'low': low_price,
                                'close': close_price, 'volume': volume
                            }
                            current_candle_series = pd.Series(current_candle_for_strategy_dict)
                            
                            if not is_first_candle_processed:
                                grid_strategy_instance.current_price = close_price 
                                logging.info(f"First candle processed. Initial strategy price set to: {close_price}")
                                
                                grid_strategy_instance.on_candle(PRODUCT_ID, current_candle_series, close_price, candle_start_dt)
                                is_first_candle_processed = True # Set flag AFTER first on_candle call
                                
                                if grid_strategy_instance.is_grid_active:
                                    logging.info("Grid is active after first candle, attempting to place initial orders via _place_initial_orders...")
                                    grid_strategy_instance._place_initial_orders() 
                                else:
                                    logging.info("Grid is NOT active after first candle. No initial orders by _place_initial_orders.")
                            else:
                                grid_strategy_instance.on_candle(PRODUCT_ID, current_candle_series, close_price, candle_start_dt)

                        except (InvalidOperation, ValueError, AttributeError) as e: # More specific catches
                            logging.error(f"Error processing candle data fields for {PRODUCT_ID}: {e}. Data: {candle_dict}", exc_info=True)
                        except Exception as e:
                            logging.error(f"General error processing specific candle for {PRODUCT_ID}: {e}", exc_info=True)
                            
        elif channel == "heartbeats": # ... (heartbeat logging)
            events = message_dict.get("events", [])
            for event in events: logging.info(f"HEARTBEAT: Counter: {event.get('heartbeat_counter')}, Time: {event.get('current_time')}")
        elif channel == "subscriptions":  # ... (subscription ack logging)
             logging.info(f"SUBSCRIPTION ACK: {message_dict}")
        elif message_dict.get("type") == "error" or "message" in message_dict : # ... (error logging)
            logging.error(f"ERROR MESSAGE from WebSocket: {message_dict}")

    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from message: {raw_message_str}")
    except Exception as e:
        logging.error(f"Outer error in on_message_handler: {e} - Raw: {raw_message_str}", exc_info=True)

def on_error_handler(ws_app, error):
    # (Same as before)
    logging.error(f"WebSocket on_error_handler triggered. Error Type: {type(error)}, Error: {error}", exc_info=True)

def on_close_handler(ws_app, close_status_code, close_msg):
    # (Same as before)
    logging.info(f"WebSocket on_close_handler triggered. Status Code: {close_status_code}, Close Message: {close_msg}")

def setup_strategy_and_portfolio():
    """Initializes the paper portfolio and grid strategy instance."""
    global paper_portfolio, grid_strategy_instance, PRODUCT_ID

    base_currency_upper = PRODUCT_ID.split('-')[0].upper() 
    
    paper_portfolio = {
        'cash': Decimal(str(INITIAL_BACKTEST_CAPITAL)),
        'crypto_holdings': {base_currency_upper: Decimal('0')} 
    }
    logging.info(f"Initial Paper Portfolio: {paper_portfolio}")

    strategy_params_for_instance = GRID_STRATEGY_PARAMS.copy()
    strategy_params_for_instance["symbol"] = PRODUCT_ID 

    grid_strategy_instance = GridStrategy(strategy_params_for_instance)
    
    # The simulated_slippage_percentage is now used by GridStrategy itself
    # when it calls execute_order_func. So, we need to ensure it's passed correctly.
    # The backtester was passing it to set_backtester_context, and GridStrategy was storing it.
    # Our execute_paper_trade function above does not currently use this stored slippage from strategy.
    # Let's simplify: the execute_paper_trade function here defines its own slippage or we assume
    # the price passed to it IS the execution price for now.
    # The GridStrategy's set_backtester_context has simulated_slippage_percentage parameter.
    # Let's ensure our current GridStrategy (from Code.txt) stores this if needed.
    # Yes, GridStrategy stores self.simulated_slippage_percentage from set_backtester_context
    # but it's the *backtester's* _execute_order that applies it.
    # Our execute_paper_trade will need to apply it if we want that level of simulation here.
    # For now, I've kept execute_paper_trade simple and it doesn't apply extra slippage itself.
    # The strategy receives the slippage % but doesn't use it when calling execute_order_func.
    
    grid_strategy_instance.set_backtester_context( 
        portfolio=paper_portfolio, 
        execute_order_func=execute_paper_trade, 
        commission_rate=Decimal(str(COMMISSION_RATE)),
        min_order_sizes_map=COINBASE_MIN_ORDER_SIZES, 
        simulated_slippage_percentage=Decimal(str(GRID_STRATEGY_PARAMS.get("simulated_slippage_percentage", "0.0005")))
    )
    logging.info("GridStrategy instance created and execution context set.")


def main_websocket_loop():
    # (Same as before: calls setup_strategy_and_portfolio, creates WebSocketApp, calls run_forever)
    global ws_client
    
    setup_strategy_and_portfolio() 

    if not grid_strategy_instance:
        logging.error("Grid strategy not initialized. Exiting.")
        return

    logging.info(f"Attempting to connect to WebSocket: {MARKET_DATA_WS_URL}")
    ws_app = websocket.WebSocketApp(
        MARKET_DATA_WS_URL,
        on_open=on_open_handler,
        on_message=on_message_handler,
        on_error=on_error_handler,
        on_close=on_close_handler
    )

    logging.info("WebSocketApp instance created. Calling run_forever()...")
    try:
        ws_app.run_forever() 
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received by main try-except. Closing connection...")
    except Exception as e:
        logging.error(f"An unexpected error occurred in main execution loop: {e}", exc_info=True)
    finally:
        if 'ws_app' in locals() and hasattr(ws_app, 'sock') and ws_app.sock and hasattr(ws_app.sock, 'connected') and ws_app.sock.connected:
             logging.info("Attempting to explicitly close WebSocket connection in main finally block...")
             ws_app.close()
        logging.info("WebSocket listener script finished.")

# --- Main Execution ---
if __name__ == "__main__":
    # websocket.enableTrace(True) 
    main_websocket_loop()