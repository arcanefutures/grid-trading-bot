# coinbase_websocket_feed.py

import websocket
import json
import logging
import os
from dotenv import load_dotenv
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from datetime import datetime, timezone
import pandas as pd
import signal
import threading
import time

from core.strategy import GridStrategy
from config.parameters import GRID_STRATEGY_PARAMS, INITIAL_BACKTEST_CAPITAL, COMMISSION_RATE

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("PRIVATE_KEY")
PRODUCT_ID = "ADA-USD" # Default product ID
MARKET_DATA_WS_URL = "wss://advanced-trade-ws.coinbase.com"

# --- Global State ---
grid_strategy_instance = None
paper_portfolio = None
is_first_candle_processed = False
ws_client = None
paper_trade_log = []
crypto_cost_basis = {}
shutdown_requested = False
MIN_ORDER_VALUE_USD = Decimal('5.00')

# --- Custom JSON Encoder for Decimal ---
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super(DecimalEncoder, self).default(obj)

# --- Signal Handler for Graceful Shutdown ---
def signal_handler(signum, frame):
    global shutdown_requested
    logging.info(f"Signal {signum} received, requesting shutdown.")
    shutdown_requested = True

# --- Dashboard Data Saver ---
def save_dashboard_data():
    """Gathers state and saves it to JSON files for the dashboard to read."""
    global paper_portfolio, grid_strategy_instance, paper_trade_log
    
    if not grid_strategy_instance or not paper_portfolio:
        return

    # --- Prepare data ---
    base_currency = PRODUCT_ID.split('-')[0].upper()
    current_price = grid_strategy_instance.current_price if grid_strategy_instance.current_price else Decimal('0')
    
    holdings_amount = paper_portfolio['crypto_holdings'].get(base_currency, Decimal('0'))
    holdings_value = holdings_amount * current_price
    total_equity = paper_portfolio['cash'] + holdings_value
    
    realized_pnl = sum(trade.get('realized_pnl', Decimal('0')) for trade in paper_trade_log)
    
    unrealized_pnl = Decimal('0')
    cost_basis_info = crypto_cost_basis.get(base_currency)
    if cost_basis_info and cost_basis_info.get('total_amount', Decimal('0')) > Decimal('0'):
        cost_of_current_holdings = cost_basis_info.get('total_cost_usd', Decimal('0'))
        unrealized_pnl = holdings_value - cost_of_current_holdings

    dashboard_data = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'is_grid_active': grid_strategy_instance.is_grid_active,
        'current_price': current_price,
        'portfolio': {
            'cash': paper_portfolio['cash'],
            'holdings_amount': holdings_amount,
            'holdings_value': holdings_value,
            'total_equity': total_equity,
        },
        'pnl': {
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'net_pnl': total_equity - Decimal(str(INITIAL_BACKTEST_CAPITAL))
        },
        'active_orders': {
            'buys': grid_strategy_instance.active_buy_orders,
            'sells': grid_strategy_instance.active_sell_orders,
        },
        'equity_history': grid_strategy_instance.equity_history,
        'trade_history': paper_trade_log # Include trade history directly
    }

    # --- Write to files ---
    try:
        with open('dashboard_status.json', 'w') as f:
            json.dump(dashboard_data, f, cls=DecimalEncoder, indent=4)
        
    except Exception as e:
        logging.error(f"Error writing dashboard data: {e}")

def run_dashboard_saver():
    """Periodically calls the save_dashboard_data function."""
    while not shutdown_requested:
        save_dashboard_data()
        time.sleep(5) # Update dashboard data every 5 seconds

# --- Paper Trading Simulation Logic ---
def execute_paper_trade(symbol, order_type, price_decimal, amount_decimal, order_id=None):
    global paper_portfolio, grid_strategy_instance, paper_trade_log, crypto_cost_basis
    
    base_currency_from_symbol = symbol.split('-')[0].upper()

    log_prefix = f"[PAPER TRADE ATTEMPT] Order ID: {order_id if order_id else 'N/A'}"
    logging.info(f"{log_prefix} {order_type.upper()} {amount_decimal:.8f} {base_currency_from_symbol} at requested price {price_decimal:.4f}")

    # --- Apply simulated slippage ---
    adjusted_price_decimal = price_decimal
    if grid_strategy_instance and hasattr(grid_strategy_instance, 'simulated_slippage_percentage'):
        slippage_percent = grid_strategy_instance.simulated_slippage_percentage
        if slippage_percent > Decimal('0'):
            if order_type == 'buy':
                adjusted_price_decimal = price_decimal * (Decimal('1') + slippage_percent)
            elif order_type == 'sell':
                adjusted_price_decimal = price_decimal * (Decimal('1') - slippage_percent)
    
    adjusted_price_decimal = adjusted_price_decimal.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)

    order_usd_value = adjusted_price_decimal * amount_decimal

    if order_usd_value < MIN_ORDER_VALUE_USD:
        logging.warning(f"{log_prefix} FAILED: Order value ${order_usd_value:.2f} is below minimum ${MIN_ORDER_VALUE_USD:.2f}. Not executing.")
        return False

    commission = order_usd_value * Decimal(str(COMMISSION_RATE))
    trade_successful = False
    realized_pnl = Decimal('0')

    if order_type == 'buy':
        total_cost = order_usd_value + commission
        if paper_portfolio['cash'] >= total_cost:
            paper_portfolio['cash'] -= total_cost
            paper_portfolio['crypto_holdings'][base_currency_from_symbol] = \
                paper_portfolio['crypto_holdings'].get(base_currency_from_symbol, Decimal('0')) + amount_decimal
            
            current_basis = crypto_cost_basis.get(base_currency_from_symbol, {'total_amount': Decimal('0'), 'total_cost_usd': Decimal('0')})
            current_basis['total_amount'] += amount_decimal
            current_basis['total_cost_usd'] += total_cost
            crypto_cost_basis[base_currency_from_symbol] = current_basis

            logging.info(f"{log_prefix} EXECUTED: Bought {amount_decimal:.8f} {base_currency_from_symbol} @ Adjusted Price: {adjusted_price_decimal:.4f}")
            trade_successful = True
        else:
            logging.warning(f"{log_prefix} FAILED: Insufficient cash: Need {total_cost:.2f}, Have {paper_portfolio['cash']:.2f}")

    elif order_type == 'sell':
        current_base_holding = paper_portfolio['crypto_holdings'].get(base_currency_from_symbol, Decimal('0'))
        if current_base_holding >= amount_decimal:
            net_revenue = order_usd_value - commission
            paper_portfolio['cash'] += net_revenue
            paper_portfolio['crypto_holdings'][base_currency_from_symbol] -= amount_decimal

            current_basis = crypto_cost_basis.get(base_currency_from_symbol, {'total_amount': Decimal('0'), 'total_cost_usd': Decimal('0')})
            if current_basis['total_amount'] > Decimal('0'):
                avg_cost_per_unit = current_basis['total_cost_usd'] / current_basis['total_amount']
                cost_of_sold_amount = avg_cost_per_unit * amount_decimal
                realized_pnl = net_revenue - cost_of_sold_amount
            
                new_total_amount = current_basis['total_amount'] - amount_decimal
                if new_total_amount > Decimal('0'):
                    current_basis['total_cost_usd'] = current_basis['total_cost_usd'] * (new_total_amount / current_basis['total_amount'])
                else:
                    current_basis['total_cost_usd'] = Decimal('0')
                current_basis['total_amount'] = new_total_amount
                crypto_cost_basis[base_currency_from_symbol] = current_basis

            logging.info(f"{log_prefix} EXECUTED: Sold {amount_decimal:.8f} {base_currency_from_symbol} @ Adjusted Price: {adjusted_price_decimal:.4f}. Realized P&L: {realized_pnl:.2f}")
            trade_successful = True
        else:
            logging.warning(f"{log_prefix} FAILED: Insufficient {base_currency_from_symbol}: Need {amount_decimal:.8f}, Have {current_base_holding:.8f}")
    
    if trade_successful:
        paper_trade_log.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol, 'order_type': order_type, 'requested_price': price_decimal,
            'filled_price': adjusted_price_decimal, 'amount': amount_decimal,
            'usd_value': order_usd_value, 'commission': commission, 'realized_pnl': realized_pnl,
            'order_id': order_id
        })
        grid_strategy_instance.notify_order_filled({
            'symbol': symbol, 'type': order_type, 'price': adjusted_price_decimal,
            'amount': amount_decimal, 'order_id': order_id
        })
    return trade_successful

# --- WebSocket Event Handlers ---
def on_open_handler(ws_app):
    logging.info("Connection opened. Subscribing to channels...")
    subscribe_candles_message = {"type": "subscribe", "product_ids": [PRODUCT_ID], "channel": "candles"}
    ws_app.send(json.dumps(subscribe_candles_message))
    subscribe_heartbeats_message = {"type": "subscribe", "channel": "heartbeats"}
    ws_app.send(json.dumps(subscribe_heartbeats_message))

def on_message_handler(ws_app, raw_message_str):
    if shutdown_requested: return

    try:
        message = json.loads(raw_message_str)
        if message.get("channel") == "candles":
            for event in message.get("events", []):
                for candle in event.get("candles", []):
                    if candle.get("product_id") == PRODUCT_ID:
                        try:
                            start_dt = datetime.fromtimestamp(int(candle['start']), tz=timezone.utc)
                            candle_series = pd.Series({
                                'open': Decimal(candle['open']), 'high': Decimal(candle['high']),
                                'low': Decimal(candle['low']), 'close': Decimal(candle['close']),
                                'volume': Decimal(candle['volume'])
                            })
                            grid_strategy_instance.on_candle(PRODUCT_ID, candle_series, candle_series['close'], start_dt)
                        except (InvalidOperation, ValueError, AttributeError) as e:
                            logging.error(f"Error processing candle data: {e}. Data: {candle}", exc_info=True)
    except Exception as e:
        logging.error(f"Error in on_message_handler: {e}", exc_info=True)

def on_error_handler(ws_app, error):
    logging.error(f"WebSocket on_error_handler triggered. Error: {error}", exc_info=True)

def on_close_handler(ws_app, close_status_code, close_msg):
    logging.info(f"WebSocket on_close_handler triggered. Status: {close_status_code} {close_msg}")

def setup_strategy_and_portfolio():
    global paper_portfolio, grid_strategy_instance, PRODUCT_ID
    base_currency = PRODUCT_ID.split('-')[0].upper()
    paper_portfolio = {'cash': Decimal(str(INITIAL_BACKTEST_CAPITAL)), 'crypto_holdings': {base_currency: Decimal('0')}}
    logging.info(f"Initial Paper Portfolio: {paper_portfolio}")
    
    params = GRID_STRATEGY_PARAMS.copy()
    params["symbol"] = PRODUCT_ID
    params["min_order_value_usd"] = str(MIN_ORDER_VALUE_USD)
    
    grid_strategy_instance = GridStrategy(params)
    grid_strategy_instance.set_backtester_context(
        portfolio=paper_portfolio, execute_order_func=execute_paper_trade,
        commission_rate=Decimal(str(COMMISSION_RATE)), min_order_sizes_map={},
        simulated_slippage_percentage=Decimal(str(GRID_STRATEGY_PARAMS.get("simulated_slippage_percentage", "0.0005")))
    )
    logging.info("GridStrategy instance created and context set.")

def print_paper_trade_summary():
    # This function is now less critical as the dashboard provides live data.
    # We can leave it for a final console summary on exit.
    logging.info("\n" + "="*50 + "\n FINAL SUMMARY \n" + "="*50)
    final_equity = paper_portfolio['cash']
    if grid_strategy_instance and grid_strategy_instance.current_price:
        base_currency = PRODUCT_ID.split('-')[0].upper()
        holdings_value = paper_portfolio['crypto_holdings'].get(base_currency, Decimal('0')) * grid_strategy_instance.current_price
        final_equity += holdings_value
    logging.info(f"Final Equity: ${final_equity:.2f}")
    logging.info(f"Total Trades: {len(paper_trade_log)}")

# --- Main Execution ---
def main_websocket_loop():
    global ws_client
    signal.signal(signal.SIGINT, signal_handler)
    
    setup_strategy_and_portfolio() 

    dashboard_thread = threading.Thread(target=run_dashboard_saver, daemon=True)
    dashboard_thread.start()
    logging.info("Dashboard data saver thread started.")

    ws_app = websocket.WebSocketApp(
        MARKET_DATA_WS_URL,
        on_open=on_open_handler, on_message=on_message_handler,
        on_error=on_error_handler, on_close=on_close_handler
    )

    logging.info("WebSocketApp instance created. Calling run_forever()...")
    try:
        ws_app.run_forever(ping_interval=10, ping_timeout=5)
    except Exception as e:
        logging.error(f"An unexpected error occurred in main execution loop: {e}", exc_info=True)
    finally:
        if ws_app and hasattr(ws_app, 'sock') and ws_app.sock and ws_app.sock.connected:
            ws_app.close()
        logging.info("WebSocket listener finished.")
        save_dashboard_data()
        print_paper_trade_summary()

if __name__ == "__main__":
    main_websocket_loop()