# core/strategy.py

import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
import pandas_ta as ta
import logging

class GridStrategy:
    def __init__(self, params):
        self.symbol = params["symbol"]
        self.config_params = params
        self.grid_levels = []
        self.active_buy_orders = []
        self.active_sell_orders = []
        self.is_grid_active = False
        self.initial_grid_placed = False
        self.current_price = None
        self.ohlcv_history = pd.DataFrame()

        # ADX Settings
        self.adx_period = int(params.get("adx_period", 14))
        self.adx_entry_threshold = Decimal(str(params.get("adx_entry_threshold", "20.0")))
        self.adx_exit_threshold = Decimal(str(params.get("adx_exit_threshold", "35.0")))
        self.adx_confirmation_periods = int(params.get("adx_confirmation_periods", 3))
        self.adx_history = []

        # Context from main app
        self.portfolio = None
        self.execute_order_func = None
        self.get_next_order_id = None
        
        # Dashboard Data
        self.equity_history = []

    def set_backtester_context(self, portfolio, execute_order_func, get_next_order_id):
        self.portfolio = portfolio
        self.execute_order_func = execute_order_func
        self.get_next_order_id = get_next_order_id

    def add_historical_candle(self, candle_series, dt):
        """Only adds a candle to the history."""
        new_row_df = pd.DataFrame([candle_series.to_dict()], index=[pd.to_datetime(dt)])
        self.ohlcv_history = pd.concat([self.ohlcv_history, new_row_df])

    def _calculate_amount(self, price):
        """Calculates order size."""
        # This is a simplified calculation; a real one would be more complex
        equity_to_use = self.portfolio['cash'] * Decimal("0.1") # Use 10% of cash
        return (equity_to_use / price).quantize(Decimal('0.00000001'))

    def _create_grid(self):
        """Lays out the grid levels."""
        if len(self.ohlcv_history) < self.adx_period: return
        self.grid_levels = []
        num_grids = int(self.config_params.get("number_of_grids", 10))
        # Simple static grid for stability. ATR logic can be re-added later.
        step = self.current_price * Decimal('0.01') # 1% step
        for i in range(1, (num_grids // 2) + 1):
            self.grid_levels.append({'price': self.current_price - (i * step), 'type': 'buy'})
            self.grid_levels.append({'price': self.current_price + (i * step), 'type': 'sell'})
        logging.info(f"[{self.symbol}] Created {len(self.grid_levels)} grid levels.")

    def _place_initial_orders(self):
        """Places initial buy orders and their corresponding sell orders."""
        self.active_buy_orders, self.active_sell_orders = [], []
        for level in self.grid_levels:
            if level['type'] == 'buy':
                amount = self._calculate_amount(level['price'])
                if amount > 0:
                    order_id = self.get_next_order_id()
                    self.active_buy_orders.append({'order_id': order_id, 'price': level['price'], 'amount': amount})
                    logging.info(f"[{self.symbol}] Staged initial BUY {order_id} @ {level['price']:.4f}")
        self.initial_grid_placed = True

    def _check_for_fills(self):
        """Simulates order fills."""
        filled_buy_orders = []
        for order in self.active_buy_orders:
            if self.current_price <= order['price']:
                self.execute_order_func(self.symbol, 'buy', order['price'], order['amount'], order['order_id'])
                filled_buy_orders.append(order)
        self.active_buy_orders = [o for o in self.active_buy_orders if o not in filled_buy_orders]

        filled_sell_orders = []
        for order in self.active_sell_orders:
            if self.current_price >= order['price']:
                self.execute_order_func(self.symbol, 'sell', order['price'], order['amount'], order['order_id'])
                filled_sell_orders.append(order)
        self.active_sell_orders = [o for o in self.active_sell_orders if o not in filled_sell_orders]

    def on_live_candle(self, candle_series, dt):
        """Main logic loop for live candles."""
        self.add_historical_candle(candle_series, dt)
        self.current_price = candle_series['close']

        # ADX Calculation
        adx_val = 0.0
        if len(self.ohlcv_history) >= self.adx_period * 2:
            try:
                adx_series = ta.adx(self.ohlcv_history['high'].astype(float), self.ohlcv_history['low'].astype(float), self.ohlcv_history['close'].astype(float), length=self.adx_period)
                if adx_series is not None and not pd.isna(adx_series.iloc[-1, 0]):
                    adx_val = adx_series.iloc[-1, 0]
            except Exception: pass
        self.adx_history.append(adx_val)
        self.adx_history = self.adx_history[-self.adx_confirmation_periods:]
        
        should_be_active = all(x < self.adx_entry_threshold for x in self.adx_history) and len(self.adx_history) == self.adx_confirmation_periods
        should_deactivate = all(x > self.adx_exit_threshold for x in self.adx_history)

        if not self.is_grid_active and should_be_active:
            self.is_grid_active = True
            self.initial_grid_placed = False
            logging.info(f"[{self.symbol}] ADX ENTRY: {self.adx_history[-1]:.2f} < {self.adx_entry_threshold}. Activating grid.")
        
        elif self.is_grid_active and should_deactivate:
            self.is_grid_active = False
            self.initial_grid_placed = False
            logging.info(f"[{self.symbol}] ADX EXIT: {self.adx_history[-1]:.2f} > {self.adx_exit_threshold}. Deactivating grid.")
            self.active_buy_orders, self.active_sell_orders = [], []

        if self.is_grid_active:
            if not self.initial_grid_placed:
                self._create_grid() # <-- This is the corrected line
                self._place_initial_orders()
            else:
                self._check_for_fills()
        
        # Always update equity history
        if self.portfolio:
            base_currency = self.symbol.split('-')[0].upper()
            holdings_value = self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')) * self.current_price
            total_equity = self.portfolio['cash'] + holdings_value
            self.equity_history.append((dt.isoformat(), total_equity))

    def notify_order_filled(self, filled_order_details):
        """Handles logic after an order is filled. NON-RECURSIVE."""
        order_type = filled_order_details['type']
        price = filled_order_details['price']
        
        if order_type == 'buy':
            # Place a corresponding sell order one level up
            sell_targets = [lvl for lvl in self.grid_levels if lvl['type'] == 'sell' and lvl['price'] > price]
            if sell_targets:
                target_level = min(sell_targets, key=lambda x: x['price'])
                amount = self._calculate_amount(target_level['price'])
                order_id = self.get_next_order_id()
                self.active_sell_orders.append({'order_id': order_id, 'price': target_level['price'], 'amount': amount})
                logging.info(f"[{self.symbol}] BUY filled. Staging counter SELL {order_id} @ {target_level['price']:.4f}")