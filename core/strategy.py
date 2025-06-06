# core/strategy.py

import pandas as pd
import math
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta, timezone
import pandas_ta as ta
import logging

class GridStrategy:
    def __init__(self, params):
        self.symbol = params["symbol"]
        self.config_params = params

        self.grid_lower_bound = None
        self.grid_upper_bound = None
        self.grid_levels = [] # Stores {'price': Decimal, 'type': 'buy'/'sell', 'status': 'open'/'pending'/'filled'}

        self.active_buy_orders = []  # Stores active orders placed: {'order_id': int, 'symbol': str, 'type': str, 'price': Decimal, 'amount': Decimal, 'original_grid_level_price': Decimal}
        self.active_sell_orders = [] # Stores active orders placed

        self.portfolio = None
        self.execute_order_func = None
        self.commission_rate = None
        self.simulated_slippage_percentage = None

        self.current_price = None

        self.atr_period = params.get("atr_period", 14)
        self.grid_atr_multiplier = Decimal(str(params.get("grid_atr_multiplier", "3.0")))
        self.ohlcv_history = pd.DataFrame()

        # --- ADX Parameters and State ---
        self.is_grid_active = False
        self.adx_period = params.get("adx_period", 14)
        self.adx_entry_threshold = params.get("adx_entry_threshold", 25.0)
        self.adx_exit_threshold = params.get("adx_exit_threshold", 30.0)
        self.adx_confirmation_periods = params.get("adx_confirmation_periods", 3)
        self.adx_history = []
        self.valid_adx_readings_count = 0
        self.adx_stable_readings_required = self.config_params.get("adx_stable_readings_required", 5)

        self.next_order_id = 1
        self.pending_order_ids = set() # This line ensures pending_order_ids is initialized

        # New for minimum USD order value at strategy level
        self.min_order_value_usd = Decimal(str(params.get("min_order_value_usd", "5.00")))
        
        # --- NEW: For Dashboard ---
        self.equity_history = []


    def set_backtester_context(self, portfolio, execute_order_func, commission_rate, min_order_sizes_map, simulated_slippage_percentage):
        """
        Sets the necessary context for the strategy to operate.
        """
        self.portfolio = portfolio
        self.execute_order_func = execute_order_func
        self.commission_rate = Decimal(str(commission_rate))
        self.simulated_slippage_percentage = Decimal(str(simulated_slippage_percentage))
        self.min_order_value_usd = Decimal(str(self.config_params.get("min_order_value_usd", "5.00")))


    def calculate_grid_levels(self, initial_price):
        """
        For dynamic grid, this method primarily prepares initial data if needed.
        Actual grid levels are calculated dynamically in _recalculate_dynamic_grid.
        Called once at the start of a backtest (or potentially live session).
        """
        pass


    def _calculate_amount_for_order(self, symbol_pair, price_decimal):
        """
        Calculates the base asset amount for an order based on a percentage of current equity,
        applies a maximum order size cap, and checks against a MINIMUM USD VALUE.
        """
        logging.debug(f"[_calc_amount] Calculating for {symbol_pair} at {price_decimal:.4f}")
        base_currency_key = symbol_pair.split('-')[0].upper()

        price_for_equity_calc = self.current_price
        if not isinstance(price_for_equity_calc, Decimal) or price_for_equity_calc is None or price_for_equity_calc <= Decimal('0'):
            logging.warning(f"[_calc_amount] self.current_price is invalid or not Decimal ({self.current_price}, Type: {type(self.current_price)}). Using order price {price_decimal} for equity calculation instead.")
            if not isinstance(price_decimal, Decimal) or price_decimal <= Decimal('0'):
                logging.error(f"[_calc_amount] Critical: Both self.current_price and order price {price_decimal} are invalid for equity calculation. Returning 0.")
                return Decimal('0')
            price_for_equity_calc = price_decimal

        holding_qty_from_portfolio = self.portfolio['crypto_holdings'].get(base_currency_key, Decimal('0'))
        if not isinstance(holding_qty_from_portfolio, Decimal):
            try:
                holding_qty = Decimal(str(holding_qty_from_portfolio))
            except Exception as e:
                logging.error(f"[_calc_amount] Could not convert holding_qty '{holding_qty_from_portfolio}' to Decimal. Error: {e}. Returning 0 for safety.")
                return Decimal('0')
        else:
            holding_qty = holding_qty_from_portfolio
            
        crypto_value = holding_qty * price_for_equity_calc
        
        cash_value_from_portfolio = self.portfolio['cash']
        if not isinstance(cash_value_from_portfolio, Decimal):
            try:
                cash_value = Decimal(str(cash_value_from_portfolio))
            except Exception as e:
                logging.error(f"[_calc_amount] Could not convert cash_value '{cash_value_from_portfolio}' to Decimal. Error: {e}. Crypto_value was {crypto_value}. Returning 0 for safety.")
                return Decimal('0')
        else:
            cash_value = cash_value_from_portfolio
                
        current_equity = cash_value + crypto_value

        percentage_per_order = Decimal(str(self.config_params["percentage_of_equity_per_grid_order"]))
        target_usd_amount = current_equity * percentage_per_order

        max_order_size_usd = Decimal(str(self.config_params.get("max_order_size_usd", "1000000.0")))
        if target_usd_amount > max_order_size_usd:
            target_usd_amount = max_order_size_usd

        if target_usd_amount <= Decimal('0'):
            return Decimal('0')

        if not isinstance(price_decimal, Decimal) or price_decimal <= Decimal('0'):
            return Decimal('0')

        base_asset_amount = (target_usd_amount / price_decimal).quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
        
        min_base_asset_from_usd = (self.min_order_value_usd / price_decimal).quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
        
        if base_asset_amount < min_base_asset_from_usd:
            base_asset_amount = min_base_asset_from_usd
            logging.info(f"[_calc_amount] Calculated amount {base_asset_amount:.8f} for {base_currency_key} adjusted up to minimum equivalent of ${self.min_order_value_usd:.2f} due to price {price_decimal:.4f}.")
            
        return base_asset_amount


    def _place_initial_orders(self):
        """
        Handles initial portfolio balancing for 'neutral' grid setups.
        For 'cash_only', this method does nothing, and the grid orders are placed
        as market conditions (ADX) become favorable via _manage_grid_orders.
        """
        if self.config_params["initial_position_type"] == "neutral" and self.is_grid_active:
            # This logic should be here if we want to deploy capital immediately for a neutral strategy.
            # However, for 'cash_only', this block is skipped.
            logging.warning(f"[{self.symbol}] _place_initial_orders with 'neutral' type not yet fully implemented/tested.")
            pass
        elif self.config_params["initial_position_type"] == "cash_only":
            logging.info(f"[{self.symbol}] Initial position type is 'cash_only'. No initial balancing orders are placed by _place_initial_orders.")


    def _recalculate_dynamic_grid(self, current_price_decimal):
        """
        Calculates ATR and dynamically adjusts grid levels based on current market conditions.
        Crucially, it preserves the 'filled' status of levels that were previously filled,
        and sets 'open' for levels that are new or were previously 'pending' but haven't filled.
        """
        if not isinstance(current_price_decimal, Decimal) or current_price_decimal <= Decimal('0'):
            logging.error(f"[_recalculate_dynamic_grid] Invalid current_price_decimal: {current_price_decimal}. Cannot recalculate grid.")
            self.grid_levels = []
            return

        if len(self.ohlcv_history) < self.atr_period:
            self.grid_levels = []
            return

        try:
            atr_series = ta.atr(self.ohlcv_history['high'].astype(float),
                                  self.ohlcv_history['low'].astype(float),
                                  self.ohlcv_history['close'].astype(float),
                                  length=self.atr_period)
        except Exception as e:
            logging.error(f"Error calculating ATR: {e}", exc_info=True)
            self.grid_levels = []
            return

        if atr_series is None or atr_series.empty or pd.isna(atr_series.iloc[-1]):
            self.grid_levels = []
            return

        current_atr = Decimal(str(atr_series.iloc[-1]))
        
        min_profitable_price_movement = current_price_decimal * (
                (Decimal('2') * self.commission_rate) + Decimal(str(self.config_params["profit_per_grid_percentage"]))
            )

        if current_atr <= Decimal('0'):
            self.grid_levels = []
            return

        total_grid_width = current_atr * self.grid_atr_multiplier
        if total_grid_width < min_profitable_price_movement * Decimal(str(self.config_params["number_of_grids"])):
            total_grid_width = min_profitable_price_movement * Decimal(str(self.config_params["number_of_grids"]))

        self.grid_lower_bound = current_price_decimal - (total_grid_width / Decimal('2'))
        self.grid_upper_bound = current_price_decimal + (total_grid_width / Decimal('2'))

        if self.grid_lower_bound <= Decimal('0'):
            self.grid_lower_bound = Decimal('0.000001')

        num_grids_param = self.config_params.get("number_of_grids", 10)
        num_grids = Decimal(str(num_grids_param))

        if num_grids <= Decimal('0'):
            self.grid_levels = []
            return

        grid_step_size = max(total_grid_width / num_grids, min_profitable_price_movement)

        if grid_step_size <= Decimal('0'):
            self.grid_levels = []
            return

        current_grid_levels_map = {} # Use a map for efficient lookup by price and type
        
        # Add buy levels below current price
        current_grid_price_buy = current_price_decimal - grid_step_size / Decimal('2')
        for i in range(int(num_grids / Decimal('2'))):
            price = current_grid_price_buy - (grid_step_size * Decimal(str(i)))
            price = price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            if price > Decimal('0') and price >= self.grid_lower_bound:
                current_grid_levels_map[(price, 'buy')] = {'price': price, 'type': 'buy', 'status': 'open'}
            else:
                break

        # Add sell levels above current price
        current_grid_price_sell = current_price_decimal + grid_step_size / Decimal('2')
        for i in range(int(num_grids / Decimal('2'))):
            price = current_grid_price_sell + (grid_step_size * Decimal(str(i)))
            price = price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            if price <= self.grid_upper_bound:
                current_grid_levels_map[(price, 'sell')] = {'price': price, 'type': 'sell', 'status': 'open'}
            else:
                break

        # Transfer status from old grid_levels and active orders to new_grid_levels
        
        # Iterate through old grid_levels first to preserve 'filled' status
        for old_level in self.grid_levels:
            key = (old_level['price'], old_level['type'])
            if key in current_grid_levels_map:
                if old_level['status'] == 'filled':
                    current_grid_levels_map[key]['status'] = 'filled'
                    
        # Then, iterate through active orders to mark 'pending' status
        # This is crucial for `_manage_grid_orders` to know which levels already have pending orders.
        for order in self.active_buy_orders + self.active_sell_orders:
            key = (order['price'], order['type'])
            if key in current_grid_levels_map:
                current_grid_levels_map[key]['status'] = 'pending'
            # No else clause needed here; _manage_grid_orders will handle stale orders.

        self.grid_levels = sorted(list(current_grid_levels_map.values()), key=lambda x: x['price'])
        
        logging.debug(f"[{self.symbol}] Dynamic Grid Re-calculated: {len(self.grid_levels)} levels. ATR: {current_atr:.2f}, Range: {self.grid_lower_bound:.2f} - {self.grid_upper_bound:.2f}")


    def _manage_grid_orders(self):
        """
        Manages active grid orders based on current price and grid levels.
        This function identifies orders to be placed and returns them.
        It explicitly marks levels as 'open' in self.grid_levels based on conditions.
        It does NOT directly modify active_buy_orders/active_sell_orders or call execute_order_func.
        """
        logging.debug(f"[{self.symbol}] Entering _manage_grid_orders. Current Price: {self.current_price:.4f}")
        logging.debug(f"[{self.symbol}] Active Buys (before this cycle's placement): {len(self.active_buy_orders)}, Active Sells: {len(self.active_sell_orders)}")
        logging.debug(f"[{self.symbol}] Total Grid Levels: {len(self.grid_levels)}")

        orders_to_place = []

        valid_grid_level_identifiers = set()
        for level in self.grid_levels:
            valid_grid_level_identifiers.add((level['price'], level['type']))

        new_active_buy_orders_list = []
        for order in self.active_buy_orders:
            if (order['price'], order['type']) in valid_grid_level_identifiers:
                new_active_buy_orders_list.append(order)
            else:
                logging.info(f"[{self.symbol}] Managing: Cancelling stale BUY order {order.get('order_id', 'N/A')} at {order['price']:.2f} (level no longer in current grid).")
                if order.get('order_id') in self.pending_order_ids:
                    self.pending_order_ids.remove(order['order_id'])
        self.active_buy_orders = new_active_buy_orders_list

        new_active_sell_orders_list = []
        for order in self.active_sell_orders:
            if (order['price'], order['type']) in valid_grid_level_identifiers:
                new_active_sell_orders_list.append(order)
            else:
                logging.info(f"[{self.symbol}] Managing: Cancelling stale SELL order {order.get('order_id', 'N/A')} at {order['price']:.2f} (level no longer in current grid).")
                if order.get('order_id') in self.pending_order_ids:
                    self.pending_order_ids.remove(order['order_id'])
        self.active_sell_orders = new_active_sell_orders_list

        current_active_order_identifiers_after_cleanup = set()
        for order in self.active_buy_orders + self.active_sell_orders:
            current_active_order_identifiers_after_cleanup.add((order['price'], order['type']))


        for i, level in enumerate(self.grid_levels):
            if (level['price'], level['type']) in current_active_order_identifiers_after_cleanup:
                continue
            if level['status'] == 'filled':
                continue

            if level['status'] == 'open':
                can_place_order = False
                if level['type'] == 'buy':
                    if len(self.active_buy_orders) < self.config_params["max_concurrent_orders_per_side"]:
                        can_place_order = True
                elif level['type'] == 'sell':
                    base_currency = self.symbol.split('-')[0].upper()
                    if self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')) > Decimal('0'):
                        if len(self.active_sell_orders) < self.config_params["max_concurrent_orders_per_side"]:
                            can_place_order = True

                if can_place_order:
                    amount_to_trade = self._calculate_amount_for_order(self.symbol, level['price'])
                    if amount_to_trade == Decimal('0'):
                        logging.debug(f"[{self.symbol}] Calculated amount 0 for {level['type']} at {level['price']:.2f}. Skipping order preparation.")
                        continue

                    order_id = self.next_order_id
                    self.next_order_id += 1
                    
                    prepared_order = {
                        'order_id': order_id,
                        'symbol': self.symbol,
                        'type': level['type'],
                        'price': level['price'],
                        'amount': amount_to_trade,
                        'original_grid_level_price': level['price'],
                        'grid_level_idx': i
                    }
                    orders_to_place.append(prepared_order)
                    self.grid_levels[i]['status'] = 'pending' 
                    current_active_order_identifiers_after_cleanup.add((level['price'], level['type'])) 

        return orders_to_place


    def on_candle(self, symbol, candle_data_series, current_price_decimal, current_time_dt):
        if not isinstance(current_price_decimal, Decimal):
            logging.error(f"[{self.symbol}][on_candle] FATAL: current_price_decimal is not Decimal: {current_price_decimal} (Type: {type(current_price_decimal)})")
            return
        if not isinstance(current_time_dt, datetime):
            logging.error(f"[{self.symbol}][on_candle] FATAL: current_time_dt is not datetime: {current_time_dt} (Type: {type(current_time_dt)})")
            return
        expected_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not expected_cols.issubset(candle_data_series.index):
            logging.error(f"[{self.symbol}][on_candle] FATAL: candle_data_series missing columns: {expected_cols - set(candle_data_series.index)}")
            return

        self.current_price = current_price_decimal
        
        new_row_df = pd.DataFrame([candle_data_series.to_dict()], index=[current_time_dt])
        
        if not self.ohlcv_history.empty:
            new_row_df = new_row_df.reindex(columns=self.ohlcv_history.columns)
            for col in new_row_df.columns:
                if col in candle_data_series:
                    new_row_df.loc[current_time_dt, col] = Decimal(str(candle_data_series[col]))

        self.ohlcv_history = pd.concat([self.ohlcv_history, new_row_df])
        required_history_length = max(self.atr_period + 10, self.adx_period * 2 + self.adx_confirmation_periods + 10)
        if len(self.ohlcv_history) > required_history_length:
            self.ohlcv_history = self.ohlcv_history.iloc[-required_history_length:]
        
        logging.debug(f"[{self.symbol}][on_candle] OHLCV history length: {len(self.ohlcv_history)}")

        min_len_for_adx_calc_and_confirm = (self.adx_period * 2 -1) + self.adx_confirmation_periods
        current_adx_float = 0.0

        if len(self.ohlcv_history) >= min_len_for_adx_calc_and_confirm :
            try:
                high_float = self.ohlcv_history['high'].astype(float)
                low_float = self.ohlcv_history['low'].astype(float)
                close_float = self.ohlcv_history['close'].astype(float)
                adx_df = ta.adx(high=high_float, low=low_float, close=close_float, length=self.adx_period)
                if adx_df is not None and not adx_df.empty and f'ADX_{self.adx_period}' in adx_df.columns:
                    current_adx_value_from_pandas_ta = adx_df[f'ADX_{self.adx_period}'].iloc[-1]
                    if pd.isna(current_adx_value_from_pandas_ta):
                        self.valid_adx_readings_count = 0
                    else:
                        current_adx_float = float(current_adx_value_from_pandas_ta)
                        self.valid_adx_readings_count += 1
                else:
                    self.valid_adx_readings_count = 0
            except Exception as e:
                self.valid_adx_readings_count = 0
                logging.error(f"[{self.symbol}][on_candle] Error calculating ADX: {e}", exc_info=True)
        else:
            self.valid_adx_readings_count = 0

        self.adx_history.append(current_adx_float)
        self.adx_history = self.adx_history[-self.adx_confirmation_periods:]
        adx_sufficient_history_for_confirmation = len(self.adx_history) >= self.adx_confirmation_periods
        adx_warmup_complete = self.valid_adx_readings_count >= self.adx_stable_readings_required

        if not self.is_grid_active:
            if adx_sufficient_history_for_confirmation and adx_warmup_complete and all(adx_val < self.adx_entry_threshold for adx_val in self.adx_history):
                self.is_grid_active = True
                logging.info(f"[{self.symbol}][on_candle] ADX ENTRY SIGNAL (ADX hist: {[f'{x:.2f}' for x in self.adx_history]}): Activating grid.")
        elif self.is_grid_active:
            if adx_sufficient_history_for_confirmation and all(adx_val > self.adx_exit_threshold for adx_val in self.adx_history):
                self.is_grid_active = False
                logging.info(f"[{self.symbol}][on_candle] ADX EXIT SIGNAL (ADX hist: {[f'{x:.2f}' for x in self.adx_history]}): Deactivating grid.")
                base_currency_key = symbol.split('-')[0].upper()
                crypto_to_sell = self.portfolio['crypto_holdings'].get(base_currency_key, Decimal('0'))
                if crypto_to_sell > Decimal('0'):
                    liquidation_order_id = self.next_order_id
                    self.next_order_id += 1
                    self.execute_order_func(symbol, 'sell', current_price_decimal, crypto_to_sell, order_id=liquidation_order_id)
                self.active_buy_orders, self.active_sell_orders, self.grid_levels = [], [], []
                self.pending_order_ids.clear()
        
        if self.is_grid_active:
            self._recalculate_dynamic_grid(current_price_decimal)
            orders_to_execute = self._manage_grid_orders()
            for order in orders_to_execute:
                sufficient_funds = False
                if order['type'] == 'buy':
                    required_cash = order['price'] * order['amount'] * (Decimal('1') + self.commission_rate)
                    if self.portfolio['cash'] >= required_cash:
                        sufficient_funds = True
                elif order['type'] == 'sell':
                    base_currency = order['symbol'].split('-')[0].upper()
                    if self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')) >= order['amount']:
                        sufficient_funds = True
                
                if sufficient_funds:
                    logging.info(f"[{self.symbol}][on_candle] Staging order {order['order_id']} for execution: {order['type']} {order['amount']:.8f} @ {order['price']:.2f}")
                    if order['type'] == 'buy': self.active_buy_orders.append(order)
                    else: self.active_sell_orders.append(order)
                    self.pending_order_ids.add(order['order_id'])
                    self.execute_order_func(order['symbol'], order['type'], order['price'], order['amount'], order['order_id'])

        if self.portfolio and self.current_price and self.current_price > 0:
            base_currency = self.symbol.split('-')[0].upper()
            holdings_value = self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')) * self.current_price
            total_equity = self.portfolio['cash'] + holdings_value
            self.equity_history.append((current_time_dt.isoformat(), total_equity))

        if not self.is_grid_active and (self.active_buy_orders or self.active_sell_orders):
            self.active_buy_orders, self.active_sell_orders, self.grid_levels = [], [], []
            self.pending_order_ids.clear()


    def notify_order_filled(self, filled_order_details):
        symbol = filled_order_details['symbol']
        order_type = filled_order_details['type']
        filled_price = filled_order_details['price']
        order_id = filled_order_details.get('order_id')

        if order_id is not None:
            if order_type == 'buy':
                self.active_buy_orders = [o for o in self.active_buy_orders if o.get('order_id') != order_id]
            elif order_type == 'sell':
                self.active_sell_orders = [o for o in self.active_sell_orders if o.get('order_id') != order_id]
        
        if order_id and order_id in self.pending_order_ids:
            self.pending_order_ids.remove(order_id)

        for level in self.grid_levels:
            if level['price'].compare(filled_price) == 0 and level['type'] == order_type:
                if level['status'] != 'filled':
                    level['status'] = 'filled'
                    logging.info(f"[{self.symbol}] Grid level marked filled for {order_type} @ {filled_price:.2f}.")
                break
        
        rearm_type = 'sell' if order_type == 'buy' else 'buy'
        potential_rearm_levels = [l for l in self.grid_levels if l['type'] == rearm_type and l['status'] == 'filled']
        if potential_rearm_levels:
            closest_level_to_rearm = min(potential_rearm_levels, key=lambda x: abs(x['price'] - filled_price))
            if closest_level_to_rearm:
                closest_level_to_rearm['status'] = 'open'
                logging.info(f"[{self.symbol}] Re-armed {rearm_type} level @ {closest_level_to_rearm['price']:.2f} after fill.")