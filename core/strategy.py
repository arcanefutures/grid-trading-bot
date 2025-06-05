import pandas as pd
import math
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
import pandas_ta as ta 
import logging
from datetime import datetime, timedelta, timezone

class GridStrategy:
    def __init__(self, params):
        self.symbol = params["symbol"] 
        self.config_params = params 

        self.grid_lower_bound = None 
        self.grid_upper_bound = None 
        self.grid_levels = [] 

        self.active_buy_orders = [] 
        self.active_sell_orders = [] 

        self.portfolio = None 
        self.execute_order_func = None 
        self.commission_rate = None 
        self.min_order_sizes = {} 
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
        self.adx_stable_readings_required = self.config_params.get("adx_stable_readings_required", 3) # Default to 3 if not in params

    def set_backtester_context(self, portfolio, execute_order_func, commission_rate, min_order_sizes_map, simulated_slippage_percentage):
        """
        Sets the necessary context for the strategy to operate.
        """
        self.portfolio = portfolio 
        self.execute_order_func = execute_order_func 
        self.commission_rate = Decimal(str(commission_rate)) 
        self.min_order_sizes = {k: Decimal(str(v)) for k, v in min_order_sizes_map.items()} 
        self.simulated_slippage_percentage = Decimal(str(simulated_slippage_percentage)) 


    def calculate_grid_levels(self, initial_price):
        """
        For dynamic grid, this method primarily prepares initial data if needed.
        Actual grid levels are calculated dynamically in _recalculate_dynamic_grid.
        Called once at the start of a backtest (or potentially live session).
        """
        # In a dynamic ATR grid, actual levels are computed in _recalculate_dynamic_grid.
        pass 


    def _calculate_amount_for_order(self, symbol_pair, price_decimal):
        """
        Calculates the base asset amount for an order based on a percentage of current equity,
        applies a maximum order size cap, and checks against exchange minimums.
        """
        logging.info(f"--- Entering _calculate_amount_for_order (in strategy.py) ---")
        logging.info(f"Attempting order for: {symbol_pair} at price: {price_decimal} (Type: {type(price_decimal)})")
        logging.info(f"Current self.portfolio state: {self.portfolio}")
        logging.info(f"Current self.current_price (used for equity calc): {self.current_price} (Type: {type(self.current_price)})")

        # --- CORRECTED BASE CURRENCY EXTRACTION ---
        base_currency_key = symbol_pair.split('-')[0].upper() # Use '-' as separator and ensure UPPERCASE like portfolio
        logging.info(f"[_calc_amount] Extracted base_currency_key for lookup: '{base_currency_key}'")
        # ---

        price_for_equity_calc = self.current_price
        if not isinstance(price_for_equity_calc, Decimal) or price_for_equity_calc is None or price_for_equity_calc <= Decimal('0'):
            logging.warning(f"[_calc_amount] self.current_price is invalid or not Decimal ({self.current_price}, Type: {type(self.current_price)}). Using order price {price_decimal} for equity calculation instead.")
            if not isinstance(price_decimal, Decimal) or price_decimal <= Decimal('0'):
                 logging.error(f"[_calc_amount] Critical: Both self.current_price and order price {price_decimal} are invalid for equity calculation. Returning 0.")
                 return Decimal('0')
            price_for_equity_calc = price_decimal

        holding_qty_from_portfolio = self.portfolio['crypto_holdings'].get(base_currency_key, Decimal('0'))
        if not isinstance(holding_qty_from_portfolio, Decimal):
            logging.warning(f"[_calc_amount] Holding quantity for {base_currency_key} is not Decimal: {holding_qty_from_portfolio} (Type: {type(holding_qty_from_portfolio)}). Attempting conversion.")
            try:
                holding_qty = Decimal(str(holding_qty_from_portfolio))
            except Exception as e:
                logging.error(f"[_calc_amount] Could not convert holding_qty '{holding_qty_from_portfolio}' to Decimal. Error: {e}. Returning 0.")
                return Decimal('0')
        else:
            holding_qty = holding_qty_from_portfolio
            
        logging.info(f"[_calc_amount] Pre-multiplication: holding_qty for '{base_currency_key}': {holding_qty} (Type: {type(holding_qty)})")
        logging.info(f"[_calc_amount] Pre-multiplication: price_for_equity_calc: {price_for_equity_calc} (Type: {type(price_for_equity_calc)})")

        crypto_value = holding_qty * price_for_equity_calc
        
        cash_value_from_portfolio = self.portfolio['cash']
        if not isinstance(cash_value_from_portfolio, Decimal):
            logging.warning(f"[_calc_amount] Cash value in portfolio is not Decimal: {cash_value_from_portfolio} (Type: {type(cash_value_from_portfolio)}). Attempting conversion.")
            try:
                cash_value = Decimal(str(cash_value_from_portfolio))
            except Exception as e:
                logging.error(f"[_calc_amount] Could not convert cash_value '{cash_value_from_portfolio}' to Decimal. Error: {e}. Crypto_value was {crypto_value}. Returning 0 for safety.")
                return Decimal('0')
        else:
            cash_value = cash_value_from_portfolio
                
        current_equity = cash_value + crypto_value

        logging.info(f"Calculated crypto_value for {base_currency_key}: {crypto_value} (Type: {type(crypto_value)})") # Changed base_currency to base_currency_key
        logging.info(f"Cash in portfolio: {cash_value} (Type: {type(cash_value)})")
        logging.info(f"Calculated current_equity: {current_equity} (Type: <class 'decimal.Decimal'>)") # Explicitly show Decimal type

        percentage_per_order = Decimal(str(self.config_params["percentage_of_equity_per_grid_order"]))
        target_usd_amount = current_equity * percentage_per_order

        logging.info(f"Percentage per order: {percentage_per_order}")
        logging.info(f"Calculated target_usd_amount (before cap): {target_usd_amount}")

        max_order_size_usd = Decimal(str(self.config_params.get("max_order_size_usd", "1000000.0"))) 
        if target_usd_amount > max_order_size_usd:
            target_usd_amount = max_order_size_usd
            logging.info(f"Target USD amount capped to: {target_usd_amount}")

        if target_usd_amount <= Decimal('0'):
            logging.info(f"Target USD amount is <= 0 ({target_usd_amount}). Returning 0.")
            return Decimal('0')

        if not isinstance(price_decimal, Decimal) or price_decimal <= Decimal('0'):
            logging.info(f"Order price_decimal is not Decimal or <= 0 ({price_decimal}, Type: {type(price_decimal)}). Returning 0.")
            return Decimal('0')

        base_asset_amount = (target_usd_amount / price_decimal).quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
        
        logging.info(f"Calculated base_asset_amount (before min check): {base_asset_amount}")
        
        min_size_from_config = self.min_order_sizes.get(base_currency_key, Decimal('0')) # Use base_currency_key
        if not isinstance(min_size_from_config, Decimal): 
            min_size = Decimal(str(min_size_from_config))
        else:
            min_size = min_size_from_config

        if base_asset_amount < min_size:
            logging.info(f"Calculated base_asset_amount {base_asset_amount} is less than min_size {min_size} for {base_currency_key}. Returning 0.") # Use base_currency_key
            return Decimal('0')
        
        logging.info(f"--- Exiting _calculate_amount_for_order, returning: {base_asset_amount} ---")
        return base_asset_amount


    def _place_initial_orders(self):
        """
        Handles initial portfolio balancing for 'neutral' grid setups,
        aiming for full deployment of allocated capital by buying a larger initial crypto proportion.
        """
        if self.current_price is None: 
            print(f"[{self.symbol}] Warning: current_price not set before _place_initial_orders. Cannot place initial orders.") 
            return 

        if self.config_params["initial_position_type"] == "neutral" and self.is_grid_active: 
            max_allocated_capital_for_strategy = Decimal(str(self.config_params["max_capital_allocation_percentage"])) * self.portfolio['cash'] 
            initial_crypto_proportion = Decimal('0.90') 
            target_crypto_usd_value_initial = max_allocated_capital_for_strategy * initial_crypto_proportion 
            base_currency = self.symbol.split('-')[0].upper() 
            current_crypto_usd_value = self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')) * self.current_price 
            amount_to_buy_usd = target_crypto_usd_value_initial - current_crypto_usd_value 

            if amount_to_buy_usd > Decimal('0'): 
                initial_buy_amount = (amount_to_buy_usd / self.current_price).quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP) 
                min_size = self.min_order_sizes.get(base_currency, Decimal('0')) 
                if initial_buy_amount < min_size: 
                    initial_buy_amount = min_size 
                    print(f"[{self.symbol}] Adjusting initial buy to minimum size: {initial_buy_amount:.8f} {base_currency}") 

                required_cash = self.current_price * initial_buy_amount * (Decimal('1') + self.commission_rate) 
                if self.portfolio['cash'] >= required_cash: 
                    if self.execute_order_func(self.symbol, 'buy', self.current_price, initial_buy_amount): 
                        print(f"[{self.symbol}] Executing initial BUY to balance portfolio (full deployment): {initial_buy_amount:.8f} {base_currency} @ {self.current_price:.2f}") 
                    else: 
                        print(f"[{self.symbol}] Initial balancing BUY FAILED unexpectedly (funds checked, but _execute_order_func reported failure).") 
                else: 
                    print(f"[{self.symbol}] Warning: Not enough cash ({self.portfolio['cash']:.2f}) to execute initial balancing buy ({required_cash:.2f}). Starting grid with cash bias.") 
            else: 
                print(f"[{self.symbol}] No initial balancing buy needed. Current crypto value: {current_crypto_usd_value:.2f} USD.") 
        pass 


    def _recalculate_dynamic_grid(self, current_price_decimal):
        """
        Calculates ATR and dynamically adjusts grid levels based on current market conditions.
        """
        if not isinstance(current_price_decimal, Decimal): # Ensure input is Decimal
            logging.error(f"[_recalculate_dynamic_grid] current_price_decimal is not Decimal: {current_price_decimal} (Type: {type(current_price_decimal)})")
            # Decide how to handle this: return, or try to convert, or use self.current_price as fallback
            if self.current_price and isinstance(self.current_price, Decimal):
                current_price_decimal = self.current_price
                logging.warning(f"[_recalculate_dynamic_grid] Using self.current_price as fallback: {current_price_decimal}")
            else: # Cannot proceed without a valid Decimal price
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
        self.grid_lower_bound = current_price_decimal - (total_grid_width / Decimal('2')) 
        self.grid_upper_bound = current_price_decimal + (total_grid_width / Decimal('2')) 

        if self.grid_lower_bound <= Decimal('0'): 
            self.grid_lower_bound = Decimal('0.000001') 

        num_grids_param = self.config_params.get("number_of_grids", 10) # Default if not present
        num_grids = Decimal(str(num_grids_param))


        if num_grids <= Decimal('0'): 
            self.grid_levels = [] 
            return 

        grid_step_size = max(total_grid_width / num_grids, min_profitable_price_movement) 

        if grid_step_size <= Decimal('0'): 
            self.grid_levels = [] 
            return 

        self.grid_levels = [] 
        current_grid_price_buy = current_price_decimal - grid_step_size / Decimal('2') 
        num_buy_levels = int(num_grids / Decimal('2')) 
        num_sell_levels = int(num_grids / Decimal('2')) 
        
        for i in range(num_buy_levels): 
            price = current_grid_price_buy - (grid_step_size * Decimal(str(i))) 
            price = price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP) # Assuming 2 decimal places for price levels
            if price > Decimal('0') and price >= self.grid_lower_bound : # Ensure price > 0
                self.grid_levels.append({'price': price, 'type': 'buy', 'status': 'open'}) 
            else: 
                break 

        current_grid_price_sell = current_price_decimal + grid_step_size / Decimal('2') 
        for i in range(num_sell_levels): 
            price = current_grid_price_sell + (grid_step_size * Decimal(str(i))) 
            price = price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP) 
            if price <= self.grid_upper_bound: 
                self.grid_levels.append({'price': price, 'type': 'sell', 'status': 'open'}) 
            else: 
                break 

        self.grid_levels.sort(key=lambda x: x['price']) 
        # logging.info(f"[{self.symbol}] Dynamic Grid Re-calculated: {len(self.grid_levels)} levels. ATR: {current_atr:.2f}, Range: {self.grid_lower_bound:.2f} - {self.grid_upper_bound:.2f}")


    # Inside core/strategy.py - GridStrategy class

    def on_candle(self, symbol, candle_data_series, current_price_decimal, current_time_dt):
        """
        Called for each new candle. Main loop for strategy decisions.
        candle_data_series: Pandas Series with 'open', 'high', 'low', 'close', 'volume' as Decimals
        current_price_decimal: Close price of the current candle as Decimal
        current_time_dt: Timestamp of the current candle as datetime object
        """
        # Input validation
        if not isinstance(current_price_decimal, Decimal):
            logging.error(f"[{self.symbol}][on_candle] FATAL: current_price_decimal is not Decimal: {current_price_decimal} (Type: {type(current_price_decimal)})")
            return 
        if not isinstance(current_time_dt, datetime):
            logging.error(f"[{self.symbol}][on_candle] FATAL: current_time_dt is not datetime: {current_time_dt} (Type: {type(current_time_dt)})")
            return
        if not isinstance(candle_data_series, pd.Series):
            logging.error(f"[{self.symbol}][on_candle] FATAL: candle_data_series is not a Pandas Series. Type: {type(candle_data_series)}")
            return
        expected_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not expected_cols.issubset(candle_data_series.index):
            logging.error(f"[{self.symbol}][on_candle] FATAL: candle_data_series missing one or more expected columns: {expected_cols - set(candle_data_series.index)}")
            return

        self.current_price = current_price_decimal 
        
        # Append current candle data to history.
        # Ensure incoming candle_data_series values are already Decimals.
        new_row_df = pd.DataFrame([candle_data_series.to_dict()], index=[current_time_dt])
        
        # Before concatenating, ensure new_row_df columns match ohlcv_history columns if it's not empty
        if not self.ohlcv_history.empty:
            new_row_df = new_row_df.reindex(columns=self.ohlcv_history.columns) # Align columns
            for col in new_row_df.columns: # Ensure new row data is Decimal
                 if col in candle_data_series:
                    new_row_df.loc[current_time_dt, col] = Decimal(str(candle_data_series[col]))


        self.ohlcv_history = pd.concat([self.ohlcv_history, new_row_df]) 

        # Manage history length
        required_history_length = max(self.atr_period + 10, self.adx_period * 2 + self.adx_confirmation_periods + 10) # Ensure ample buffer
        if len(self.ohlcv_history) > required_history_length:
            self.ohlcv_history = self.ohlcv_history.iloc[-required_history_length:] 
        
        logging.debug(f"[{self.symbol}][on_candle] OHLCV history length: {len(self.ohlcv_history)}")

        # --- ADX Market Condition Filtering Logic ---
        # Minimum length for ADX calculation by pandas_ta is typically adx_period + (adx_period - 1) + 1 = 2*adx_period
        # And we need enough for confirmation periods too.
        min_len_for_adx_calc_and_confirm = (self.adx_period * 2 -1) + self.adx_confirmation_periods 
        # A more common rule of thumb for TA-Lib ADX is at least 2*period candles for first ADX value
        
        current_adx_float = 0.0 # Default if calculation fails or not enough data

        if len(self.ohlcv_history) >= min_len_for_adx_calc_and_confirm : 
            try:
                # Ensure data passed to ta.adx is float as it expects
                high_float = self.ohlcv_history['high'].astype(float)
                low_float = self.ohlcv_history['low'].astype(float)
                close_float = self.ohlcv_history['close'].astype(float)

                adx_df = ta.adx(high=high_float, low=low_float, close=close_float, length=self.adx_period)
                
                if adx_df is not None and not adx_df.empty and f'ADX_{self.adx_period}' in adx_df.columns:
                    current_adx_value_from_pandas_ta = adx_df[f'ADX_{self.adx_period}'].iloc[-1]
                    
                    if pd.isna(current_adx_value_from_pandas_ta):
                        self.valid_adx_readings_count = 0 # Reset if NaN occurs
                        logging.warning(f"[{self.symbol}][on_candle] ADX calculation resulted in NaN, using 0.0. History length: {len(self.ohlcv_history)}")
                    else:
                        current_adx_float = float(current_adx_value_from_pandas_ta)
                        self.valid_adx_readings_count += 1
                else:
                    self.valid_adx_readings_count = 0 # Reset if ADX calculation fails
                    logging.warning(f"[{self.symbol}][on_candle] ADX calculation failed or returned empty/unexpected df, using 0.0. History length: {len(self.ohlcv_history)}")
            
            except Exception as e:
                self.valid_adx_readings_count = 0 # Reset on error
                logging.error(f"[{self.symbol}][on_candle] Error calculating ADX: {e}", exc_info=True)
                # current_adx_float remains 0.0
        else:
            self.valid_adx_readings_count = 0 # Not enough history, so no valid reading yet
            logging.info(f"[{self.symbol}][on_candle] Not enough OHLCV history for ADX calculation. Length: {len(self.ohlcv_history)}, Need: {min_len_for_adx_calc_and_confirm}")
            # current_adx_float remains 0.0

        self.adx_history.append(current_adx_float) 
        self.adx_history = self.adx_history[-self.adx_confirmation_periods:] 
        
        adx_sufficient_history_for_confirmation = len(self.adx_history) >= self.adx_confirmation_periods
        adx_warmup_complete = self.valid_adx_readings_count >= self.adx_stable_readings_required

        # Grid Activation/Deactivation Logic
        if not self.is_grid_active: 
            if adx_sufficient_history_for_confirmation and \
               adx_warmup_complete and \
               all(adx_val < self.adx_entry_threshold for adx_val in self.adx_history): 
                
                self.is_grid_active = True 
                logging.info(f"[{self.symbol}][on_candle] ADX ENTRY SIGNAL (ADX hist: {[f'{x:.2f}' for x in self.adx_history]}, "
                             f"Current ADX calc: {current_adx_float:.2f}, StableReadings: {self.valid_adx_readings_count}/{self.adx_stable_readings_required}): "
                             f"Activating grid at {current_time_dt}")
                # _place_initial_orders() is NOT called here anymore.
                # _manage_grid_orders (called below if grid active) will handle initial placement.
            elif adx_sufficient_history_for_confirmation and adx_warmup_complete : # Log if warmup met but ADX too high
                 logging.info(f"[{self.symbol}][on_candle] ADX WARMUP MET but ADX condition not met for entry. ADX hist: {[f'{x:.2f}' for x in self.adx_history]}")
            elif not adx_warmup_complete and len(self.ohlcv_history) >= min_len_for_adx_calc_and_confirm:
                 logging.info(f"[{self.symbol}][on_candle] ADX calculated ({current_adx_float:.2f}) but warmup not yet complete ({self.valid_adx_readings_count}/{self.adx_stable_readings_required}).")


        elif self.is_grid_active: # Grid is currently active, check for deactivation
            if adx_sufficient_history_for_confirmation and \
               all(adx_val > self.adx_exit_threshold for adx_val in self.adx_history): 
                
                self.is_grid_active = False 
                logging.info(f"[{self.symbol}][on_candle] ADX EXIT SIGNAL (ADX hist: {[f'{x:.2f}' for x in self.adx_history]}, "
                             f"Current ADX calc: {current_adx_float:.2f}): Deactivating grid and liquidating at {current_time_dt}") 
                base_currency_key = symbol.split('-')[0].upper() 
                crypto_to_sell = self.portfolio['crypto_holdings'].get(base_currency_key, Decimal('0')) 
                
                if crypto_to_sell > Decimal('0'): 
                    if self.execute_order_func(symbol, 'sell', current_price_decimal, crypto_to_sell):
                         logging.info(f"[{self.symbol}][on_candle] Liquidated {crypto_to_sell:.8f} {base_currency_key} due to ADX exit.")
                    else:
                         logging.warning(f"[{self.symbol}][on_candle] Liquidation SELL order FAILED for {crypto_to_sell:.8f} {base_currency_key} on ADX exit.")
                else: 
                    logging.info(f"[{self.symbol}][on_candle] No crypto holdings to liquidate on ADX exit.") 
                
                # Clear active orders and grid levels upon deactivation
                if self.active_buy_orders or self.active_sell_orders:
                    logging.info(f"[{self.symbol}][on_candle] Cancelling {len(self.active_buy_orders)} buy orders and {len(self.active_sell_orders)} sell orders on ADX exit.")
                self.active_buy_orders = [] 
                self.active_sell_orders = [] 
                self.grid_levels = [] 
        
        # --- Grid Management (ONLY if grid is now active) ---
        if self.is_grid_active: 
            self._recalculate_dynamic_grid(current_price_decimal) 
            self._manage_grid_orders() 
        else: 
            # Ensure orders are cleared if grid became inactive NOT due to ADX exit (e.g., manual override in future)
            # The ADX exit logic already clears them. This is a safeguard.
            if self.active_buy_orders or self.active_sell_orders: 
                logging.info(f"[{self.symbol}][on_candle] Grid is inactive, ensuring active orders are cleared. Buys: {len(self.active_buy_orders)}, Sells: {len(self.active_sell_orders)}")
                self.active_buy_orders = [] 
                self.active_sell_orders = [] 
            # self.grid_levels = [] # Grid levels are also cleared by ADX exit logic

    def _manage_grid_orders(self): 
        """
        Manages active grid orders based on current price and grid levels.
        """
        active_order_identifiers = set() 
        temp_active_buy_orders = [] 
        for order in self.active_buy_orders: 
            current_level_found = False 
            for level_in_grid in self.grid_levels: 
                if level_in_grid['price'].compare(order['price']) == 0 and level_in_grid['type'] == order['type']: 
                    if level_in_grid['status'] != 'filled': 
                        temp_active_buy_orders.append(order) 
                        active_order_identifiers.add((order['price'], order['type'])) 
                    current_level_found = True 
                    break 
        temp_active_sell_orders = [] 
        for order in self.active_sell_orders: 
            current_level_found = False 
            for level_in_grid in self.grid_levels: 
                if level_in_grid['price'].compare(order['price']) == 0 and level_in_grid['type'] == order['type']: 
                    if level_in_grid['status'] != 'filled': 
                        temp_active_sell_orders.append(order) 
                        active_order_identifiers.add((order['price'], order['type'])) 
                    current_level_found = True 
                    break 
        self.active_buy_orders = temp_active_buy_orders 
        self.active_sell_orders = temp_active_sell_orders 

        potential_buy_levels = [] 
        for i, level in enumerate(self.grid_levels): 
            if level['type'] == 'buy' and level['price'] < self.current_price and level['status'] == 'open': 
                potential_buy_levels.append({'level': level, 'idx': i}) 
        potential_buy_levels.sort(key=lambda x: x['level']['price'], reverse=True) 

        for item in potential_buy_levels: 
            level = item['level'] 
            level_idx = item['idx'] 
            if (level['price'], level['type']) in active_order_identifiers: 
                continue 
            if len(self.active_buy_orders) >= self.config_params["max_concurrent_orders_per_side"]: 
                break 
            amount_to_trade = self._calculate_amount_for_order(self.symbol, level['price']) 
            if amount_to_trade == Decimal('0'): continue 
            required_cash = level['price'] * amount_to_trade * (Decimal('1') + self.commission_rate) 
            if self.portfolio['cash'] >= required_cash: 
                # The execute_order_func will be our paper trader, which returns True/False
                # It also logs the execution attempt and result.
                if self.execute_order_func(self.symbol, 'buy', level['price'], amount_to_trade): 
                    new_order = { 
                        'symbol': self.symbol, 
                        'type': 'buy', 
                        'price': level['price'], 
                        'amount': amount_to_trade, 
                        'grid_level_idx': level_idx 
                    }
                    self.active_buy_orders.append(new_order) 
                    self.grid_levels[level_idx]['status'] = 'pending' # Mark as pending until fill confirmed by notify_order_filled
                    logging.info(f"[{self.symbol}] Managing: Sent paper BUY order for {amount_to_trade:.8f} @ {level['price']:.2f}") 
            else:
                logging.warning(f"[{self.symbol}] Managing: Insufficient cash for BUY {amount_to_trade:.8f} @ {level['price']:.2f}. Need {required_cash:.2f}, Have {self.portfolio['cash']:.2f}")


        potential_sell_levels = [] 
        for i, level in enumerate(self.grid_levels): 
            if level['type'] == 'sell' and level['price'] > self.current_price and level['status'] == 'open': 
                potential_sell_levels.append({'level': level, 'idx': i}) 
        potential_sell_levels.sort(key=lambda x: x['level']['price']) 

        for item in potential_sell_levels: 
            level = item['level'] 
            level_idx = item['idx'] 
            if (level['price'], level['type']) in active_order_identifiers: 
                continue 
            if len(self.active_sell_orders) >= self.config_params["max_concurrent_orders_per_side"]: 
                break 
            amount_to_trade = self._calculate_amount_for_order(self.symbol, level['price']) 
            if amount_to_trade == Decimal('0'): continue 
            base_currency = self.symbol.split('-')[0].upper() 
            if self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')) >= amount_to_trade: 
                if self.execute_order_func(self.symbol, 'sell', level['price'], amount_to_trade): 
                    new_order = { 
                        'symbol': self.symbol, 
                        'type': 'sell', 
                        'price': level['price'], 
                        'amount': amount_to_trade, 
                        'grid_level_idx': level_idx 
                    }
                    self.active_sell_orders.append(new_order) 
                    self.grid_levels[level_idx]['status'] = 'pending' 
                    logging.info(f"[{self.symbol}] Managing: Sent paper SELL order for {amount_to_trade:.8f} @ {level['price']:.2f}")
            else:
                logging.warning(f"[{self.symbol}] Managing: Insufficient crypto for SELL {amount_to_trade:.8f} @ {level['price']:.2f}. Need {amount_to_trade:.8f}, Have {self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')):.8f}")


    def notify_order_filled(self, filled_order_details):
        """
        Called by the execution environment (backtester or live/paper trader)
        when an order is confirmed filled.
        filled_order_details should be a dictionary like the ones in active_buy/sell_orders
        or compatible with it, especially 'symbol', 'type', 'price'.
        """
        symbol = filled_order_details['symbol']
        order_type = filled_order_details['type']
        # The price in filled_order_details should be the actual fill price
        # For paper trading, execute_paper_trade uses the requested price for now.
        filled_price = filled_order_details['price'] 
        if not isinstance(filled_price, Decimal): # Ensure filled_price is Decimal
            filled_price = Decimal(str(filled_price))

        # Remove the filled order from active lists
        if order_type == 'buy':
            self.active_buy_orders = [o for o in self.active_buy_orders if not (o['price'].compare(filled_price) == 0 and o['amount'].compare(filled_order_details['amount'])==0) ] # Simple removal
        elif order_type == 'sell':
            self.active_sell_orders = [o for o in self.active_sell_orders if not (o['price'].compare(filled_price) == 0 and o['amount'].compare(filled_order_details['amount'])==0) ]


        # Find the current index of the filled_price in the current grid_levels
        # This logic is sensitive to grid shifts if they happen between order placement and fill notification.
        current_level_idx = -1
        for i, level in enumerate(self.grid_levels): 
            if level['price'].compare(filled_price) == 0 and level['type'] == order_type: 
                current_level_idx = i 
                break 

        if current_level_idx != -1: 
            self.grid_levels[current_level_idx]['status'] = 'filled' 
            logging.info(f"[{self.symbol}] Notified fill for {order_type} @ {filled_price:.2f}. Grid level {current_level_idx} marked filled.")

            # Re-arm the corresponding opposite order on the DYNAMIC grid.
            if order_type == 'buy': 
                corresponding_sell_level_found = False 
                for i in range(current_level_idx + 1, len(self.grid_levels)): 
                    level_to_rearm = self.grid_levels[i] 
                    if level_to_rearm['type'] == 'sell': 
                        if level_to_rearm['status'] == 'filled': # Only re-open if it was previously part of a completed grid cycle
                            level_to_rearm['status'] = 'open' 
                            logging.info(f"[{self.symbol}] Re-arming SELL level @ {level_to_rearm['price']:.2f} (index {i}) after BUY fill.")
                            corresponding_sell_level_found = True 
                            break 
                if not corresponding_sell_level_found: 
                    logging.debug(f"[{self.symbol}] No corresponding sell level to re-arm or found one not in 'filled' state after BUY fill @ {filled_price:.2f}.")
            elif order_type == 'sell': 
                corresponding_buy_level_found = False 
                for i in range(current_level_idx - 1, -1, -1): 
                    level_to_rearm = self.grid_levels[i] 
                    if level_to_rearm['type'] == 'buy': 
                        if level_to_rearm['status'] == 'filled':
                            level_to_rearm['status'] = 'open' 
                            logging.info(f"[{self.symbol}] Re-arming BUY level @ {level_to_rearm['price']:.2f} (index {i}) after SELL fill.")
                            corresponding_buy_level_found = True 
                            break 
                if not corresponding_buy_level_found: 
                    logging.debug(f"[{self.symbol}] No corresponding buy level to re-arm or found one not in 'filled' state after SELL fill @ {filled_price:.2f}.")
        else: 
            logging.warning(f"[{self.symbol}] Could not find matching grid level for filled {order_type} order @ {filled_price:.2f}. Grid might have shifted or order was not from current grid_levels.")
        
        # After a fill, it's good to call _manage_grid_orders again to reassess and place new orders if needed
        self._manage_grid_orders()


# Example usage (for testing strategy.py independently - not part of actual bot run)
# Remove or comment out this __main__ block if you are importing GridStrategy elsewhere.
if __name__ == "__main__":
    print("Running strategy.py standalone test section (should be commented out for actual bot use)...")
    
    # Ensure these are Decimal for mock data
    mock_initial_capital = Decimal('1000.0')
    mock_commission_rate = Decimal('0.004')
    mock_sim_slippage = Decimal('0.0005')

    mock_min_order_sizes_data = { 
        'BTC': Decimal('0.0001'), 'ETH': Decimal('0.005'), 'SOL': Decimal('0.1'), 
        'ADA': Decimal('1.0'), 'DOT': Decimal('1.0'), 'AVAX': Decimal('0.1'), 
        'ATOM': Decimal('1.0'), 'LINK': Decimal('1.0'), 'XRP': Decimal('10.0')
    }

    # Get parameters from config (or define a minimal set for testing)
    try:
        from config.parameters import GRID_STRATEGY_PARAMS
        test_params = GRID_STRATEGY_PARAMS.copy()
        test_params["symbol"] = "ADA-USD" # Example symbol
        # Override specific params for testing if needed
        test_params["adx_entry_threshold"] = 20 # Lower for easier activation in mock
        test_params["adx_exit_threshold"] = 35
        test_params["percentage_of_equity_per_grid_order"] = 0.05 
        test_params["number_of_grids"] = 6 # Fewer grids for simpler test
        test_params["grid_atr_multiplier"] = Decimal("2.0") # Smaller ATR multiplier
    except ImportError:
        print("Warning: config.parameters not found, using hardcoded test_params for strategy.py standalone test.")
        test_params = { # Minimal params for testing
            "symbol": "ADA-USD", "atr_period": 14, "grid_atr_multiplier": Decimal("2.0"),
            "number_of_grids": 6, "max_capital_allocation_percentage": 1.0,
            "percentage_of_equity_per_grid_order": 0.05, "max_concurrent_orders_per_side": 3,
            "profit_per_grid_percentage": 0.01, "initial_position_type": "neutral",
            "adx_period": 14, "adx_entry_threshold": 20, "adx_exit_threshold": 35,
            "adx_confirmation_periods": 2
        }


    mock_portfolio_state = { 
        'cash': mock_initial_capital,
        'crypto_holdings': {'ADA': Decimal('0')}, # Start with no ADA for this test
    }

    def mock_execute_order_function(symbol, order_type, price, amount):
        # Simplified mock execution for standalone test
        print(f"[MOCK EXEC] Attempt: {order_type} {amount} {symbol.split('_')[0]} @ {price}")
        # Simulate success for testing purposes
        return True 

    strategy_instance = GridStrategy(test_params)
    strategy_instance.set_backtester_context(
        mock_portfolio_state, 
        mock_execute_order_function, 
        mock_commission_rate, 
        mock_min_order_sizes_data, 
        mock_sim_slippage
    )

    # Simulate some candle data for ADA-USD
    # Prices around 0.65, ATR around 0.01, ADX to trigger entry
    print("\n--- Simulating candles with ADX logic for strategy.py standalone test ---")
    base_price = Decimal('0.65')
    sim_current_time = datetime.now(timezone.utc)

    # Generate enough dummy OHLCV data to satisfy ADX calculation + some grid action
    ohlc_data_for_test = []
    # Initial candles with low volatility (low ADX) to trigger entry
    for i in range(test_params["adx_period"] + test_params["adx_confirmation_periods"] + 5):
        offset = Decimal(str(math.sin(i * 0.05) * 0.001)) # Very low volatility
        price = base_price + offset
        sim_current_time += timedelta(minutes=1)
        ohlc_data_for_test.append({
            'open': price, 'high': price + Decimal('0.0001'), 
            'low': price - Decimal('0.0001'), 'close': price, 'volume': Decimal('1000')
        })
    
    # Candles with slightly more movement to interact with grid
    for i in range(10):
        offset = Decimal(str(math.sin(i * 0.2) * 0.005)) # More distinct movement
        price = base_price + offset + (Decimal(str(i)) * Decimal('0.0005')) # Gradual rise
        sim_current_time += timedelta(minutes=1)
        ohlc_data_for_test.append({
            'open': price, 'high': price + Decimal('0.002'), 
            'low': price - Decimal('0.002'), 'close': price, 'volume': Decimal('2000')
        })

    print(f"Simulating {len(ohlc_data_for_test)} candles...")
    for i, c_data_dict in enumerate(ohlc_data_for_test):
        candle_series = pd.Series(c_data_dict)
        close_price_dec = c_data_dict['close']
        candle_time_dt = sim_current_time - timedelta(minutes=(len(ohlc_data_for_test) - 1 - i)) # Ensure time progresses

        print(f"\n--- Candle {i+1} --- Time: {candle_time_dt}, Close: {close_price_dec:.4f}")
        strategy_instance.on_candle("ADA-USD", candle_series, close_price_dec, candle_time_dt)
        if strategy_instance.is_grid_active:
            print(f"Grid Active. Grid Levels: {len(strategy_instance.grid_levels)}")
            # for level in strategy_instance.grid_levels: print(level) # Uncomment for detailed grid levels
        else:
            print("Grid Inactive.")
        print(f"Portfolio: {mock_portfolio_state}")
        print(f"Active Buys: {len(strategy_instance.active_buy_orders)}, Active Sells: {len(strategy_instance.active_sell_orders)}")

    print("\n--- Final state after standalone test ---")
    print(f"Final Portfolio: {mock_portfolio_state}")
    print(f"Final Active Buy Orders: {strategy_instance.active_buy_orders}")
    print(f"Final Active Sell Orders: {strategy_instance.active_sell_orders}")