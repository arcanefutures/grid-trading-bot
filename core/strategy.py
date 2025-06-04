import pandas as pd
import math
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
import pandas_ta as ta # Import pandas_ta for ATR and ADX calculation

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
        self.grid_atr_multiplier = Decimal(str(params.get("grid_atr_multiplier", 3.0)))
        self.ohlcv_history = pd.DataFrame()

        # --- NEW ADX Parameters and State ---
        self.is_grid_active = False # Initial state: grid is not active
        self.adx_period = params.get("adx_period", 14)
        self.adx_entry_threshold = params.get("adx_entry_threshold", 25.0)
        self.adx_exit_threshold = params.get("adx_exit_threshold", 30.0)
        self.adx_confirmation_periods = params.get("adx_confirmation_periods", 3)
        self.adx_history = [] # To store recent ADX values for consecutive checks

    def set_backtester_context(self, portfolio, execute_order_func, commission_rate, min_order_sizes_map, simulated_slippage_percentage):
        """
        Sets the necessary context from the backtester for the strategy to operate.
        """
        self.portfolio = portfolio
        self.execute_order_func = execute_order_func
        self.commission_rate = Decimal(str(commission_rate))
        self.min_order_sizes = {k: Decimal(str(v)) for k, v in min_order_sizes_map.items()}
        self.simulated_slippage_percentage = Decimal(str(simulated_slippage_percentage))


    def calculate_grid_levels(self, initial_price):
        """
        For dynamic grid, this method will primarily prepare initial data if needed.
        The actual grid levels will be calculated dynamically on each candle in _recalculate_dynamic_grid.
        This method is called once at the start of the backtest.
        """
        # In a dynamic ATR grid, the grid levels are not calculated here.
        # They are computed in _recalculate_dynamic_grid which is called by on_candle.
        pass


    def _calculate_amount_for_order(self, symbol_pair, price_decimal):
        """
        Calculates the base asset amount for an order based on a percentage of current equity,
        applies a maximum order size cap, and checks against exchange minimums.
        """
        # Determine total current equity
        base_currency = symbol_pair.split('_')[0]
        current_equity = self.portfolio['cash'] + self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')) * self.current_price

        # Calculate target USD amount based on percentage of current equity
        percentage_per_order = Decimal(str(self.config_params["percentage_of_equity_per_grid_order"]))
        target_usd_amount = current_equity * percentage_per_order

        # Apply the maximum order size cap
        max_order_size_usd = Decimal(str(self.config_params.get("max_order_size_usd", 1000000.0)))
        if target_usd_amount > max_order_size_usd:
            target_usd_amount = max_order_size_usd
            # print(f"[{self.symbol}] Capping order size to max: {max_order_size_usd:,.2f} USD") # Uncomment for debugging

        # Ensure target_usd_amount is not too small (e.g., if equity is very low)
        if target_usd_amount <= Decimal('0'):
            return Decimal('0')

        if price_decimal <= Decimal('0'):
            return Decimal('0')

        base_asset_amount = (target_usd_amount / price_decimal).quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)

        min_size = self.min_order_sizes.get(base_currency, Decimal('0'))

        if base_asset_amount < min_size:
            return Decimal('0')

        return base_asset_amount

    def _place_initial_orders(self):
        """
        Handles initial portfolio balancing for 'neutral' grid setups,
        aiming for full deployment of allocated capital by buying a larger initial crypto proportion.
        This is only executed if grid is active from the start (which it won't be with ADX filter).
        """
        # With ADX filter, the grid won't be active initially, so initial orders will be handled
        # only once ADX conditions are met. This method might become redundant or need adjustment.
        # For now, it will only run if self.is_grid_active is True at the start, which it won't be.
        if self.current_price is None:
            print(f"[{self.symbol}] Warning: current_price not set before _place_initial_orders. Cannot place initial orders.")
            return

        # We defer initial order placement until ADX confirms entry.
        # The logic below only runs if the strategy *starts* active, which it won't with ADX.
        # This part will effectively be skipped until ADX signals entry.
        if self.config_params["initial_position_type"] == "neutral" and self.is_grid_active:
            max_allocated_capital_for_strategy = Decimal(str(self.config_params["max_capital_allocation_percentage"])) * self.portfolio['cash']

            initial_crypto_proportion = Decimal('0.90')

            target_crypto_usd_value_initial = max_allocated_capital_for_strategy * initial_crypto_proportion

            base_currency = self.symbol.split('_')[0]
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


        # Initial grid orders are now managed by _manage_grid_orders, called by on_candle for the first time.
        pass


    def _recalculate_dynamic_grid(self, current_price_decimal):
        """
        Calculates ATR and dynamically adjusts grid levels based on current market conditions.
        This method replaces the static grid calculation.
        """
        if len(self.ohlcv_history) < self.atr_period:
            self.grid_levels = [] # Clear grid if not enough data for ATR
            return

        # Ensure OHLCV columns are numeric for pandas_ta, and then convert to float
        # pandas_ta requires floats for calculation, convert back to Decimal after.
        atr_series = ta.atr(self.ohlcv_history['high'].astype(float),
                             self.ohlcv_history['low'].astype(float),
                             self.ohlcv_history['close'].astype(float),
                             length=self.atr_period)

        if atr_series.iloc[-1] is None or pd.isna(atr_series.iloc[-1]):
            # This can happen if the last ATR value is NaN, which means not enough valid data points.
            self.grid_levels = []
            return

        current_atr = Decimal(str(atr_series.iloc[-1]))
        # Calculate the minimum price step required to cover commissions and target profit
        # Two commissions (buy + sell) + target profit on that base.
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

        num_grids = Decimal(str(self.config_params["number_of_grids"]))

        if num_grids <= Decimal('0'):
            self.grid_levels = []
            return

        grid_step_size = max(total_grid_width / num_grids, min_profitable_price_movement)

        if grid_step_size <= Decimal('0'):
            self.grid_levels = []
            return

        self.grid_levels = []

        # Generate buy orders below current price
        # Start from half a step below current price
        current_grid_price = current_price_decimal - grid_step_size / Decimal('2')

        # Calculate number of buy/sell levels from total num_grids (rounded down for integer division)
        # to ensure roughly equal distribution
        num_buy_levels = int(num_grids / Decimal('2'))
        num_sell_levels = int(num_grids / Decimal('2'))
        # The current approach already implicitly handles it by taking int() of num_grids/2
        # on both sides, possibly resulting in one less level than num_grids if odd.

        for i in range(num_buy_levels):
            price = current_grid_price - (grid_step_size * Decimal(str(i)))
            price = price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            if price >= self.grid_lower_bound:
                self.grid_levels.append({'price': price, 'type': 'buy', 'status': 'open'})
            else:
                break

        # Generate sell orders above current price
        # Start from half a step above current price
        current_grid_price = current_price_decimal + grid_step_size / Decimal('2')

        for i in range(num_sell_levels):
            price = current_grid_price + (grid_step_size * Decimal(str(i)))
            price = price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            if price <= self.grid_upper_bound:
                self.grid_levels.append({'price': price, 'type': 'sell', 'status': 'open'})
            else:
                break

        self.grid_levels.sort(key=lambda x: x['price'])

        # print(f"[{self.symbol}] Dynamic Grid Re-calculated: {len(self.grid_levels)} levels. ATR: {current_atr:.2f}, Range: {self.grid_lower_bound:.2f} - {self.grid_upper_bound:.2f}")


    def on_candle(self, symbol, candle_data, current_price_decimal, current_time):
        """
        Called by the Backtester for each new candle. This is the main loop for strategy decisions.
        """
        self.current_price = current_price_decimal

        # Append current candle data to history for ATR and ADX calculation
        candle_data_decimal = candle_data.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            candle_data_decimal[col] = Decimal(str(candle_data[col]))

        new_row_df = pd.DataFrame([candle_data_decimal], index=[current_time])
        self.ohlcv_history = pd.concat([self.ohlcv_history, new_row_df])

        # Keep only enough history for calculations (ATR + ADX periods)
        # Ensure it's enough for both ATR (atr_period + 5) and ADX (adx_period + adx_confirmation_periods + buffer)
        required_history_length = max(self.atr_period + 5, self.adx_period + self.adx_confirmation_periods + 5)
        self.ohlcv_history = self.ohlcv_history.iloc[-required_history_length:]

        # --- ADX Market Condition Filtering Logic ---
        # 1. Calculate ADX
        # Need enough data for ADX calculation (ADX period + buffer, usually ADX period * 2 or more candles are good)
        if len(self.ohlcv_history) > self.adx_period + self.adx_confirmation_periods + 1:
            adx_series = ta.adx(self.ohlcv_history['high'].astype(float),
                                self.ohlcv_history['low'].astype(float),
                                self.ohlcv_history['close'].astype(float),
                                length=self.adx_period).iloc[:, 0] # Select just the ADX column, not +DI/-DI
            current_adx = adx_series.iloc[-1]

            if pd.isna(current_adx):
                current_adx = 0.0 # Treat NaN ADX (not enough data or bad data) as non-trending initially

            self.adx_history.append(float(current_adx)) # Store as float for consistency with thresholds
            # Keep history short for consecutive checks
            self.adx_history = self.adx_history[-self.adx_confirmation_periods:]

            # 2. Check ADX conditions for grid activation/deactivation
            adx_sufficient_history = len(self.adx_history) >= self.adx_confirmation_periods

            if not self.is_grid_active:
                # Attempt to activate grid if ADX is below entry threshold for confirmation periods
                if adx_sufficient_history and all(adx_val < self.adx_entry_threshold for adx_val in self.adx_history):
                    self.is_grid_active = True
                    print(f"[{self.symbol}] ADX ENTRY SIGNAL (ADX: {current_adx:.2f}): Activating grid at {current_time}")
                    # Place initial orders immediately upon activation if portfolio needs balancing
                    # This now depends on is_grid_active
                    self._place_initial_orders() # Re-call this after activation
                    self._recalculate_dynamic_grid(current_price_decimal) # Recalculate grid for fresh entry
                    self._manage_grid_orders() # Place grid orders immediately

            elif self.is_grid_active:
                # Attempt to deactivate grid if ADX is above exit threshold for confirmation periods
                if adx_sufficient_history and all(adx_val > self.adx_exit_threshold for adx_val in self.adx_history):
                    self.is_grid_active = False
                    print(f"[{self.symbol}] ADX EXIT SIGNAL (ADX: {current_adx:.2f}): Deactivating grid and liquidating at {current_time}")

                    # Trigger full liquidation (sell all crypto holdings)
                    base_currency = symbol.split('_')[0]
                    crypto_to_sell = self.portfolio['crypto_holdings'].get(base_currency, Decimal('0'))
                    if crypto_to_sell > Decimal('0'):
                        # Use the current_price_decimal for liquidation
                        self.execute_order_func(symbol, 'sell', current_price_decimal, crypto_to_sell)
                        print(f"[{self.symbol}] Liquidated {crypto_to_sell:.8f} {base_currency} due to ADX exit.")
                    else:
                        print(f"[{self.symbol}] No crypto holdings to liquidate on ADX exit.")

                    # Cancel all active grid orders and clear grid levels
                    if self.active_buy_orders or self.active_sell_orders:
                        print(f"[{self.symbol}] Cancelling {len(self.active_buy_orders)} buy orders and {len(self.active_sell_orders)} sell orders.")
                    self.active_buy_orders = []
                    self.active_sell_orders = []
                    self.grid_levels = [] # Clear grid levels as they are no longer managed

        # --- Grid Management (ONLY if active) ---
        if self.is_grid_active:
            self._recalculate_dynamic_grid(current_price_decimal)
            self._manage_grid_orders()
        else:
            # If not active, ensure no orders are pending and grid levels are cleared
            # This handles cases where ADX might have deactivated it, but a manage_grid_orders call might follow if not explicitly cleared.
            if self.active_buy_orders or self.active_sell_orders:
                print(f"[{self.symbol}] Grid inactive, clearing {len(self.active_buy_orders)} buy orders and {len(self.active_sell_orders)} sell orders.")
            self.active_buy_orders = []
            self.active_sell_orders = []
            self.grid_levels = [] # Keep grid levels clear when inactive

def _manage_grid_orders(self): # This method correctly starts here.
    """
    Ensures that the `active_buy_orders` and `active_sell_orders` lists
    reflect the desired state of the grid based on `max_concurrent_orders_per_side`
    and current price. This re-places orders after fills on a DYNAMIC grid.
    """

    # Create a set of prices/types of currently active orders to prevent duplicates
    # This reflects orders that were *successfully placed* and are still active.
    active_order_identifiers = set()

    # Rebuild active lists based on current grid_levels status and if they were previously executed.
    temp_active_buy_orders = []
    for order in self.active_buy_orders:
        current_level_found = False
        for i, level_in_grid in enumerate(self.grid_levels):
            if level_in_grid['price'].compare(order['price']) == 0 and level_in_grid['type'] == order['type']:
                if level_in_grid['status'] != 'filled': # If it's not filled, keep it active
                    temp_active_buy_orders.append(order)
                    active_order_identifiers.add((order['price'], order['type'])) # Add to set if truly active
                current_level_found = True
                break
        # If the order's level is no longer in the dynamic grid, it's implicitly removed by not re-adding.
        # If current_level_found is False, it means the order's price is no longer part of the grid.

    temp_active_sell_orders = []
    for order in self.active_sell_orders:
        current_level_found = False
        for i, level_in_grid in enumerate(self.grid_levels):
            if level_in_grid['price'].compare(order['price']) == 0 and level_in_grid['type'] == order['type']:
                if level_in_grid['status'] != 'filled':
                    temp_active_sell_orders.append(order)
                    active_order_identifiers.add((order['price'], order['type'])) # Add to set if truly active
                current_level_found = True
                break

    self.active_buy_orders = temp_active_buy_orders
    self.active_sell_orders = temp_active_sell_orders


    # Identify potential buy orders: levels below current price that are 'open'
    potential_buy_levels = []
    for i, level in enumerate(self.grid_levels):
        if level['type'] == 'buy' and level['price'] < self.current_price and level['status'] == 'open':
            potential_buy_levels.append({'level': level, 'idx': i})
    potential_buy_levels.sort(key=lambda x: x['level']['price'], reverse=True)

    # Place new buy orders up to `max_concurrent_orders_per_side`
    for item in potential_buy_levels:
        level = item['level']
        level_idx = item['idx']

        # Use the identifier set to prevent duplicate active orders
        if (level['price'], level['type']) in active_order_identifiers:
            continue

        if len(self.active_buy_orders) >= self.config_params["max_concurrent_orders_per_side"]:
            break

        amount_to_trade = self._calculate_amount_for_order(self.symbol, level['price'])
        if amount_to_trade == Decimal('0'): continue

        required_cash = level['price'] * amount_to_trade * (Decimal('1') + self.commission_rate)
        if self.portfolio['cash'] >= required_cash:
            # CRITICAL FIX: Only add to active_buy_orders if _execute_order_func is successful
            if self.execute_order_func(self.symbol, 'buy', level['price'], amount_to_trade):
                self.active_buy_orders.append({
                    'symbol': self.symbol,
                    'type': 'buy',
                    'price': level['price'],
                    'amount': amount_to_trade,
                    'grid_level_idx': level_idx
                })
                self.grid_levels[level_idx]['status'] = 'pending'
                print(f"[{self.symbol}] Managing: Placed BUY order @ {level['price']:.2f} for {amount_to_trade:.8f}")
            # else: `_execute_order_func` will log failure if funds are insufficient. Order not added to active list.

    # Identify potential sell orders: levels above current price that are 'open'
    potential_sell_levels = []
    for i, level in enumerate(self.grid_levels):
        if level['type'] == 'sell' and level['price'] > self.current_price and level['status'] == 'open':
            potential_sell_levels.append({'level': level, 'idx': i})
    potential_sell_levels.sort(key=lambda x: x['level']['price'])

    # Place new sell orders up to `max_concurrent_orders_per_side`
    for item in potential_sell_levels:
        level = item['level']
        level_idx = item['idx']

        if (level['price'], level['type']) in active_order_identifiers:
            continue

        if len(self.active_sell_orders) >= self.config_params["max_concurrent_orders_per_side"]:
            break

        amount_to_trade = self._calculate_amount_for_order(self.symbol, level['price'])
        if amount_to_trade == Decimal('0'): continue

        base_currency = self.symbol.split('_')[0]
        if self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')) >= amount_to_trade:
            # CRITICAL FIX: Only add to active_sell_orders if _execute_order_func is successful
            if self.execute_order_func(self.symbol, 'sell', level['price'], amount_to_trade):
                self.active_sell_orders.append({
                    'symbol': self.symbol,
                    'type': 'sell',
                    'price': level['price'],
                    'amount': amount_to_trade,
                    'grid_level_idx': level_idx
                })
                self.grid_levels[level_idx]['status'] = 'pending'
                print(f"[{self.symbol}] Managing: Placed SELL order @ {level['price']:.2f} for {amount_to_trade:.8f}")
            # else: `_execute_order_func` will log failure if funds are insufficient. Order not added to active list.


def notify_order_filled(self, filled_order):
    """
    Called by the backtester when an order is filled.
    Updates the strategy's internal state and re-arms the grid for a DYNAMIC grid.
    It finds the *current* index of the filled order within the *current* grid_levels.
    """
    symbol = filled_order['symbol']
    order_type = filled_order['type']
    filled_price = filled_order['price']

    # Find the *current* index of the filled_price in the *current* grid_levels
    current_level_idx = -1
    for i, level in enumerate(self.grid_levels):
        # Using compare(0) for exact Decimal equality due to floating point nuances
        if level['price'].compare(filled_price) == 0 and level['type'] == order_type:
            current_level_idx = i
            break

    if current_level_idx != -1:
        self.grid_levels[current_level_idx]['status'] = 'filled'

        # Re-arm the corresponding opposite order on the DYNAMIC grid.
        # This logic now explicitly looks for the opposite type at a nearby level.
        # If a BUY at Level N fills, re-enable the SELL level just above it (Level N+X).
        # If a SELL at Level N fills, re-enable the BUY level just below it (Level N-X).

        if order_type == 'buy':
            # Iterate upwards from the filled level's current index
            corresponding_sell_level_found = False
            for i in range(current_level_idx + 1, len(self.grid_levels)):
                level_to_rearm = self.grid_levels[i]
                if level_to_rearm['type'] == 'sell': # We are looking for a sell level
                    # If this sell level was previously 'filled' (meaning it was sold), we re-open it.
                    # Also, ensure it's not currently pending to avoid re-adding an order still in play.
                    if level_to_rearm['status'] == 'filled':
                        level_to_rearm['status'] = 'open'
                        corresponding_sell_level_found = True
                        break
            if not corresponding_sell_level_found:
                pass # Warning is fine, grid might be out of range, or no higher sell levels.


        elif order_type == 'sell':
            # Iterate downwards from the filled level's current index
            corresponding_buy_level_found = False
            for i in range(current_level_idx - 1, -1, -1):
                level_to_rearm = self.grid_levels[i]
                if level_to_rearm['type'] == 'buy': # We are looking for a buy level
                    if level_to_rearm['status'] == 'filled':
                        level_to_rearm['status'] = 'open'
                        corresponding_buy_level_found = True
                        break
            if not corresponding_buy_level_found:
                pass # Warning is fine, grid might be out of range, or no lower buy levels.
        else:
            # This can happen if a filled order's level is no longer in the dynamic grid
            # (e.g., grid shifted significantly) or if its status was already updated.
            pass


# Example usage (for testing purposes only, not part of actual bot run)
if __name__ == "__main__":
    from config.parameters import GRID_STRATEGY_PARAMS, INITIAL_BACKTEST_CAPITAL, COMMISSION_RATE
    from config.settings import KRAKEN_MIN_ORDER_SIZES

    # Ensure GRID_STRATEGY_PARAMS has the ADX settings for this test
    test_params = GRID_STRATEGY_PARAMS.copy()
    test_params["symbol"] = "BTC_USD"
    test_params["adx_period"] = 14
    test_params["adx_entry_threshold"] = 25.0
    test_params["adx_exit_threshold"] = 30.0
    test_params["adx_confirmation_periods"] = 3
    test_params["percentage_of_equity_per_grid_order"] = 0.05 # For testing dynamic sizing in example

    mock_portfolio = {
        'cash': Decimal(str(INITIAL_BACKTEST_CAPITAL)),
        'crypto_holdings': {},
        'open_orders': []
    }

    mock_min_order_sizes = {
        'BTC': Decimal('0.0001'), 'ETH': Decimal('0.005'), 'SOL': Decimal('0.1'), 'ADA': Decimal('1.0'),
        'DOT': Decimal('1.0'), 'AVAX': Decimal('0.1'), 'ATOM': Decimal('1.0'), 'LINK': Decimal('1.0'), 'XRP': Decimal('10.0')
    }

    # Add simulated slippage for mock_execute_order as well for full testing
    sim_slippage_for_mock = Decimal(str(GRID_STRATEGY_PARAMS.get("simulated_slippage_percentage", 0.0)))

    def mock_execute_order(symbol, order_type, price, amount):
        # Apply simulated slippage for mock execution
        adjusted_price_decimal = price
        if order_type == 'buy':
            adjusted_price_decimal = price * (Decimal('1') + sim_slippage_for_mock)
        elif order_type == 'sell':
            adjusted_price_decimal = price * (Decimal('1') - sim_slippage_for_mock)

        cost_revenue = adjusted_price_decimal * amount
        commission = cost_revenue * Decimal(str(COMMISSION_RATE))

        base_currency = symbol.split('_')[0]

        if amount <= Decimal('0'):
            print(f"MOCK EXECUTE FAILED: Zero or negative amount requested for {order_type} {symbol}. Amount: {amount:.8f}")
            return False

        if order_type == 'buy':
            if mock_portfolio['cash'] >= cost_revenue + commission:
                mock_portfolio['cash'] -= (cost_revenue + commission)
                mock_portfolio['crypto_holdings'][base_currency] = mock_portfolio['crypto_holdings'].get(base_currency, Decimal('0')) + amount
                print(f"MOCK EXECUTE: BUY {amount:.8f} {base_currency} @ {adjusted_price_decimal:.2f} (Orig: {price:.2f}) (Cash: {mock_portfolio['cash']:.2f}, {base_currency}: {mock_portfolio['crypto_holdings'].get(base_currency, Decimal('0')):.8f})")
                return True
            else:
                print(f"MOCK EXECUTE FAILED (BUY): Insufficient cash for {amount:.8f} {symbol} @ {adjusted_price_decimal:.2f}. Needed: {cost_revenue + commission:.2f}, Have: {mock_portfolio['cash']:.2f}")
                return False
        elif order_type == 'sell':
            if mock_portfolio['crypto_holdings'].get(base_currency, Decimal('0')) >= amount:
                mock_portfolio['cash'] += (cost_revenue - commission)
                mock_portfolio['crypto_holdings'][base_currency] -= amount
                print(f"MOCK EXECUTE: SELL {amount:.8f} {base_currency} @ {adjusted_price_decimal:.2f} (Orig: {price:.2f}) (Cash: {mock_portfolio['cash']:.2f}, {base_currency}: {mock_portfolio['crypto_holdings'].get(base_currency, Decimal('0')):.8f})")
                return True
            else:
                print(f"MOCK EXECUTE FAILED (SELL): Insufficient crypto for {amount:.8f} {symbol} @ {adjusted_price_decimal:.2f}. Needed: {amount:.8f}, Have: {mock_portfolio['crypto_holdings'].get(base_currency, Decimal('0')):.8f}")
                return False
        return False

    # Test for BTC_USD with ADX logic
    strategy = GridStrategy(test_params)
    strategy.set_backtester_context(mock_portfolio, mock_execute_order, COMMISSION_RATE, mock_min_order_sizes, sim_slippage_for_mock)

    # Need enough initial data for ADX calculation before strategy can activate
    # A period of 2 * adx_period + confirmation periods is a safe starting buffer
    initial_btc_price = Decimal('30000.0') # Set a lower price for easier manual calculation verification
    current_time_mock = datetime.now()

    # Generate enough dummy OHLCV data to satisfy ADX calculation minimums
    # And make ADX initially low to trigger entry
    adx_calc_buffer = max(strategy.adx_period * 2, strategy.atr_period + 5) + strategy.adx_confirmation_periods + 5
    dummy_ohlcv_data = []
    for i in range(adx_calc_buffer):
        price_offset = Decimal(str(math.sin(i * 0.1) * 100)) # Simple sine wave for ranging behavior
        dummy_price = initial_btc_price + price_offset
        dummy_ohlcv_data.append({
            'timestamp': current_time_mock + timedelta(minutes=i),
            'open': dummy_price,
            'high': dummy_price * Decimal('1.001'),
            'low': dummy_price * Decimal('0.999'),
            'close': dummy_price,
            'volume': Decimal('100')
        })
    dummy_df = pd.DataFrame(dummy_ohlcv_data).set_index('timestamp')
    strategy.ohlcv_history = dummy_df # Pre-fill history

    print("\n--- Simulating candles with ADX logic ---")

    # Simulate a period of ranging market (ADX below entry threshold)
    print("\nSimulating RANGING market conditions to trigger ADX entry...")
    for i in range(5): # Simulate a few ranging candles
        current_time_mock += timedelta(minutes=1)
        # Keep price relatively stable for low ADX
        candle_data = {
            'open': initial_btc_price,
            'high': initial_btc_price * Decimal('1.0005'),
            'low': initial_btc_price * Decimal('0.9995'),
            'close': initial_btc_price,
            'volume': Decimal('100')
        }
        # Simulate very low ADX values artificially for testing entry trigger
        # (This is just for testing the ADX state change, real ADX will be calculated)
        strategy.on_candle("BTC_USD", pd.Series(candle_data), initial_btc_price, current_time_mock)
        print(f"[{strategy.symbol}] ADX history (last {strategy.adx_confirmation_periods} values): {[f'{x:.2f}' for x in strategy.adx_history]}")
        print(f"[{strategy.symbol}] Grid Active: {strategy.is_grid_active}")

    print("\nSimulating a trending market to trigger ADX exit...")
    # Simulate a strong trend (ADX rises above exit threshold)
    trending_price = initial_btc_price
    for i in range(5): # Simulate a few trending candles
        current_time_mock += timedelta(minutes=1)
        trending_price *= Decimal('1.005') # Simulate price moving up strongly
        candle_data = {
            'open': trending_price / Decimal('1.005'),
            'high': trending_price * Decimal('1.001'),
            'low': trending_price * Decimal('0.999'),
            'close': trending_price,
            'volume': Decimal('100')
        }
        # Simulate very high ADX values artificially for testing exit trigger
        strategy.on_candle("BTC_USD", pd.Series(candle_data), trending_price, current_time_mock)
        print(f"[{strategy.symbol}] ADX history (last {strategy.adx_confirmation_periods} values): {[f'{x:.2f}' for x in strategy.adx_history]}")
        print(f"[{strategy.symbol}] Grid Active: {strategy.is_grid_active}")

    print("\n--- Final active orders after ADX simulation ---")
    print(f"Active Buy Orders: {len(strategy.active_buy_orders)}")
    print(f"Active Sell Orders: {len(strategy.active_sell_orders)}")
    print(f"Final Portfolio Cash: {mock_portfolio['cash']:.2f}")
    print(f"Final Portfolio Crypto Holdings (BTC): {mock_portfolio['crypto_holdings'].get('BTC', Decimal('0')):.8f}")