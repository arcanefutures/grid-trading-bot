# backtesting/backtester.py
import pandas as pd
import os
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP 
import logging 

# Import our strategy and configuration
from core.strategy import GridStrategy 
from config.parameters import GRID_STRATEGY_PARAMS, INITIAL_BACKTEST_CAPITAL, COMMISSION_RATE
from config.settings import KRAKEN_MIN_ORDER_SIZES 

class Backtester:
    def __init__(self, data_folder='backtesting/data/raw/',
                 initial_capital=INITIAL_BACKTEST_CAPITAL,
                 commission_rate=COMMISSION_RATE,
                 min_order_sizes_map=KRAKEN_MIN_ORDER_SIZES,
                 simulated_slippage_percentage=GRID_STRATEGY_PARAMS["simulated_slippage_percentage"]): # NEW PARAMETER

        self.data_folder = data_folder
        self.initial_capital = initial_capital
        self.commission_rate = Decimal(str(commission_rate))
        self.min_order_sizes = min_order_sizes_map
        self.simulated_slippage_percentage = Decimal(str(simulated_slippage_percentage)) # Store as Decimal
        
        self.market_data = {} 
        self.strategies = {} 
        
        self.portfolio = {
            'cash': Decimal(str(initial_capital)), 
            'crypto_holdings': {}, 
        }
        self.trade_log = [] 
        self.equity_curve = [] 
        self.current_time = None
        self.current_price = None 

        self.highest_equity_peak = Decimal('0') 
        self.global_max_drawdown_percentage = Decimal(str(GRID_STRATEGY_PARAMS["global_max_drawdown_percentage"]))
        
        self.initial_deployed_equity = Decimal('0') 

        self.debug_logger = logging.getLogger('backtest_debug')
        self.debug_logger.setLevel(logging.DEBUG)
        logs_dir = 'logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        debug_log_filepath = os.path.join(logs_dir, f"backtest_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        if not self.debug_logger.handlers:
            file_handler = logging.FileHandler(debug_log_filepath)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.debug_logger.addHandler(file_handler)
        self.debug_logger.propagate = False 
        print(f"Verbose debug log saved to: {debug_log_filepath}") 


    def load_data(self, symbol_pair, timeframe='1m'):
        filename = f"{symbol_pair}_{timeframe}.csv"
        filepath = os.path.join(self.data_folder, filename)
        
        if not os.path.exists(filepath):
            print(f"Error: Data file not found for {symbol_pair} at {filepath}")
            return False

        print(f"Loading data for {symbol_pair} from {filepath}...")
        df = pd.read_csv(filepath)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Convert relevant columns to Decimal for precision
        df['open'] = df['open'].apply(lambda x: Decimal(str(x)))
        df['high'] = df['high'].apply(lambda x: Decimal(str(x)))
        df['low'] = df['low'].apply(lambda x: Decimal(str(x)))
        df['close'] = df['close'].apply(lambda x: Decimal(str(x)))
        df['volume'] = df['volume'].apply(lambda x: Decimal(str(x)))

        if df.empty:
            print(f"Warning: Loaded an empty DataFrame for {symbol_pair}.")
            return False
        
        self.market_data[symbol_pair] = df
        print(f"Loaded {len(df)} candles for {symbol_pair}.")
        return True

    def run_backtest(self, symbol_pair):
        if symbol_pair not in self.market_data:
            print(f"Error: Data for {symbol_pair} not loaded. Please call load_data() first.")
            return

        print(f"\n--- Starting Backtest for {symbol_pair} ---")
        data = self.market_data[symbol_pair]

        strategy_params_for_symbol = GRID_STRATEGY_PARAMS.copy()
        strategy_params_for_symbol["symbol"] = symbol_pair

        strategy_instance = GridStrategy(strategy_params_for_symbol)
        self.strategies[symbol_pair] = strategy_instance

        # Set Backtester's initial current_time and current_price based on the first candle
        first_timestamp, first_candle = list(data.iloc[0:1].iterrows())[0]
        self.current_time = first_timestamp
        self.current_price = first_candle['close']

        # Inject context into the strategy
        strategy_instance.set_backtester_context(
            self.portfolio,
            self._execute_order,
            self.commission_rate,
            self.min_order_sizes,
            self.simulated_slippage_percentage # <--- THIS IS THE MISSING ARGUMENT YOU NEED TO ADD
        )

        # Pass the initial price for grid calculation
        strategy_instance.calculate_grid_levels(self.current_price)
 
        
        # Initial order placement for the strategy. Strategy also needs initial price.
        strategy_instance.current_price = self.current_price 
        strategy_instance._place_initial_orders()
        
        # --- CRITICAL: RECORD INITIAL EQUITY AFTER SETUP AND SET HIGHEST_EQUITY_PEAK ---
        initial_equity_after_setup = self.portfolio['cash'] + \
                                     self.portfolio['crypto_holdings'].get(symbol_pair.split('_')[0], Decimal('0')) * self.current_price
        
        self.initial_deployed_equity = initial_equity_after_setup 
        
        self.highest_equity_peak = initial_equity_after_setup 
        self._record_equity(symbol_pair, self.current_price) 
        
        self.debug_logger.debug(f"Initial equity after setup: {initial_equity_after_setup:,.2f}")
        self.debug_logger.debug(f"Initial highest_equity_peak (for global drawdown): {self.highest_equity_peak:,.2f}")
        self.debug_logger.debug(f"Global Drawdown target (20% from peak): {self.highest_equity_peak * (Decimal('1') - self.global_max_drawdown_percentage):,.2f}")

        backtest_halted_by_drawdown = False 


        strategy_instance.set_backtester_context(
            self.portfolio,
            self._execute_order,
            self.commission_rate,
            self.min_order_sizes,
            self.simulated_slippage_percentage # NEW ARGUMENT
        )

        # Main backtest loop (start from the *second* candle)
        for i, (timestamp, candle) in enumerate(data.iloc[1:].iterrows()): 
            self.current_time = timestamp 
            current_price = candle['close'] 
            
            # 1. Update holdings based on previous candle's fills and strategy actions
            self._check_for_fills(symbol_pair, candle, strategy_instance)
            strategy_instance.on_candle(symbol_pair, candle, current_price, self.current_time)
            
            # 2. Calculate current equity after all actions for this candle are considered
            current_equity_for_check = self.portfolio['cash'] + \
                                       self.portfolio['crypto_holdings'].get(symbol_pair.split('_')[0], Decimal('0')) * current_price
            
            # 3. Update highest peak
            if current_equity_for_check > self.highest_equity_peak: 
                self.highest_equity_peak = current_equity_for_check

            # 4. Record equity in the curve
            self._record_equity(symbol_pair, current_price) 

            # 5. Perform drawdown check
            drawdown_threshold = self.highest_equity_peak * (Decimal('1') - self.global_max_drawdown_percentage)
            self.debug_logger.debug(f"Candle {self.current_time}: Equity: {current_equity_for_check:,.2f}, Peak: {self.highest_equity_peak:,.2f}, Threshold: {drawdown_threshold:,.2f}")

            if current_equity_for_check < drawdown_threshold:
                self.debug_logger.critical(f"\n!!! GLOBAL MAX DRAWDOWN TRIGGERED at {self.current_time} !!!") 
                self.debug_logger.critical(f"Equity: {current_equity_for_check:,.2f}, Peak: {self.highest_equity_peak:,.2f}, Max Allowed Drawdown: {self.global_max_drawdown_percentage*100:.2f}%")
                
                # Simulate liquidation
                base_currency = symbol_pair.split('_')[0]
                if self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')) > Decimal('0'):
                    self.debug_logger.critical(f"Simulating liquidation of {self.portfolio['crypto_holdings'][base_currency]:.8f} {base_currency} at {current_price:.2f} to stop losses.")
                    self._execute_order(symbol_pair, 'sell', current_price, self.portfolio['crypto_holdings'][base_currency])
                
                # Record final equity point after liquidation
                self._record_equity(symbol_pair, current_price) 
                
                self.debug_logger.critical("Backtest halted due to global max drawdown.")
                print("Backtest halted due to global max drawdown. Check debug log for details.") 
                backtest_halted_by_drawdown = True 
                break 

            if i % 1000 == 0: 
                total_equity_at_progress_point = self.equity_curve[-1]['equity'] 
                print(f"Processing candle: {self.current_time} - Current Equity: {total_equity_at_progress_point:,.2f}")

        print("\n--- Backtest Finished ---")
        self.generate_report(symbol_pair, backtest_halted_by_drawdown) 

    def _execute_order(self, symbol, order_type, price_decimal, amount_decimal):
        """
        Simulates the execution of an order (buy or sell) with simulated slippage.
        Updates the portfolio cash and crypto holdings, and records the trade.
        Returns True if successful, False otherwise (e.g., insufficient funds/holdings).
        """
        # Apply simulated slippage to the fill price
        adjusted_price_decimal = price_decimal
        if order_type == 'buy':
            # Buy orders: slippage makes the price higher (worse for us)
            adjusted_price_decimal = price_decimal * (Decimal('1') + self.simulated_slippage_percentage)
        elif order_type == 'sell':
            # Sell orders: slippage makes the price lower (worse for us)
            adjusted_price_decimal = price_decimal * (Decimal('1') - self.simulated_slippage_percentage)

        cost_decimal = adjusted_price_decimal * amount_decimal # Use adjusted price for cost/revenue
        commission_decimal = cost_decimal * self.commission_rate

        base_currency = symbol.split('_')[0]

        if amount_decimal <= Decimal('0'):
            self.debug_logger.warning(f"[{self.current_time.strftime('%Y-%m-%d %H:%M')}] EXEC FAILED: Zero or negative amount requested for {order_type} {symbol}. Amount: {amount_decimal:.8f}")
            return False

        if order_type == 'buy':
            if self.portfolio['cash'] >= cost_decimal + commission_decimal:
                self.portfolio['cash'] -= (cost_decimal + commission_decimal)
                self.portfolio['crypto_holdings'][base_currency] = self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')) + amount_decimal
                self.trade_log.append({
                    'timestamp': self.current_time,
                    'symbol': symbol,
                    'type': 'BUY',
                    'price': float(adjusted_price_decimal), # Log adjusted price
                    'amount': float(amount_decimal),
                    'cost': float(cost_decimal),
                    'commission': float(commission_decimal),
                    'pnl': float(-commission_decimal) # PnL for a buy is just commission (until sold)
                })
                print(f"[{self.current_time.strftime('%Y-%m-%d %H:%M')}] EXEC: BUY {amount_decimal:.8f} {base_currency} @ {adjusted_price_decimal:.2f} (Orig: {price_decimal:.2f}) (Cash: {self.portfolio['cash']:.2f}, {base_currency}: {self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')):.8f})")
                return True
            else:
                self.debug_logger.warning(f"[{self.current_time.strftime('%Y-%m-%d %H:%M')}] EXEC FAILED: Insufficient cash for BUY {amount_decimal:.8f} {symbol}. Needed: {cost_decimal + commission_decimal:.2f}, Have: {self.portfolio['cash']:.2f}")
                return False
        elif order_type == 'sell':
            if self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')) >= amount_decimal:
                self.portfolio['cash'] += (cost_decimal - commission_decimal) # cost_decimal here is revenue
                self.portfolio['crypto_holdings'][base_currency] -= amount_decimal
                self.trade_log.append({
                    'timestamp': self.current_time,
                    'symbol': symbol,
                    'type': 'SELL',
                    'price': float(adjusted_price_decimal), # Log adjusted price
                    'amount': float(amount_decimal),
                    'revenue': float(cost_decimal),
                    'commission': float(commission_decimal),
                    'pnl': float(cost_decimal - commission_decimal) # PnL for a sell is revenue minus commission
                })
                print(f"[{self.current_time.strftime('%Y-%m-%d %H:%M')}] EXEC: SELL {amount_decimal:.8f} {base_currency} @ {adjusted_price_decimal:.2f} (Orig: {price_decimal:.2f}) (Cash: {self.portfolio['cash']:.2f}, {base_currency}: {self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')):.8f})")
                return True
            else:
                self.debug_logger.warning(f"[{self.current_time.strftime('%Y-%m-%d %H:%M')}] EXEC FAILED: Insufficient crypto for SELL {amount_decimal:.8f} {symbol}. Needed: {amount_decimal:.8f}, Have: {self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')):.8f}")
                return False
        return False


    def _check_for_fills(self, symbol, candle, strategy_instance):
        """
        Checks if any active orders from the strategy would have been filled by the current candle's price action.
        Notifies the strategy of fills.
        """
        filled_orders = [] 
        
        # Check Buy Orders (fill if candle's low <= order price)
        for order in strategy_instance.active_buy_orders[:]: 
            order_price = order['price'] 
            
            if candle['low'] <= order_price:
                if self._execute_order(order['symbol'], order['type'], order_price, order['amount']):
                    filled_orders.append(order)
        
        # Check Sell Orders (fill if candle's high >= order price)
        for order in strategy_instance.active_sell_orders[:]: 
            order_price = order['price'] 
            
            if candle['high'] >= order_price:
                if self._execute_order(order['symbol'], order['type'], order_price, order['amount']):
                    filled_orders.append(order)

        # Notify the strategy of all filled orders after processing all checks for the current candle.
        for filled_order in filled_orders:
            strategy_instance.notify_order_filled(filled_order)


    def _record_equity(self, symbol, current_price_decimal):
        """
        Calculates and records the total portfolio equity at the current timestamp.
        """
        base_currency = symbol.split('_')[0]
        
        crypto_value = self.portfolio['crypto_holdings'].get(base_currency, Decimal('0')) * current_price_decimal
        
        total_equity = self.portfolio['cash'] + crypto_value
        
        self.equity_curve.append({
            'timestamp': self.current_time,
            'equity': float(total_equity), 
            'cash': float(self.portfolio['cash']),
            'crypto_value': float(crypto_value)
        })

    def generate_report(self, symbol_pair, was_halted=False): 
        """
        Generates and prints a backtest report, and plots the equity curve.
        """
        if not self.trade_log:
            print("No trades executed during backtest.")
            return

        trades_df = pd.DataFrame(self.trade_log)
        equity_df = pd.DataFrame(self.equity_curve).set_index('timestamp')

        # --- CONSOLE REPORT ---
        report_output = []
        report_output.append(f"\n--- Backtest Report for {symbol_pair} ---")
        report_output.append(f"Overall Initial Capital: ${self.initial_capital:,.2f}") 
        
        final_equity = equity_df['equity'].iloc[-1]
        report_output.append(f"Final Overall Equity: ${final_equity:,.2f}")
        report_output.append(f"Overall Net PnL (from Initial Capital): ${final_equity - self.initial_capital:,.2f}")
        
        report_output.append(f"Strategy Deployed Capital: ${float(self.initial_deployed_equity):,.2f}") 
        report_output.append(f"Net PnL on Deployed Capital: ${final_equity - float(self.initial_deployed_equity):,.2f}")
        
        # Calculate ROI
        net_pnl_deployed = final_equity - float(self.initial_deployed_equity)
        roi_percentage = (net_pnl_deployed / float(self.initial_deployed_equity)) * 100 if float(self.initial_deployed_equity) != 0 else 0
        report_output.append(f"ROI on Deployed Capital: {roi_percentage:.2f}%")


        report_output.append(f"Total Trades: {len(trades_df)}")

        total_commissions = trades_df['commission'].sum()
        report_output.append(f"Total Commissions Paid: ${total_commissions:,.2f}")

        initial_equity_for_drawdown_calc = equity_df['equity'].iloc[0] 
        equity_series_for_drawdown = pd.Series([initial_equity_for_drawdown_calc] + equity_df['equity'].tolist(), 
                                               index=[equity_df.index[0] - pd.Timedelta(seconds=1)] + equity_df.index.tolist())
        
        peak = equity_series_for_drawdown.expanding(min_periods=1).max()
        drawdown = (equity_series_for_drawdown - peak) / peak
        max_drawdown = drawdown.min()
        report_output.append(f"Maximum Drawdown (from Deployed Capital): {max_drawdown * 100:.2f}%")

        winning_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        report_output.append(f"Win Rate (per leg): {win_rate * 100:.2f}%")

        for line in report_output:
            print(line)

        # --- PLOTTING EQUITY CURVE ---
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 7)) 
        plt.plot(equity_df.index, equity_df['equity'], label='Equity Curve', color='blue')
        plt.title(f'Equity Curve for {symbol_pair} Grid Strategy Backtest')
        plt.xlabel('Time')
        plt.ylabel('Equity ($)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        plt.show() 

        # --- SAVE REPORT TO FILE ---
        reports_dir = 'reports'
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        report_filename = os.path.join(reports_dir, f"backtest_report_{symbol_pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(report_filename, 'w') as f:
            for line in report_output:
                f.write(line + "\n")
            f.write("\n") 
            f.write("--- Trade Log ---\n")
            f.write(trades_df.to_string() + "\n") 
            f.write(f"\n--- Backtest Status: {'HALTED by Global Drawdown' if was_halted else 'COMPLETED Full Data'}. ---") 

        print(f"Report saved to {report_filename}")