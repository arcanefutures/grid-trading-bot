# config/parameters.py

# --- GENERAL BOT SETTINGS ---
BOT_NAME = "MyCryptoGridBot"
INITIAL_BACKTEST_CAPITAL = 1000.0 # USD
COMMISSION_RATE = 0.004 # e.g., 0.4% Taker fee on Kraken (conservative)

# --- GRID STRATEGY PARAMETERS ---
GRID_STRATEGY_PARAMS = {
    "symbols": [
        "BTC_USD", "ETH_USD", "SOL_USD", "ADA_USD", 
        "DOT_USD", "AVAX_USD", "ATOM_USD", "LINK_USD", "XRP_USD"
    ],
    
    # Static Grid Sizing Parameters (These are now placeholders; dynamic calculation will override)
    "grid_lower_bound": 80000.0, 
    "grid_upper_bound": 90000.0, 
    
    # Dynamic Grid Sizing Parameters (These will be used by the dynamic grid logic)
    "atr_period": 14, # Period for ATR calculation (e.g., 14 candles)
    "grid_atr_multiplier": 20.0, # How many times ATR to use for total grid width
                                 # We will set this to 6.0 after this fix is applied if desired.
    "grid_center_deviation_percentage": 0.0, # Optional: For a static percentage deviation from current price if not using ATR

    "number_of_grids": 10, # Initial number of grids. Will define density within ATR range.
                            # We will set this to 5 after this fix is applied if desired.
    
    # Capital Allocation Settings (CRITICAL: All parameters needed here)
    "max_capital_allocation_percentage": 1.0, # Percentage of INITIAL_BACKTEST_CAPITAL to deploy
    "percentage_of_equity_per_grid_order": 0.05, # e.g., 0.005 for 5% of current equity per order
    "max_concurrent_orders_per_side": 5, # Maximum number of active buy/sell orders at any time

    "profit_per_grid_percentage": 0.02, # Target profit percentage per grid level move (e.g., 0.02 for 2%)
    
    "initial_position_type": "neutral", # How the bot initializes its position ('neutral', 'cash_only')
    "initial_long_position_amount_usd": 0.0, # If starting with specific crypto amount (not used by 'neutral' type)

    "global_max_drawdown_percentage": 0.20, # Global stop loss percentage from peak equity

    # --- NEW: Simulated Slippage and Max Order Size ---
    "simulated_slippage_percentage": 0.0005, # e.g., 0.0005 for 0.05% slippage on each leg (buy/sell)
    "max_order_size_usd": 5000.0, # Maximum USD amount for a single order, regardless of equity
}