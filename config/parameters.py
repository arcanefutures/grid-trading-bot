# config/parameters.py

# --- List of assets to trade (Use '-' for Coinbase) ---
PRODUCT_IDS = [
    "ADA-USD",
    "SOL-USD",
    "BTC-USD",
    "ETH-USD",
]

# --- Portfolio & Commission ---
INITIAL_BACKTEST_CAPITAL = 1000.0
COMMISSION_RATE = 0.004 

# --- Grid Strategy Parameters ---
GRID_STRATEGY_PARAMS = {
    # Grid Dimensions
    "number_of_grids": 10,
    "atr_period": 14,
    "grid_atr_multiplier": "2.5",

    # Sizing & Risk
    "percentage_of_equity_per_grid_order": "0.02", # e.g., 2% of current equity per order
    "max_order_size_usd": 500.0, # A safety cap on the max USD value of any single order
    "max_concurrent_orders_per_side": 5, 

    # ADX Filter
    "adx_period": 14,
    "adx_confirmation_periods": 3,
    "adx_stable_readings_required": 5,
    "adx_entry_threshold": 20.0,
    "adx_exit_threshold": 35.0,
    
    # Simulation Settings
    "simulated_slippage_percentage": "0.0005", # 0.05%
    "initial_position_type": "cash_only",
}