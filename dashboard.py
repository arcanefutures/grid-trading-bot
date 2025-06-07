# dashboard.py

import streamlit as st
import pandas as pd
import json
from decimal import Decimal
import time

# Page configuration
st.set_page_config(
    page_title="Grid Trading Bot Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load data from our JSON file
def load_data():
    try:
        with open('dashboard_status.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return a default structure if file not found or is empty
        return None

# Title
st.title("📈 Grid Trading Bot Dashboard")

# Create a placeholder for the entire dashboard
placeholder = st.empty()

# Main loop to auto-refresh
while True:
    data = load_data()

    if not data:
        with placeholder.container():
            st.warning("Waiting for dashboard_status.json file... Please ensure the trading bot is running.")
        time.sleep(5)
        continue

    # Extract data with default values
    timestamp = data.get('timestamp', 'N/A')
    is_grid_active = data.get('is_grid_active', False)
    current_price = Decimal(data.get('current_price', '0'))
    portfolio = data.get('portfolio', {})
    pnl = data.get('pnl', {})
    active_orders = data.get('active_orders', {'buys': [], 'sells': []})
    equity_history_raw = data.get('equity_history', [])
    trade_history_raw = data.get('trade_history', [])

    total_equity = Decimal(portfolio.get('total_equity', '0'))
    cash = Decimal(portfolio.get('cash', '0'))
    holdings_value = Decimal(portfolio.get('holdings_value', '0'))
    holdings_amount = Decimal(portfolio.get('holdings_amount', '0'))
    
    realized_pnl = Decimal(pnl.get('realized_pnl', '0'))
    unrealized_pnl = Decimal(pnl.get('unrealized_pnl', '0'))
    net_pnl = Decimal(pnl.get('net_pnl', '0'))
    
    # --- Dashboard Layout ---
    with placeholder.container():
        
        # --- Top Row: Key Metrics ---
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Equity (USD)", f"${total_equity:,.2f}", f"{net_pnl:+,.2f}")
        col2.metric("Cash", f"${cash:,.2f}")
        col3.metric("Holdings Value", f"${holdings_value:,.2f}", f"{holdings_amount:,.4f} ADA")
        col4.metric("Realized P&L", f"${realized_pnl:,.2f}")
        col5.metric("Unrealized P&L", f"${unrealized_pnl:,.2f}")

        st.divider()

        # --- Second Row: Chart and Status ---
        chart_col, status_col = st.columns([3, 1]) # Give more space to the chart
        
        with chart_col:
            st.subheader("Equity Curve")
            if equity_history_raw:
                equity_df = pd.DataFrame(equity_history_raw, columns=['Timestamp', 'Equity'])
                equity_df['Timestamp'] = pd.to_datetime(equity_df['Timestamp'])
                equity_df['Equity'] = pd.to_numeric(equity_df['Equity'])
                st.line_chart(equity_df.set_index('Timestamp'))
            else:
                st.info("Waiting for equity history...")

        with status_col:
            st.subheader("Bot Status")
            if is_grid_active:
                st.success("GRID ACTIVE")
            else:
                st.warning("GRID INACTIVE")
            
            st.metric("Current Market Price", f"${current_price:,.4f}")
            st.write(f"Last update: {pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S %Z')}")

        st.divider()

        # --- Third Row: Active Orders and Trade History ---
        buys_col, sells_col, history_col = st.columns([1,1,2])

        with buys_col:
            st.subheader("🟢 Active Buy Orders")
            if active_orders['buys']:
                st.dataframe(pd.DataFrame(active_orders['buys']), use_container_width=True)
            else:
                st.info("No active buy orders.")

        with sells_col:
            st.subheader("🔴 Active Sell Orders")
            if active_orders['sells']:
                st.dataframe(pd.DataFrame(active_orders['sells']), use_container_width=True)
            else:
                st.info("No active sell orders.")
        
        with history_col:
            st.subheader("📋 Trade History")
            if trade_history_raw:
                history_df = pd.DataFrame(trade_history_raw)
                # Format for better display
                history_df['filled_price'] = pd.to_numeric(history_df['filled_price']).map('{:,.4f}'.format)
                history_df['amount'] = pd.to_numeric(history_df['amount']).map('{:,.4f}'.format)
                history_df['realized_pnl'] = pd.to_numeric(history_df['realized_pnl']).map('{:,.2f}'.format)
                st.dataframe(history_df[['timestamp', 'order_type', 'amount', 'filled_price', 'realized_pnl']], use_container_width=True)
            else:
                st.info("No trades executed yet.")

    # --- Refresh Rate ---
    time.sleep(2)