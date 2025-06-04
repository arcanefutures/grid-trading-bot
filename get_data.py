import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import os # Import os for directory creation

class DataCollector:
    def __init__(self):
        # Initialize Binance exchange with a higher timeout
        self.exchange = ccxt.binance({
            'timeout': 30000, # Increase timeout to 30 seconds (30000 ms)
            'enableRateLimit': True, # This tells ccxt to respect exchange rate limits automatically
        })
        self.pairs = [
            "SOL/USDT", "ADA/USDT", "DOT/USDT", "AVAX/USDT", "ATOM/USDT", "LINK/USDT", "XRP/USDT",
            # Add the new pairs you intend to download (ensure they are USDT if from Binance)
            "LTC/USDT", "BCH/USDT", "XLM/USDT", "ETC/USDT", "UNI/USDT", "DOGE/USDT"
        ]
        self.data_dir = 'backtesting/data/raw/' # Define your data directory

        # Create the data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory: {self.data_dir}")

    def fetch_ohlcv(self, symbol, timeframe='1m', days=90):
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        all_ohlcv = []
        request_count = 0
        
        print(f"Collecting {symbol} for {days} days...")
        print(f"Start time: {datetime.fromtimestamp(since/1000)}")
        print(f"End time: {datetime.fromtimestamp(end_time/1000)}")
        
        # Binance limit for klines is typically 1000, 720 is fine, but lets be flexible
        # Binance has a weight of 1 for 1-minute klines up to 500, then 5 for 500-1000.
        # We'll use 1000 to maximize data per request within rate limits
        limit_per_request = 1000 
        
        while since < end_time:
            request_count += 1
            print(f"Request {request_count} for {symbol}: Fetching from {datetime.fromtimestamp(since/1000)}")
            
            try:
                # Use exchange.fetch_ohlcv, ccxt's enableRateLimit will handle delays
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit=limit_per_request)
                
                if not ohlcv:
                    print(f"No more data returned for {symbol}.")
                    break # No more data to fetch for this period
                
                all_ohlcv.extend(ohlcv)
                
                # Update 'since' to the timestamp of the last candle + 1 minute (in ms)
                # Ensure we don't request data beyond end_time
                since = ohlcv[-1][0] + 60000 
                
                print(f"Got {len(ohlcv)} candles. Total collected: {len(all_ohlcv)}")
                
                # ccxt's enableRateLimit should handle pauses, but a small extra sleep can't hurt
                # Especially if your connection is unstable or exchange is busy.
                time.sleep(0.5) # A small buffer
                
            except ccxt.base.errors.RequestTimeout as e:
                print(f"Timeout error fetching {symbol} data: {e}. Retrying after a longer delay...")
                time.sleep(10) # Wait for 10 seconds on a timeout before retrying
                continue # Try fetching the same block again
            except ccxt.base.errors.DDoSProtection as e:
                print(f"DDoS Protection error: {e}. Waiting longer...")
                time.sleep(self.exchange.rateLimit / 1000 + 5) # Use exchange's rate limit + 5 seconds
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Breaking loop for {symbol}.")
                break # Break on other unexpected errors to prevent infinite loops
        
        print(f"Finished collecting {symbol}. Total requests: {request_count}, Total candles: {len(all_ohlcv)}")
        if not all_ohlcv:
            print(f"Warning: No OHLCV data collected for {symbol}.")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Sort by timestamp to ensure chronological order (important if extending previous data)
        df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
        return df
    
    def collect_all_pairs(self):
        for pair in self.pairs:
            # Replace '/' with '_' for filename consistency
            filename = f'{self.data_dir}{pair.replace("/", "_")}_1m.csv'
            
            print(f"Processing {pair}...")
            df = self.fetch_ohlcv(pair)
            
            if not df.empty:
                df.to_csv(filename, index=False)
                print(f"Saved {len(df)} candles for {pair} to {filename}")
            else:
                print(f"Skipping save for {pair} as no data was collected.")


if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_all_pairs()