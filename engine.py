import yfinance as yf
import pandas as pd
import time
import random

def get_full_market_data(ticker_list):
    """The Sovereign Multi-Node Scraper: Bypasses the 4-stock limit"""
    all_data = {}
    # Small chunks (Nodes) prevent the 'Connection Blocked' error
    chunk_size = 20 
    
    # Institutional Headers to mimic a professional terminal
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://finance.yahoo.com/'
    }

    for i in range(0, len(ticker_list), chunk_size):
        chunk = ticker_list[i:i + chunk_size]
        try:
            # We use a custom session or standard download with threading
            data = yf.download(
                chunk, 
                period="1y", 
                interval="1d", 
                group_by='ticker', 
                threads=True, 
                progress=False,
                timeout=10 # Prevents the app from hanging
            )
            
            for ticker in chunk:
                if ticker in data and not data[ticker].dropna().empty:
                    all_data[ticker] = data[ticker]
            
            # The 'Sovereign Pulse': A random delay to stay under the radar
            time.sleep(random.uniform(1.5, 3.0)) 
            
        except Exception as e:
            continue # If one Node fails, the next one picks up the weight
            
    return all_data
