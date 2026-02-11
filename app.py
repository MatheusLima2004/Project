import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time
import random

# --- 1. SOVEREIGN ENGINE CLASS (Internalized) ---
class SovereignEngine:
    @staticmethod
    def get_tickers():
        try:
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            return sp500['Symbol'].tolist()[:100] # Scanning top 100 to ensure speed
        except:
            return ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "SPY", "QQQ"]

    @staticmethod
    def multi_node_mine(ticker_list):
        """Bypasses the 4-stock limit using Staggered Chunks"""
        all_data = {}
        chunk_size = 15 # Small nodes stay under the radar
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        for i in range(0, len(ticker_list), chunk_size):
            chunk = ticker_list[i:i + chunk_size]
            status_text.text(f"Mining Node: {i}-{i+chunk_size}")
            
            data = yf.download(chunk, period="1y", interval="1d", group_by='ticker', threads=True, progress=False)
            
            for t in chunk:
                if t in data: all_data[t] = data[t]
            
            # Update Progress
            progress_bar.progress(min((i + chunk_size) / len(ticker_list), 1.0))
            time.sleep(random.uniform(1.0, 2.5)) # The Sovereign Pulse
            
        return all_data

# --- 2. UI & MATRIX RENDER ---
st.set_page_config(page_title="Sovereign Command Matrix", layout="wide")

@st.fragment(run_every=900) # 15-Minute Cycle for maximum IP safety
def render_terminal():
    st.title("💹 Sovereign Matrix | Institutional Multi-Node")
    
    engine = SovereignEngine()
    tickers = engine.get_tickers()
    
    with st.spinner(f"Mining {len(tickers)} Assets across Multi-Node Clusters..."):
        data_mesh = engine.multi_node_mine(tickers)
        
        # Matrix Math
        results = []
        for t, df in data_mesh.items():
            try:
                if len(df) < 50: continue
                curr = df['Close'].iloc[-1]
                # Z-Score
                z = (curr - df['Close'].rolling(50).mean().iloc[-1]) / df['Close'].rolling(50).std().iloc[-1]
                # PoP %
                sigma = df['Close'].pct_change().std() * np.sqrt(252)
                results.append({"Symbol": t, "Price": curr, "Z-Score": z, "Volatility": sigma})
            except: continue

    if results:
        df_final = pd.DataFrame(results)
        st.dataframe(df_final.style.background_gradient(subset=['Z-Score'], cmap='RdYlGn_r'), use_container_width=True)
    else:
        st.error("Protocol Failure: Node Cluster Blocked. Check logs.")

render_terminal()
