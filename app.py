import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import time

# --- 1. QUANTITATIVE NODES (THE MATH) ---
class QuantEngine:
    @staticmethod
    def calculate_pop(S, K, T, sigma, r=0.045):
        """Probability of Profit using Black-Scholes d2"""
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return si.norm.cdf(abs(d2)) * 100
        except: return 0

    @staticmethod
    def process_ticker(ticker, df):
        """Vectorized Institutional Analysis for a single ticker"""
        try:
            if df is None or len(df) < 60: return None
            curr = df['Close'].iloc[-1]
            
            # Z-Score Mean Reversion
            ma_50 = df['Close'].rolling(50).mean()
            std_50 = df['Close'].rolling(50).std()
            z = (curr - ma_50.iloc[-1]) / std_50.iloc[-1]
            
            # ML Projection (30 Days)
            y = df['Close'].tail(60).values
            x = np.arange(len(y)).reshape(-1, 1)
            target = LinearRegression().fit(x, y).predict([[len(y) + 30]])[0]
            
            # Options Intelligence
            rets = df['Close'].pct_change().dropna()
            sigma = rets.std() * np.sqrt(252)
            pop = QuantEngine.calculate_pop(curr, target, 30/365, sigma)
            
            return {
                "Symbol": ticker, "Price": curr, "Z-Score": z, 
                "PoP %": pop, "ML Target": target, "Vol": sigma
            }
        except: return None

# --- 2. COMMAND MATRIX RENDER ---
st.set_page_config(page_title="Hyper-Sovereign Terminal", layout="wide")

@st.fragment(run_every=720) # 12-Minute Cycle
def sovereign_matrix():
    st.title("💹 Hyper-Sovereign | Institutional HFT Matrix")
    
    # Scrape the Universe (S&P 500 Node)
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    
    with st.status("Deploying Multi-Threaded Quant Nodes...", expanded=True) as status:
        # Step 1: Bulk Download (Mining Node)
        df_mega = yf.download(tickers, period="2y", interval="1d", group_by='ticker', threads=True, progress=False)
        
        # Step 2: Parallel Processing (Quant Node)
        # We use ThreadPoolExecutor to run math for all 500 stocks concurrently
        results = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(QuantEngine.process_ticker, t, df_mega[t]) for t in tickers if t in df_mega]
            results = [f.result() for f in futures if f.result() is not None]
            
        status.update(label=f"Scan Complete: {len(results)} Assets Analyzed.", state="complete", expanded=False)

    if results:
        df_final = pd.DataFrame(results)
        st.dataframe(
            df_final.sort_values(by="Z-Score").style.background_gradient(subset=['PoP %'], cmap='RdYlGn')
            .format({"Price": "${:.2f}", "Z-Score": "{:.2f}", "PoP %": "{:.1f}%", "ML Target": "${:.2f}", "Vol": "{:.1%}"}),
            use_container_width=True, hide_index=True, height=600
        )
    else:
        st.error("Protocol Failure: Node Cluster Unresponsive.")

sovereign_matrix()
