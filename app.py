import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time
import random

# --- 1. CORE MATH ENGINE (GREEKS & MEDALLION) ---
class MedallionMath:
    @staticmethod
    def calculate_greeks(S, K, T, sigma, r=0.045):
        """Standard Black-Scholes Greeks"""
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            delta = si.norm.cdf(d1)
            theta = (-(S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d1 - sigma * np.sqrt(T))) / 252
            return delta, theta
        except: return 0, 0

    @staticmethod
    def run_analysis(df, ticker):
        """The Medallion Sovereign Matrix Logic"""
        try:
            if df is None or len(df) < 50: return None
            curr = df['Close'].iloc[-1]
            # Z-Score
            ma, std = df['Close'].rolling(50).mean().iloc[-1], df['Close'].rolling(50).std().iloc[-1]
            z = (curr - ma) / std
            # Probability of Profit (ML Projection)
            rets = df['Close'].pct_change().dropna()
            sigma = rets.std() * np.sqrt(252)
            y = df['Close'].tail(60).values
            model = LinearRegression().fit(np.arange(len(y)).reshape(-1, 1), y)
            target = model.predict([[len(y) + 30]])[0]
            
            # Greeks for 30D ATM Option
            delta, theta = MedallionMath.calculate_greeks(curr, curr, 30/365, sigma)
            
            return {
                "Symbol": ticker, "Price": curr, "Z-Score": z, 
                "Delta": delta, "Theta": theta, "ML Target": target, "Vol": sigma
            }
        except: return None

# --- 2. MULTI-NODE DATA MINER ---
def sovereign_mine(tickers):
    """The Multi-Node Scraper: Prevents IP blocks by staggering chunks"""
    all_data = {}
    chunk_size = 25
    # Progress UI must be inside the fragment body for stability
    p_text = st.empty()
    p_bar = st.progress(0)
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        p_text.markdown(f"📡 **Mining Node Cluster:** {i} of {len(tickers)} assets...")
        data = yf.download(chunk, period="1y", interval="1d", group_by='ticker', threads=True, progress=False)
        for t in chunk:
            if t in data and not data[t].dropna().empty: all_data[t] = data[t]
        p_bar.progress(min((i + chunk_size) / len(tickers), 1.0))
        time.sleep(random.uniform(1.5, 3.0)) # The Sovereign Pulse
    
    p_text.empty()
    p_bar.empty()
    return all_data

# --- 3. THE COMMAND INTERFACE ---
st.set_page_config(page_title="Sovereign Auto-Pilot Terminal", layout="wide")

@st.fragment(run_every=900)
def automated_terminal():
    st.title("💹 Medallion Sovereign | Full Market Auto-Pilot")
    
    # Auto-Scrape S&P 500 Node
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        ticker_list = sp500['Symbol'].tolist()
    except:
        ticker_list = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "AMZN", "META", "SPY", "QQQ"]

    st.sidebar.info(f"Last Full Scan: {datetime.now().strftime('%H:%M:%S')}")
    
    with st.status("Initializing Distributed Multi-Node Clusters...", expanded=True) as status:
        data_mesh = sovereign_mine(ticker_list)
        status.update(label="Matrix Synchronized. Vectorizing ML Models...", state="complete", expanded=False)
        
        results = []
        for t, df in data_mesh.items():
            res = MedallionMath.run_analysis(df, t)
            if res: results.append(res)

    if results:
        df_final = pd.DataFrame(results)
        st.subheader("🏛️ Global Command Matrix")
        st.dataframe(
            df_final.style.background_gradient(subset=['Z-Score'], cmap='RdYlGn_r')
            .format({"Price": "${:.2f}", "Z-Score": "{:.2f}", "Delta": "{:.2f}", "Theta": "{:.3f}", "ML Target": "${:.2f}", "Vol": "{:.1%}"}),
            use_container_width=True, hide_index=True, height=600
        )
    else:
        st.error("Connection Blocked. IP Refresh required in next cycle.")

automated_terminal()
