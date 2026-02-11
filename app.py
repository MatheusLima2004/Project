import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
import math
from datetime import datetime

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Safe-Batch Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2127; border: 1px solid #333; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Medallion Batch-Scanner (Safe Mode)")

# --- 2. THE BIG LIST ---
# This list can be expanded to 500+ without getting banned using the Batch method
def get_sp500_tickers():
    # You can paste the full S&P 500 here. For now, here are the top 60.
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "V", 
        "JPM", "JNJ", "WMT", "MA", "PG", "HD", "CVX", "LLY", "ABBV", "PFE", 
        "COST", "PEP", "KO", "ORCL", "BAC", "AVGO", "TMO", "CSCO", "ACN", "ADBE",
        "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "ABEV3.SA", 
        "WEGE3.SA", "B3SA3.SA", "RENT3.SA", "SUZB3.SA", "GGBR4.SA", "JBSS3.SA",
        "SPY", "QQQ", "IWM", "EEM", "GLD", "SLV", "DIA", "XLE", "XLF", "XLK"
    ]

# --- 3. THE ALGORITHMS ---
def black_scholes(S, K, T, r, sigma):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return max(S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2), 0)
    except: return 0

# --- 4. DATA ENGINE (The Safe Way) ---
@st.cache_data(ttl=3600)
def run_safe_scan(ticker_list, rf):
    # DOWNLOAD ALL DATA AT ONCE
    # This is the secret to not getting banned.
    data = yf.download(ticker_list, period="60d", interval="1d", group_by='ticker', progress=False)
    
    results = []
    for ticker in ticker_list:
        try:
            df = data[ticker].dropna()
            if df.empty: continue
            
            curr = df['Close'].iloc[-1]
            # Z-Score Algorithm
            ma = df['Close'].mean()
            std = df['Close'].std()
            z_score = (curr - ma) / std if std > 0 else 0
            
            # Volatility Algorithm
            sigma = df['Close'].pct_change().std() * np.sqrt(252)
            
            # Kelly Criterion Algorithm
            # Probability of win increases as Z-Score hits extremes
            win_prob = 0.5 + (abs(z_score) * 0.04) 
            kelly = (win_prob * 3 - 1) / 2 # simplified Kelly
            
            results.append({
                "Ticker": ticker,
                "Price": curr,
                "Z-Score": z_score,
                "Kelly %": max(0, kelly * 100),
                "Opt Fair Val": black_scholes(curr, curr*1.05, 0.08, rf, sigma),
                "Signal": "🟢 BUY" if z_score < -1.8 else ("🔴 SELL" if z_score > 1.8 else "😴 HOLD")
            })
        except: continue
    return pd.DataFrame(results)

# --- 5. RENDER ---
tickers = get_sp500_tickers()
rf_rate = 0.045

if st.button(f"🚀 RUN ALGORITHMIC SCAN ON {len(tickers)} ASSETS"):
    with st.spinner("Executing Medallion Batch Algos..."):
        df = run_safe_scan(tickers, rf_rate)
    
    if not df.empty:
        df = df.sort_values("Z-Score")
        st.subheader("📊 Algorithmic Scanner Results")
        st.dataframe(
            df.style.background_gradient(subset=['Z-Score'], cmap='RdYlGn_r')
            .format({"Price": "${:.2f}", "Z-Score": "{:.2f}", "Kelly %": "{:.1f}%", "Opt Fair Val": "${:.2f}"}),
            use_container_width=True, hide_index=True
        )
    else:
        st.error("Connection Interrupted. Please wait 5 minutes and retry.")
else:
    st.info("The scanner is ready. Click the button to run the algorithms without getting banned.")
