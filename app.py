import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Cloud Medallion", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2127; border: 1px solid #333; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Medallion Cloud Scanner")
st.info("💡 **Note:** To prevent being blocked by Yahoo, this version scans stocks one by one with a delay.")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("Watchlist")
    # Using a smaller list by default to ensure it loads successfully
    tickers = st.text_area("Tickers (Comma separated)", "SPY, QQQ, NVDA, AAPL, TSLA", height=150)
    ticker_list = [x.strip() for x in tickers.split(',')]
    start_btn = st.button("🚀 START CLOUD SCAN", type="primary")

# --- 3. DATA ENGINE ---
if start_btn:
    results = []
    progress_bar = st.progress(0)
    
    for i, t in enumerate(ticker_list):
        try:
            # We use a smaller time window (5 days) to be "quieter" on the API
            stock = yf.Ticker(t)
            hist = stock.history(period="5d")
            
            if not hist.empty:
                curr = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((curr - prev) / prev) * 100
                
                # AI Logic
                tip = "HOLD"
                if change < -2: tip = "✅ BUY DIP"
                elif change > 2: tip = "⚠️ SELL/TRIM"
                
                results.append({
                    "Ticker": t,
                    "Price": curr,
                    "Change %": change,
                    "Signal": tip
                })
            
            # CRUCIAL: Wait 1 second between stocks so Yahoo doesn't block the cloud server
            time.sleep(1.5)
            
        except Exception as e:
            st.error(f"Error loading {t}. Yahoo may be temporarily blocking requests.")
            
        progress_bar.progress((i + 1) / len(ticker_list))

    if results:
        df = pd.DataFrame(results)
        st.subheader("📊 Market Intelligence")
        st.dataframe(
            df.style.background_gradient(subset=['Change %'], cmap='RdYlGn'),
            use_container_width=True, hide_index=True
        )
    else:
        st.error("Connection Failed. Yahoo is currently blocking this cloud server. Try again in 10 minutes.")

else:
    st.markdown("### 👋 Welcome back.")
    st.write("1. Update your tickers in the sidebar.")
    st.write("2. Click **Start Cloud Scan**.")
    st.write("3. If it fails, wait a few minutes—Yahoo is likely rate-limiting the server.")
