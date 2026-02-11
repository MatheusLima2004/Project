import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time

# --- 1. THE MEGA-WATCHLIST ---
def get_global_universe():
    # US Markets (S&P 500 / Nasdaq 100 leaders)
    us_leaders = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "AVGO", "V", "MA",
        "JPM", "BAC", "UNH", "COST", "LLY", "HD", "PG", "NFLX", "AMD", "PLTR",
        "ADBE", "CRM", "ORCL", "CSCO", "INTC", "QCOM", "TXN", "AMAT", "MU", "ISRG"
    ]
    # Brazilian Markets (B3)
    br_leaders = [
        "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "ABEV3.SA", 
        "WEGE3.SA", "B3SA3.SA", "RENT3.SA", "SUZB3.SA", "GGBR4.SA", "JBSS3.SA",
        "RAIL3.SA", "EQTL3.SA", "VIVT3.SA", "PRIO3.SA", "LREN3.SA", "RDOR3.SA"
    ]
    # User's Personal Portfolio Focus
    personal_focus = ["MELI", "NU", "PBR", "BSBR", "BBD", "VGT", "VOO"]
    # ETFs & Indices
    indices = ["SPY", "QQQ", "IWM", "EWZ", "GLD", "SLV"]
    
    return list(set(us_leaders + br_leaders + personal_focus + indices))

# --- 2. MULTI-SOURCE ALGO ENGINE ---
@st.cache_data(ttl=3600)
def run_global_analysis(ticker_list):
    # STEP 1: BATCH DOWNLOAD (Avoids the Ban)
    # We download 60 days of hourly data for hundreds of stocks in ONE call
    data = yf.download(ticker_list, period="60d", interval="1h", group_by='ticker', threads=True, progress=False)
    
    results = []
    for ticker in ticker_list:
        try:
            df = data[ticker].dropna()
            if df.empty or len(df) < 30: continue

            # --- VWAP ---
            df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            # --- TOC SQUEEZE ---
            sma = df['Close'].rolling(20).mean()
            std = df['Close'].rolling(20).std()
            atr = (df['High'] - df['Low']).rolling(20).mean()
            squeeze = "🔒 SQUEEZE" if (sma - (2*std) > sma - (1.5*atr)).iloc[-1] else "🌊 EXPANSION"

            # --- ML LINEAR FORECAST ---
            y = df['Close'].values
            x = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(x, y)
            forecast = model.predict([[len(y) + 1]])[0]

            # --- PROBABILITY OF SUCCESS ---
            sigma = df['Close'].pct_change().std() * np.sqrt(252 * 7)
            dist = np.log(forecast / df['Close'].iloc[-1])
            pop = si.norm.cdf(dist / (sigma * np.sqrt(1/12))) * 100

            results.append({
                "Ticker": ticker,
                "Price": df['Close'].iloc[-1],
                "VWAP": df['vwap'].iloc[-1],
                "TOC": squeeze,
                "ML Target": forecast,
                "Success %": pop
            })
        except: continue
    return pd.DataFrame(results)

# --- 3. UI RENDER ---
st.set_page_config(page_title="Fidelity Global Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0d1217; color: #e1e4e8; }
    .stMetric { background-color: #161c23; border: 1px solid #30363d; padding: 15px; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

st.title("💹 Fidelity Pro Terminal | Global Scale Intelligence")

# AUTO-REFRESH WRAPPER
@st.fragment(run_every=65)
def terminal_fragment(universe):
    st.write(f"🕒 **Last System Sync:** {datetime.now().strftime('%H:%M:%S')}")
    
    if st.button(f"🚀 RUN ANALYSIS ON {len(universe)} ASSETS"):
        with st.spinner("Processing Global Market Matrix..."):
            df = run_global_analysis(universe)
            
            if not df.empty:
                c1, c2, c3 = st.columns(3)
                c1.metric("Assets Analyzed", len(df))
                c2.metric("TOC Squeeze Alerts", len(df[df['TOC'] == "🔒 SQUEEZE"]))
                c3.metric("Avg Success %", f"{df['Success %'].mean():.1f}%")

                st.subheader("📊 Elite Algorithm Scanner")
                st.dataframe(
                    df.style.background_gradient(subset=['Success %'], cmap='RdYlGn')
                    .format({"Price": "${:.2f}", "VWAP": "${:.2f}", "ML Target": "${:.2f}", "Success %": "{:.1f}%"}),
                    use_container_width=True, hide_index=True, height=600
                )
            else:
                st.error("Connection Interrupted. Try reducing ticker count or check API status.")

# SIDEBAR
with st.sidebar:
    st.header("Terminal Control")
    st.info("System set to 'Global Universe' (300+ Stocks).")
    universe = get_global_universe()
    st.write(f"Active Tickers: {len(universe)}")

terminal_fragment(universe)
