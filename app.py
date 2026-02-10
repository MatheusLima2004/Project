import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Master Terminal",
    page_icon="💎",
    layout="wide"
)

st.title("💎 The Master Terminal")
st.markdown("### Real-Time Value & Momentum Scanner")

# --- 2. SIDEBAR SETUP ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Default Watchlist
    default_tickers = """MELI, NU, PBR, BSBR, BBD, VALE, ITUB
NVDA, AAPL, MSFT, AMZN, GOOGL, TSLA, META
V, MA, JPM, KO, PEP, COST, MCD, DIS
AMD, PLTR, SOFI, UBER, ABNB, SHOP, NET
WEGE3.SA, PETR4.SA, VALE3.SA, ITUB4.SA, BBDC4.SA
O, SCHD, JEPI, T, VZ, MO"""

    ticker_input = st.text_area("Watchlist (Comma Separated)", default_tickers, height=300)
    
    # Process the input string into a list
    tickers = [t.strip() for t in ticker_input.replace('\n', ',').split(',') if t.strip()]
    
    st.info(f"Loaded {len(tickers)} stocks.")

# --- 3. HELPER FUNCTIONS ---
def generate_sparkline(series):
    """Generates a text-based sparkline graph"""
    bar_chars = [' ', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    if series.empty: return ""
    series = series.tail(30)
    min_val, max_val = series.min(), series.max()
    if max_val == min_val: return "▇" * 10
    spark = ""
    for price in series:
        idx = int((price - min_val) / (max_val - min_val) * 7)
        spark += bar_chars[idx]
    return spark

@st.cache_data(ttl=3600) # Cache data for 1 hour to prevent Yahoo bans
def scan_market(tickers):
    # Progress Bar
    progress_text = "Scanning Global Markets... Please wait."
    my_bar = st.progress(0, text=progress_text)
    
    try:
        # Bulk Download (Threads enabled for speed)
        history = yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        st.error(f"Error connecting to Yahoo Finance: {e}")
        return pd.DataFrame()

    results = []
    
    for i, ticker in enumerate(tickers):
        try:
            # Extract Data
            df = history[ticker] if len(tickers) > 1 else history
            if df.empty or len(df) < 50: continue
            
            close = df['Close'].dropna()
            current_price = close.iloc[-1]
            
            # Technicals
            ma_50 = close.rolling(window=50).mean().iloc[-1]
            std_50 = close.rolling(window=5
