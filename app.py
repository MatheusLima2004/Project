import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import time
import random
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ProTrader AI Terminal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    .stMetric { background-color: #1e2127; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    .stDataFrame { border: 1px solid #333; }
    div[data-testid="stExpander"] { background-color: #1e2127; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📡 ProTrader AI Terminal | Multi-Source Data Engine")

# --- 2. MATH & AI ENGINE ---
def generate_market_resume(df):
    if df.empty: return "Initializing..."
    spy = df[df['Ticker'] == 'SPY']
    if not spy.empty:
        change = spy['Change %'].values[0]
        sentiment = "BULLISH 🐂" if change > 0 else "BEARISH 🐻"
    else:
        sentiment = "NEUTRAL ⚖️"
        change = 0.0
    
    top = df.sort_values(by="Change %", ascending=False).iloc[0]
    return f"### 📝 Market Status: {sentiment} ({change:+.2f}%)\n* **Leader:** {top['Ticker']} ({top['Change %']:+.2f}%)"

def calculate_rsi(series, period=14):
    if len(series) < period: return 50
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_vwap(df):
    try:
        v = df['Volume'].values
        p = df['Close'].values
        return (p * v).cumsum() / v.cumsum()
    except: return pd.Series([0]*len(df))

def detect_stealth_algo(df):
    try:
        recent = df.tail(5)
        avg_vol = df['Volume'].mean()
        curr_vol = recent['Volume'].mean()
        price_change = abs(recent['Close'].pct_change().mean()) * 100
        
        if curr_vol > (avg_vol * 1.5) and price_change < 0.1: return "🧊 ICEBERG"
        elif curr_vol > (avg_vol * 2.0) and price_change > 1.5: return "🌊 SURGE"
        else: return "---"
    except: return "---"

def ai_analyst(row):
    rsi = row['RSI']
    algo = row['Algo Signal']
    if "ICEBERG" in algo and rsi < 50: return "🔥 STRONG BUY: Whale"
    if rsi < 30: return "✅ BUY DIP: Oversold"
    if rsi > 70: return "⚠️ SELL: Overbought"
    return "😴 HOLD"

# --- 3. ROBUST DATA ENGINE ---
def get_fallback_data(tickers):
    """Generates realistic placeholder data if API fails completely."""
    data = []
    for t in tickers:
        price = 150.0 + random.uniform(-5, 5)
        change = random.uniform(-2, 2)
        data.append({
            "Ticker": t, "Price": price, "Change %": change,
            "RSI": 50 + change * 5, "VWAP": price * 0.99,
            "Algo Signal": "---", "Target Price": price * 1.1,
            "Upside %": 10.0, "Fair Value": price * 0.8,
            "Headline": "Data Unavailable (Offline Mode)",
            "Link": "#", "Publisher": "System", "🤖 AI Tip": "⚠️ OFFLINE"
        })
    return pd.DataFrame(data)

@st.cache_data(ttl=60)
def fetch_market_data(stocks):
    data_points = []
    failed_tickers = 0
    
    for ticker in stocks:
        try:
            # 1. LIVE FETCH (Yahoo)
            stock = yf.Ticker(ticker)
            # Fetch minimal data to minimize blocking risk
            hist = stock.history(period="5d", interval="1d")
            
            if hist.empty:
                raise ValueError("Empty Data")

            # 2. CALCULATIONS
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else curr
            change = ((curr - prev) / prev) * 100
            rsi = calculate_rsi(hist['Close']).iloc[-1]
            
            # Simple Fundamental Check (failsafe)
            try:
                info = stock.info
                tgt = info.get('targetMeanPrice', 0)
                news = info.get('news', [{}])[0]
            except:
                tgt = 0
                news = {}

            row = {
                "Ticker": ticker,
                "Price": curr,
                "Change %": change,
                "RSI": rsi,
                "VWAP": curr, # Simplified for robustness
                "Algo Signal": "---", # Needs intraday, skipping for speed
                "Target Price": tgt,
                "Upside %": ((tgt - curr)/curr)*100 if tgt else 0,
                "Fair Value": 0, # Skipped to save API calls
                "Headline": news.get('title', 'No News'),
                "Link": news.get('link', '#'),
                "Publisher": news.get('publisher', 'N/A')
            }
            row["🤖 AI Tip"] = ai_analyst(row)
            data_points.append(row)
            
            time.sleep(0.2) # Throttling
            
        except Exception:
            failed_tickers += 1
            continue
    
    # If ALL failed, return Fallback
    if failed_tickers == len(stocks):
        return pd.DataFrame() # Trigger fallback in main loop
        
    return pd.DataFrame(data_points)

# --- 4. DASHBOARD RENDER ---
sidebar = st.sidebar
sidebar.header("⚙️ Control Panel")
live_mode = sidebar.toggle("🔴 LIVE MODE", value=True)
stock_list = ["NVDA", "TSLA", "AAPL", "AMD", "MSFT", "AMZN", "SPY", "QQQ"]

placeholder = st.empty()

while True:
    with placeholder.container():
        # ATTEMPT FETCH
        df = fetch_market_data(stock_list)
        
        # FALLBACK LOGIC
        if df.empty:
            st.warning("⚠️ Yahoo API Blocked/Down. Switched to OFFLINE MODE.")
            df = get_fallback_data(stock_list) # Use backup generator
        
        # RESUME
        st.info(generate_market_resume(df))
        
        # METRICS
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📅 Date", datetime.now().strftime("%Y-%m-%d"))
        
        spy_row = df[df['Ticker'] == 'SPY']
        if not spy_row.empty:
            c2.metric("SPY", f"${spy_row['Price'].values[0]:.2f}", f"{spy_row['Change %'].values[0]:.2f}%")
        else:
            c2.metric("Status", "Online")

        # TABS
        tab_tv, tab_main = st.tabs(["📺 Live Bloomberg TV", "🚀 AI Scanner"])
        
        with tab_tv:
            st.video("https://www.youtube.com/watch?v=dp8PhLsUcFE")
            
        with tab_main:
            st.dataframe(
                df.style.background_gradient(subset=['Change %'], cmap='RdYlGn', vmin=-3, vmax=3)
                .format({"Price": "${:.2f}", "Change %": "{:+.2f}%", "RSI": "{:.0f}"}),
                column_config={
                    "Link": st.column_config.LinkColumn("News"),
                    "🤖 AI Tip": st.column_config.TextColumn("AI Signal")
                },
                use_container_width=True, hide_index=True
            )

    if not live_mode: break
    time.sleep(60)
