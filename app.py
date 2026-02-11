import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import time
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ProTrader AI Terminal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom "Fidelity-Dark" CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    .stMetric { background-color: #1e2127; border: 1px solid #333; padding: 15px; border-radius: 8px; }
    .stDataFrame { border: 1px solid #333; }
    div[data-testid="stExpander"] { background-color: #1e2127; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 ProTrader AI Terminal | Intelligent Market Analysis")

# --- 2. MATH & AI ENGINE ---

def calculate_rsi(series, period=14):
    """Relative Strength Index (Momentum)"""
    if len(series) < period: return 50
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_vwap(df):
    """Volume Weighted Average Price (Institutional Price)"""
    try:
        v = df['Volume'].values
        p = df['Close'].values
        return (p * v).cumsum() / v.cumsum()
    except: return pd.Series([0]*len(df))

def detect_stealth_algo(df):
    """Detects 'Iceberg' orders (High Vol, Low Price Move)"""
    try:
        recent = df.tail(5)
        avg_vol = df['Volume'].mean()
        curr_vol = recent['Volume'].mean()
        price_change = abs(recent['Close'].pct_change().mean()) * 100
        
        if curr_vol > (avg_vol * 1.5) and price_change < 0.1:
            return "🧊 ICEBERG (Accumulation)"
        elif curr_vol > (avg_vol * 2.0) and price_change > 1.5:
            return "🌊 MOMENTUM SURGE"
        else:
            return "---"
    except: return "---"

def ai_analyst(row):
    """Generates a trading tip based on technical confluence."""
    rsi = row['RSI']
    price = row['Price']
    vwap = row['VWAP']
    algo = row['Algo Signal']
    
    if algo == "🧊 ICEBERG (Accumulation)" and rsi < 50:
        return "🔥 STRONG BUY: Whale detected."
    if rsi < 30 and price < vwap:
        return "✅ BUY THE DIP: Oversold."
    if rsi > 60 and price > vwap and algo == "🌊 MOMENTUM SURGE":
        return "🚀 RIDE TREND: Breakout."
    if rsi > 75:
        return "⚠️ SELL/TRIM: Overbought."
    return "😴 HOLD/WAIT"

def calculate_graham_value(eps, book_value):
    if eps and book_value and eps > 0 and book_value > 0:
        return math.sqrt(22.5 * eps * book_value)
    return 0

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    live_mode = st.toggle("🔴 LIVE MODE (Auto-Refresh)", value=False)
    if live_mode: st.caption("Refreshing every 60s...")
    
    st.divider()
    
    # Cleaned Watchlist to prevent errors
    default_tickers = "NVDA, TSLA, AAPL, AMD, MSFT, AMZN, GOOGL, META, PLTR, SPY, QQQ"
    ticker_input = st.text_area("Watchlist", default_tickers, height=150)
    # Filter out empty strings
    stock_list = [t.strip() for t in ticker_input.replace('\n', ',').split(',') if t.strip()]
    
    st.divider()
    rf_rate = st.number_input("Risk Free Rate (%)", value=4.5) / 100

# --- 4. DATA ENGINE (ROBUST VERSION) ---
@st.cache_data(ttl=60 if live_mode else 3600)
def fetch_market_data(stocks):
    data_points = []
    
    # Using individual Ticker calls instead of batch to prevent "No Data" errors
    for ticker in stocks:
        try:
            stock = yf.Ticker(ticker)
            
            # 1. Get History (1mo for trend, 5d for algo)
            hist = stock.history(period="1mo", interval="1d")
            intraday = stock.history(period="5d", interval="60m")
            
            if hist.empty: continue
            
            # 2. Basic Metrics
            curr_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else curr_price
            pct_change = ((curr_price - prev_close) / prev_close) * 100
            
            # 3. Technicals
            rsi = calculate_rsi(hist['Close']).iloc[-1]
            vwap = calculate_vwap(intraday).iloc[-1] if not intraday.empty else curr_price
            algo_signal = detect_stealth_algo(intraday) if not intraday.empty else "---"
            
            # 4. Fundamentals (Handle missing keys gracefully)
            info = stock.info
            target_price = info.get('targetMeanPrice', 0)
            
            # Safety check for Target Price
            if target_price and target_price > 0:
                upside = ((target_price - curr_price) / curr_price) * 100
            else:
                upside = 0
                
            fair_val = calculate_graham_value(info.get('trailingEps', 0), info.get('bookValue', 0))
            
            # 5. News (Top Headline)
            news_title = "No recent news"
            news_link = "#"
            if 'news' in info and len(info['news']) > 0:
                news_title = info['news'][0].get('title', 'News')
                news_link = info['news'][0].get('link', '#')

            # 6. Build Row
            row_data = {
                "Ticker": ticker,
                "Price": curr_price,
                "Change %": pct_change,
                "RSI": rsi,
                "VWAP": vwap,
                "Algo Signal": algo_signal,
                "Target Price": target_price,
                "Upside %": upside,
                "Fair Value": fair_val,
                "Headline": news_title
            }
            
            # 7. AI Tip
            row_data["🤖 AI Tip"] = ai_analyst(row_data)
            
            data_points.append(row_data)
            
        except Exception:
            continue
            
    return pd.DataFrame(data_points)

# --- 5. RENDER DASHBOARD ---
placeholder = st.empty()

while True:
    with placeholder.container():
        # Fetch Data
        df = fetch_market_data(stock_list)
        
        if df.empty:
            st.warning("⚠️ No data loaded yet. If this persists, check if the markets are open or simplify your watchlist.")
        else:
            # METRICS ROW
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📅 Date", datetime.now().strftime("%Y-%m-%d"))
            c2.metric("🕒 Time", datetime.now().strftime("%H:%M:%S"))
            c3.metric("🦅 Risk Free Rate", f"{rf_rate*1
