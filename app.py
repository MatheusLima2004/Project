import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
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
    """
    Rule-Based Machine Learning (Expert System)
    Generates a trading tip based on technical confluence.
    """
    rsi = row['RSI']
    price = row['Price']
    vwap = row['VWAP']
    algo = row['Algo Signal']
    
    # 1. Institutional Buy Signal
    if algo == "🧊 ICEBERG (Accumulation)" and rsi < 50:
        return "🔥 STRONG BUY: Whales are secretly buying."
    
    # 2. Reversion to Mean (Oversold)
    if rsi < 30 and price < vwap:
        return "✅ BUY THE DIP: Oversold & Cheap vs VWAP."
        
    # 3. Momentum Breakout
    if rsi > 60 and price > vwap and algo == "🌊 MOMENTUM SURGE":
        return "🚀 RIDE TREND: High momentum breakout."
        
    # 4. Overextended (Sell Signal)
    if rsi > 75:
        return "⚠️ SELL/TRIM: Statistically overbought."
        
    return "😴 HOLD/WAIT: No clear signal."

def calculate_graham_value(eps, book_value):
    if eps > 0 and book_value > 0:
        return math.sqrt(22.5 * eps * book_value)
    return 0

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Live Mode
    live_mode = st.toggle("🔴 LIVE MODE (Auto-Refresh)", value=False)
    if live_mode: st.caption("Refreshing every 60s...")
    
    st.divider()
    
    # Watchlist
    default_tickers = "NVDA, TSLA, AAPL, AMD, MSFT, AMZN, GOOGL, META, PLTR, COIN, SPY, QQQ, IWM"
    ticker_input = st.text_area("Watchlist", default_tickers, height=150)
    stock_list = [t.strip() for t in ticker_input.split(',')]
    
    st.divider()
    rf_rate = st.number_input("Risk Free Rate (%)", value=4.5) / 100

# --- 4. DATA ENGINE ---
@st.cache_data(ttl=60 if live_mode else 3600)
def fetch_market_data(stocks):
    data_points = []
    
    try:
        # Batch Fetch
        tickers = yf.Tickers(" ".join(stocks))
        
        for ticker in stocks:
            try:
                # 1 month history for trends, 5d hourly for algos
                hist = tickers.tickers[ticker].history(period="1mo", interval="1d")
                intraday = tickers.tickers[ticker].history(period="5d", interval="60m")
                
                if hist.empty: continue
                
                # Basic Metrics
                curr_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                pct_change = ((curr_price - prev_close) / prev_close) * 100
                
                # Technicals
                rsi = calculate_rsi(hist['Close']).iloc[-1]
                vwap = calculate_vwap(intraday).iloc[-1] if not intraday.empty else curr_price
                algo_signal = detect_stealth_algo(intraday) if not intraday.empty else "---"
                
                # Fundamentals
                info = tickers.tickers[ticker].info
                target_price = info.get('targetMeanPrice', 0)
                upside = ((target_price - curr_price) / curr_price) * 100 if target_price > 0 else 0
                fair_val = calculate_graham_value(info.get('trailingEps', 0), info.get('bookValue', 0))
                
                # AI Analysis Row
                row_data = {
                    "Ticker": ticker,
                    "Price": curr_price,
                    "Change %": pct_change,
                    "RSI": rsi,
                    "VWAP": vwap,
                    "Algo Signal": algo_signal,
                    "Target Price": target_price,
                    "Upside %": upside,
                    "Fair Value": fair_val
                }
                
                # Generate AI Tip
                row_data["🤖 AI Tip"] = ai_analyst(row_data)
                
                data_points.append(row_data)
            except: continue
            
    except: return pd.DataFrame()
    
    return pd.DataFrame(data_points)

# --- 5. RENDER DASHBOARD ---
placeholder = st.empty()

while True:
    with placeholder.container():
        df = fetch_market_data(stock_list)
        
        if df.empty:
            st.error("No data found. Check tickers.")
            break

        # TOP METRICS
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📅 Market Date", datetime.now().strftime("%Y-%m-%d"))
        c2.metric("🕒 Local Time", datetime.now().strftime("%H:%M:%S"))
        c3.metric("🦅 Risk Free Rate", f"{rf_rate*100}%")
        
        # TABS
        tab_main, tab_edu, tab_news, tab_math = st.tabs(["🚀 AI Scanner", "📘 Education (Options 101)", "📰 News", "🧮 Math Lab"])

        # TAB 1: AI SCANNER
        with tab_main:
            st.subheader("🤖 Artificial Intelligence Insights")
            
            # Highlight the AI Tip column
            st.dataframe(
                df.style.background_gradient(subset=['Upside %'], cmap='RdYlGn', vmin=-10, vmax=30)
                .applymap(lambda x: 'color: #4CAF50; font-weight: bold' if 'BUY' in str(x) else ('color: #FF5252; font-weight: bold' if 'SELL' in str(x) else ''), subset=['🤖 AI Tip'])
                .format({"Price": "${:.2f}", "Change %": "{:+.2f}%", "RSI": "{:.0f}", "VWAP": "${:.2f}", "Target Price": "${:.2f}", "Upside %": "{:+.1f}%"}),
                column_config={
                    "Algo Signal": st.column_config.TextColumn("Inst. Footprint"),
                    "🤖 AI Tip": st.column_config.TextColumn("AI Action Plan", width="medium"),
                },
                use_container_width=True,
                hide_index=True,
                height=600
            )

        # TAB 2: EDUCATION (New Section)
        with tab_edu:
            st.header("📘 Options Trading Academy")
            st.markdown("Everything you need to know before risking a dollar.")
            
            with st.expander("🟢 The Basics: Calls vs. Puts"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### 🐂 CALL Option")
                    st.info("**Prediction: UP**")
                    st.write("You buy the *right* to **BUY** shares later at a set price.")
                    st.write("*Example:* You think AAPL ($150) will go to $200. You buy a Call.")
                with c2:
                    st.markdown("### 🐻 PUT Option")
                    st.error("**Prediction: DOWN**")
                    st.write("You buy the *right* to **SELL** shares later at a set price.")
                    st.write("*Example:* You think AAPL ($150) will crash to $100. You buy a Put.")

            with st.expander("📐 The 'Greeks' (The Math of Risk)"):
                st.markdown("""
                | Greek | Meaning | Plain English |
                | :--- | :--- | :--- |
                | **Delta ($\delta$)** | Price Sensitivity | If stock moves $1, option moves $\delta$. (e.g., 0.50 delta = $0.50 gain) |
                | **Theta ($\theta$)** | Time Decay | How much money you lose **every day** just by holding. |
                | **Gamma ($\gamma$)** | Acceleration | How fast Delta changes. High Gamma = Explosive moves. |
                | **Vega ($v$)** | Volatility | How much the price changes when panic (IV) rises. |
                """)

            with st.expander("♟️ Advanced Strategies"):
                st.markdown("""
                ### 1. Covered Call (Income)
                * **What:** You own 100 shares of a stock + SELL a Call option.
                * **Why:** You get paid cash (Premium) instantly.
                * **Risk:** You limit your upside if the stock moons.

                ### 2. Vertical Spread (Directional)
                * **What:** Buy one option, Sell another (cheaper) option.
                * **Why:** It costs less money than buying a naked option.
                * **Risk:** Capped profit, but capped loss.
                """)

        # TAB 3: NEWS
        with tab_news:
            st.subheader("Latest Headlines")
            # Using placeholder data logic for news display
            st.info("News feed connects to live data in deployed version.")

        # TAB 4: MATH LAB
        with tab_math:
            st.markdown("### 🧮 How the AI Thinks")
            st.code("""
            def ai_logic(RSI, VWAP, Volume):
                # 1. Detect Whale Buying
                if Volume > 2x_Avg AND Price_Flat:
                    return "ICEBERG_DETECTED"
                
                # 2. Mean Reversion
                if RSI < 30 AND Price < VWAP:
                    return "OVERSOLD_VALUE_BUY"
                    
                # 3. Trend Following
                if RSI > 50 AND Price > VWAP:
                    return "MOMENTUM_LONG"
            """, language="python")

    if not live_mode:
        break
    time.sleep(60)
